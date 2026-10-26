"""12-bit lossless BF16 weights for skinny GEMMs: the format, and the kernel on the simulator.

python -m h200sim.bf16c [--n 2048] [--k 4096] [--m 16] [--warps 8] [--split 4] [--u 3]

A BF16 value is [S E7..E1 | E0 M6..M0]. Its low byte is stored as is (8 bits). The high byte,
sign plus E >> 1, takes 4 bits: the sign and a 3-bit index idx = (E >> 1) - b into a window of
8 values [b, b + 7] (16 exponents, 2b to 2b + 15) chosen per matrix to hold the most elements.
Decoding writes the high byte as S << 7 | (b' | idx) with b' = b & ~7, an OR, which is the
weight times 2^-2d, d = b & 7: the kernel scales its FP32 result by 2^2d, exactly. Elements
outside the window (~0.03 % for normal weights, and any zero) are escapes: stored with idx 0,
and corrected exactly in FP32 from a side list holding their true value.

Fragment order, as the router kernel: a 16-row x 64-k tile, lane (g, c) holding rows g + 8h
and k 32q + 8c + e (h, q in 0..1, e in 0..7): 32 elements, 3 x 16 bytes per lane, so a tile is
3 warp loads of 512 contiguous bytes (1536 B, against 2048 B in BF16):
- uint4 h (0, 1): the low bytes of row g + 8h, byte 8q + e.
- uint4 2: word 2h + q is the code word of (h, q). Its byte j holds, from bit 7 down: S of
  element j, idx of element 4 + (j + 1) % 4, S of element 4 + j, idx of element j. Then
  (word & 0x87878787) | b' gives the high bytes of elements 0..3, and the same on the word
  rotated left by 4 those of elements 4..7: one LOP3, one SHF, one LOP3 for 8 elements, then
  one PRMT per pair of elements builds the BF16x2 register mma.sync takes.
"""

import argparse
import time

import numpy as np

from . import H200, Sim, from_bf16_bits, to_bf16_bits
from .isa import Asm

LUT_AND_OR = 0xEA  # (a & b) | c


def window(bits: np.ndarray) -> int:
    """b: the window [b, b + 7] of E >> 1 holding the most elements (E > 0)."""
    eh = (bits.astype(np.int64) >> 8) & 0x7F
    cnt = np.bincount(eh.ravel(), minlength=136)
    cnt[0] -= int((bits & 0x7F80 == 0).sum())  # zeros and subnormals are escapes anyway
    best = max(range(0, 121), key=lambda b: cnt[b:b + 8].sum())
    return best


def pack(bits: np.ndarray):
    """bits: uint16 [N, K] BF16. Returns (packed uint8 [N/16, K/64, 3, 32, 16], b, escapes) where
    escapes is a list of (row, k, true bits, decoded bits)."""
    n, k = bits.shape
    assert n % 16 == 0 and k % 64 == 0
    b = window(bits)
    bp, d = b & ~7, b & 7
    s = (bits >> 15) & 1
    eh = (bits >> 8) & 0x7F
    lo = (bits & 0xFF).astype(np.uint8)
    e = (bits >> 7) & 0xFF
    ok = (eh >= b) & (eh <= b + 7) & (e > 0)
    idx = np.where(ok, eh.astype(np.int64) - b, 0).astype(np.uint8)
    dec = ((s.astype(np.uint16) << 15) | ((bp + idx).astype(np.uint16) << 8) | lo).astype(np.uint16)
    rows, cols = np.nonzero(~ok)
    esc = [(int(r), int(c), int(bits[r, c]), int(dec[r, c])) for r, c in zip(rows, cols)]

    def frag(a):  # [N, K] -> [rt, kt, h, g, c, q, e]
        return a.reshape(n // 16, 2, 8, k // 64, 2, 4, 8).transpose(0, 3, 1, 2, 5, 4, 6)
    L, S, I = frag(lo), frag(s.astype(np.uint8)), frag(idx)
    out = np.zeros((n // 16, k // 64, 3, 32, 16), np.uint8)
    # uint4 h: byte 8q + e of lane 4g + c
    out[:, :, 0:2] = L.transpose(0, 1, 2, 3, 4, 5, 6).reshape(n // 16, k // 64, 2, 32, 16)
    # uint4 2: word 2h + q, byte j
    Sa, Sb, Ia, Ib = S[..., 0:4], S[..., 4:8], I[..., 0:4], I[..., 4:8]
    code = (Sa << 7) | (np.roll(Ib, -1, axis=-1) << 4) | (Sb << 3) | Ia  # [rt, kt, h, g, c, q, j]
    out[:, :, 2] = code.transpose(0, 1, 3, 4, 2, 5, 6).reshape(n // 16, k // 64, 32, 16)
    return out, b, esc


def unpack(packed: np.ndarray, b: int) -> np.ndarray:
    """The decode the kernel does, in numpy: uint16 [N, K] of W * 2^-2d (escapes not corrected)."""
    rt, kt = packed.shape[:2]
    bp = b & ~7
    w = packed.view(np.uint32).reshape(rt, kt, 3, 32, 4)
    out = np.zeros((rt, kt, 2, 32, 2, 8), np.uint16)  # rt, kt, h, lane, q, e
    for h in range(2):
        for q in range(2):
            nw = w[:, :, 2, :, 2 * h + q]
            rot = (nw << 4) | (nw >> 28)
            for grp, word in ((0, nw), (1, rot)):
                hh = (word & 0x87878787) | (bp * 0x01010101)
                lw = w[:, :, h, :, 2 * q + grp]
                for j in range(4):
                    out[:, :, h, :, q, 4 * grp + j] = (((hh >> (8 * j)) & 0xFF) << 8) | ((lw >> (8 * j)) & 0xFF)
    o = out.reshape(rt, kt, 2, 8, 4, 2, 8).transpose(0, 2, 3, 1, 5, 4, 6)  # rt, h, g, kt, q, c, e
    return o.reshape(rt * 16, kt * 64)


def frag_bf16(bits: np.ndarray) -> np.ndarray:
    """Uncompressed BF16 in the same fragment order: [N/16, K/64, 4 loads (2h + q), 32, 8]."""
    n, k = bits.shape
    v = bits.reshape(n // 16, 2, 8, k // 64, 2, 4, 8)  # rt, h, g, kt, q, c, e
    return v.transpose(0, 3, 1, 4, 2, 5, 6).copy()


def skinny_kernel(K: int, NT: int, warps: int, ktc: int, U: int, compressed: bool, bp: int = 0):
    """out[m, N] += x[m, K] @ W'[N, K]^T over one k slice per CTA, fp32 red.add (split K).

    CTA (rb, s): warp w owns row tile rb * warps + w, k tiles s * ktc .. + ktc, U tiles of
    weights in flight in registers; x (8 NT rows) comes through L1, one tile ahead.
    params: W, x, out, nrb (row blocks)."""
    assert ktc % U == 0 and ktc // U >= 2
    nl = 3 if compressed else 4                 # 16 B loads per lane per tile
    tile_b = nl * 512
    KT = K // 64
    a = Asm("skinny_w12" if compressed else "skinny_bf16")
    lane, wid, cta, t, t2, g, c = (a.reg() for _ in range(7))
    wp, xp, i, p = a.reg(), a.reg(), a.reg(), a.preg()
    def vec(n):  # n registers, 4-aligned for 16 B loads
        a.nxt = (a.nxt + 3) // 4 * 4
        return a.reg(n)
    W = [vec(4 * nl) for _ in range(U)]
    X = [vec(8 * NT) for _ in range(2)]         # x of this tile and the next
    acc = [a.reg(4) for _ in range(NT)]
    A = [a.reg(4) for _ in range(4)]
    m87, bb, hA, rot, hB = (a.reg() for _ in range(5))

    a.s2r(lane, "laneid")
    a.s2r(wid, "warpid")
    a.s2r(cta, "ctaid")
    a.shr(g, lane, 2)
    a.and_(c, lane, 3)
    # s = cta / nrb (multiply-high by floor(2^32 / nrb) + 1), rb = cta - s * nrb
    rb, s_ = a.reg(), a.reg()
    a.ldc(s_, "inv_nrb")
    a.imul_hi(s_, cta, s_)
    a.ldc(t, "neg_nrb")
    a.imad(rb, s_, t, cta)
    rt = a.reg()
    a.imad(rt, rb, warps, wid)
    # wp = W + ((rt * KT + s * ktc) * nl * 32 + lane) * 16
    a.imad(t, s_, ktc, 0)
    a.imad(t, rt, KT, t)
    a.imad(t, t, nl * 32, lane)
    a.shl(t, t, 4)
    a.ldc(wp, "W")
    a.iadd(wp, wp, t)
    # xp = x + (g * K + s * ktc * 64 + 8 c) * 2; token tile tt adds 8 tt rows
    a.imad(t, g, K, 0)
    a.imad(t2, s_, ktc * 64, t)
    a.imad(t2, c, 8, t2)
    a.shl(t2, t2, 1)
    a.ldc(xp, "x")
    a.iadd(xp, xp, t2)
    for r in acc:
        for e in range(4):
            a.mov(r + e, 0)
    if compressed:
        a.mov(m87, 0x87878787)
        a.mov(bb, bp * 0x01010101)

    def load_w(u):
        for l in range(nl):
            a.ldg(W[u] + 4 * l, wp, l * 512, 16, cache="cs")
        a.iadd(wp, wp, tile_b)

    def load_x(slot):
        for tt in range(NT):
            for q in range(2):
                a.ldg(X[slot] + 8 * tt + 4 * q, xp, (8 * tt * K) * 2 + 64 * q, 16, cache="ca")
        a.iadd(xp, xp, 128)

    def compute(u, xs):
        w = W[u]
        for q in range(2):
            for half in range(2):
                if compressed:
                    if half == 0:
                        # code words (h, q) for h = 0, 1; high bytes of elements 0..3 / 4..7
                        pass
                    for h in range(2):
                        nw = w + 8 + 2 * h + q
                        lw = w + 4 * h + 2 * q + half
                        if half == 0:
                            a.lop3(hA, nw, m87, bb, LUT_AND_OR)
                            src = hA
                        else:
                            a.shf(rot, nw, nw, 4)
                            a.lop3(hB, rot, m87, bb, LUT_AND_OR)
                            src = hB
                        a.prmt(A[2 * q + half] + h, lw, src, 0x5140)
                        a.prmt(A[2 * q + half] + 2 + h, lw, src, 0x7362)
                    areg = A[2 * q + half]
                else:
                    # a0 = (h0, q, w0), a1 = (h1, q, w0), a2 = (h0, q, w0 + 1), a3 = (h1, q, w0 + 1)
                    w0 = 2 * half
                    areg = A[2 * q + half]
                    a.mov(areg + 0, w + 4 * q + w0)
                    a.mov(areg + 1, w + 4 * (2 + q) + w0)
                    a.mov(areg + 2, w + 4 * q + w0 + 1)
                    a.mov(areg + 3, w + 4 * (2 + q) + w0 + 1)
                for tt in range(NT):
                    a.hmma(acc[tt], areg, X[xs] + 8 * tt + 4 * q + 2 * half, acc[tt])

    for u in range(U):
        load_w(u)
    load_x(0)
    iters = ktc // U
    a.mov(i, 0)
    a.label("loop")
    for u in range(U):
        load_x((u + 1) % 2)
        compute(u, u % 2)
        load_w(u)
    a.iadd(i, i, 1)
    a.setp_lt(p, i, iters - 1)
    a.brnz(p, "loop")
    for u in range(U):
        if u + 1 < U:
            load_x((u + 1) % 2)
        compute(u, u % 2)
    # out[tok, row] += acc: row rt 16 + g + 8 (e >> 1), token 8 tt + 2 c + (e & 1)
    a.ldc(t, "out")
    a.ldc(t2, "N")
    op = a.reg()
    for tt in range(NT):
        for e in range(4):
            a.imad(op, c, 2, 8 * tt + (e & 1))       # token
            a.imad(op, op, t2, g)                    # token * N + g
            a.imad(op, rt, 16, op)
            a.shl(op, op, 2)
            a.iadd(op, op, t)
            a.red_add(op, acc[tt] + e, imm=4 * 8 * (e >> 1))
    a.exit()
    return a.build()


def run(n, k, m, warps, split, U, compressed, seed=0, verbose=True):
    NT = (m + 7) // 8
    rng = np.random.default_rng(seed)
    wbits = to_bf16_bits(rng.standard_normal((n, k), dtype=np.float32) / np.sqrt(k))
    x = from_bf16_bits(to_bf16_bits(rng.standard_normal((8 * NT, k), dtype=np.float32)))
    x[m:] = 0
    sim = Sim(H200())
    if compressed:
        packed, b, esc = pack(wbits)
        wdat, d = packed, b & 7
    else:
        wdat, b, esc, d = frag_bf16(wbits), 0, [], 0
    pW, px, po = sim.mem.alloc(wdat.nbytes), sim.mem.alloc(x.size * 2), sim.mem.alloc(8 * NT * n * 4)
    sim.mem.write(pW, wdat)
    sim.mem.write(px, to_bf16_bits(x))
    sim.mem.write(po, np.zeros(8 * NT * n, np.float32))
    nrb = n // 16 // warps
    ktc = k // 64 // split
    prog = skinny_kernel(k, NT, warps, ktc, U, compressed, b & ~7)
    t0 = time.time()
    st = sim.launch(prog, nrb * split, warps * 32,
                    {"W": pW, "x": px, "out": po, "inv_nrb": (1 << 32) // nrb + 1, "neg_nrb": (1 << 32) - nrb, "N": n})
    wall = time.time() - t0
    out = sim.mem.read(po, np.float32, 8 * NT * n).reshape(8 * NT, n)[:m].astype(np.float64)
    out *= 2.0 ** (2 * d)
    W64 = from_bf16_bits(wbits).astype(np.float64)
    for r, kk, tb, db in esc:  # the escape correction, here on the host
        out[:, r] += x[:m, kk].astype(np.float64) * (from_bf16_bits(np.array([tb], np.uint16))[0]
                                                    - from_bf16_bits(np.array([db], np.uint16))[0] * 2.0 ** (2 * d))
    ref = x[:m].astype(np.float64) @ W64.T
    err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    if verbose:
        print(f"{'w12' if compressed else 'bf16'} n={n} k={k} m={m} warps={warps} split={split} U={U}: "
              f"{st.time_us:.2f} us, {st.hbm_bytes / 1e6:.1f} MB at {st.hbm_tbs:.2f} TB/s, issue "
              f"{100 * st.issue_util:.0f}%, err {err:.1e}, escapes {len(esc)} (wall {wall:.0f}s)")
    return st, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--split", type=int, default=4)
    ap.add_argument("--u", type=int, default=3)
    ap.add_argument("--only", choices=["bf16", "w12"])
    args = ap.parse_args()
    for comp in (False, True):
        if args.only and args.only != ("w12" if comp else "bf16"):
            continue
        st, _ = run(args.n, args.k, args.m, args.warps, args.split, args.u, comp)
        print("  " + st.report().replace("\n", "\n  "))


if __name__ == "__main__":
    main()
