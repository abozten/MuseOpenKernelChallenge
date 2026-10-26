"""W12 weights into wgmma: the decoded weights are wgmma's A operand (registers), x its B (shared).

python -m h200sim.w12wg [--n 2304] [--k 8192] [--m 128] [--wg 1] [--st 4] [--split 4] [--mcast 1] [--bf16]

The W12 decode (bf16c.py) leaves, in each lane, 4 groups of 4 registers in the mma.sync
A-fragment layout; group 2q + half holds the tile's elements (h, q, e) with e in 4 half .. +3.
wgmma with A in registers takes the same per-warp layout for its k16 step, so packing the
matrix with its columns permuted within each 64-k tile,

    W'[:, 32q + 8c + e] = W[:, 32q + 16 (e >> 2) + 8 ((e >> 1) & 1) + 2c + (e & 1)],

makes decode group s exactly the A fragment of k16 step s in natural k order, and x can sit in
shared memory in natural order, as TMA writes it (128 B swizzle), for B.

Kernel. A CTA computes 64 WG weight rows (warpgroup g: 4 warps, warp i row tile 4 g + i) for
all M tokens (wgmma N = M, padded to 8) over one k slice. The last warp produces: per stage of
one 64-k tile, a 2D TMA load of x's [M, 64] box (optionally multicast to a cluster of `mcast`
CTAs with the same k slice, which then each load 1 / mcast of the box) and one 1-D bulk copy of
the CTA's 4 WG weight tiles (k-major tiles: contiguous). A computing warp waits for the stage,
decodes its tile into one of two register sets, issues 4 wgmma (k16 steps) and commits; then
waits for the previous group, which frees the previous stage. Split K adds into an FP32 output
with red.add here (the timing of a real reduction is not the point of this model).
"""

import argparse
import time

import numpy as np

from . import H200, Sim, from_bf16_bits, to_bf16_bits
from .bf16c import LUT_AND_OR, pack
from .isa import Asm


def wg_perm() -> np.ndarray:
    """perm[k_old] = k_nat within a 64-k tile: stored position -> natural k (see the docstring)."""
    p = np.zeros(64, np.int64)
    for q in range(2):
        for c in range(4):
            for e in range(8):
                p[32 * q + 8 * c + e] = 32 * q + 16 * (e >> 2) + 8 * ((e >> 1) & 1) + 2 * c + (e & 1)
    assert sorted(p) == list(range(64))
    return p


def pack_wg(bits: np.ndarray):
    """pack() of the column-permuted matrix; escapes' k mapped back to natural k."""
    n, k = bits.shape
    perm = wg_perm()
    cols = (np.arange(k) // 64) * 64 + perm[np.arange(k) % 64]
    packed, b, esc = pack(bits[:, cols])
    esc = [(r, int(cols[kk]), tb, db) for r, kk, tb, db in esc]
    return packed, b, esc


def kernel(K: int, M: int, WG: int, ST: int, ktc: int, bp: int, tmap_x: int, mcast: int, NRT: int, bf16=False):
    """params: W (k-major tiles [KT][NRT][1536 B]), out (fp32 [M, N]), N.

    bf16: the same pipeline on plain bf16 weights (k-major tiles [KT][NRT][2048 B], 128 B
    pre-swizzled), A read by wgmma from shared memory: the uncompressed baseline."""
    TB = 2048 if bf16 else 1536                 # bytes per 16 x 64 weight tile
    NP = (M + 7) // 8 * 8                       # wgmma N
    XB = NP * 128                               # x stage: NP rows x 128 B (64 bf16), swizzled
    WB = 4 * WG * TB
    SB = (XB + WB + 1023) // 1024 * 1024        # stages 1024 B aligned (128 B swizzle atoms)
    MB = ST * SB                                # full barriers, then empty barriers
    PROD = 4 * WG                               # producer warp index
    a = Asm("w12_wgmma")
    wid, lane, cta, t, t2, p, q = (a.reg() for _ in range(7))
    l0 = a.preg()                               # lane 0
    rb, s_, it, st, par, sa, mb, mbe, col, row0, zero, wsrc = (a.reg() for _ in range(12))
    g, c, wi, rt = (a.reg() for _ in range(4))

    def vec(n):
        a.nxt = (a.nxt + 3) // 4 * 4
        return a.reg(n)
    Wr = vec(12)
    A = [[vec(4) for _ in range(4)] for _ in range(2)]   # two register sets of 4 k16 groups
    acc = vec(NP // 2)
    m87, bb, hA, rot, hB = (a.reg() for _ in range(5))

    a.s2r(wid, "warpid")
    a.s2r(lane, "laneid")
    a.s2r(cta, "ctaid")
    # cta = s * nrb + rb (consecutive rb share a k slice: a multicast cluster)
    a.ldc(s_, "inv_nrb")
    a.imul_hi(s_, cta, s_)
    a.ldc(t, "neg_nrb")
    a.imad(rb, s_, t, cta)
    a.mov(zero, 0)
    a.setp_eq(l0, lane, 0)

    a.setp_eq(p, wid, PROD)
    a.brnz(p, "producer")

    # ---- computing warps ----
    a.shr(g, lane, 2)
    a.and_(c, lane, 3)
    a.mov(m87, 0x87878787)
    a.mov(bb, bp * 0x01010101)
    for r in range(NP // 2):
        a.mov(acc + r, 0)
    a.bar()                                     # barriers initialised by the producer
    a.mov(it, 0)
    a.label("loop")
    a.and_(st, it, ST - 1)
    a.shr(par, it, ST.bit_length() - 1)
    a.and_(par, par, 1)
    a.imad(sa, st, SB, 0)
    a.imad(mb, st, 8, MB)
    a.mbar_wait(mb, par)
    if bf16:
        a.shr(t2, wid, 2)
        a.imad(t2, t2, 8192, XB)
        a.iadd(t2, t2, sa)                      # this warpgroup's 64 x 64 A tile
        a.wgmma_fence()
        for s in range(4):
            a.iadd(t, sa, 32 * s)
            a.iadd(q, t2, 32 * s)
            a.wgmma(acc, q, t, NP, 128, 128, accumulate=True, swizzle=128)
        a.wgmma_commit()
        a.bra("issued")
    # this warp's weight tile: 3 x 16 B per lane at XB + wid * 1536
    a.imad(t, wid, 1536, XB)
    a.iadd(t, t, sa)
    a.shl(t2, lane, 4)
    a.iadd(t, t, t2)
    for l in range(3):
        a.lds(Wr + 4 * l, t, 512 * l, 16)
    a.and_(q, it, 1)                            # register set it % 2
    a.brnz(q, "odd")
    for half_set, label in ((0, None), (1, "odd")):
        if label:
            a.label(label)
        Aset = A[half_set]
        for qq in range(2):
            for half in range(2):
                for h in range(2):
                    nw = Wr + 8 + 2 * h + qq
                    lw = Wr + 4 * h + 2 * qq + half
                    if half == 0:
                        a.lop3(hA, nw, m87, bb, LUT_AND_OR)
                        src = hA
                    else:
                        a.shf(rot, nw, nw, 4)
                        a.lop3(hB, rot, m87, bb, LUT_AND_OR)
                        src = hB
                    a.prmt(Aset[2 * qq + half] + h, lw, src, 0x5140)
                    a.prmt(Aset[2 * qq + half] + 2 + h, lw, src, 0x7362)
        a.wgmma_fence()
        for s in range(4):
            a.iadd(t, sa, 32 * s)
            a.wgmma(acc, Aset[s], t, NP, 0, 128, accumulate=True, swizzle=128, a_in_regs=True)
        a.wgmma_commit()
        a.bra("issued")
    a.label("issued")
    # the previous group is done: its stage (it - 1) is free
    a.wgmma_wait(1)
    a.setp_eq(p, it, 0)
    a.brnz(p, "noprev")
    a.iadd(t, it, ST - 1)
    a.and_(t, t, ST - 1)
    a.imad(mbe, t, 8, MB + 8 * ST)
    with a.guard(l0):
        a.mbar_arrive(mbe)
    a.label("noprev")
    a.iadd(it, it, 1)
    a.setp_lt(p, it, ktc)
    a.brnz(p, "loop")
    a.wgmma_wait(0)
    a.iadd(t, it, ST - 1)                       # the last stage
    a.and_(t, t, ST - 1)
    a.imad(mbe, t, 8, MB + 8 * ST)
    with a.guard(l0):
        a.mbar_arrive(mbe)
    # out[tok, row] += acc: warp wi of its warpgroup, row 16 wi + g (+8), token 8 j + 2 c (+1)
    a.and_(wi, wid, 3)
    a.shr(t, wid, 2)
    a.imad(rt, rb, 4 * WG, 0)
    a.imad(rt, t, 4, rt)
    a.iadd(rt, rt, wi)                          # this warp's row tile
    a.ldc(t, "out")
    a.ldc(t2, "N")
    op = a.reg()
    for j in range(NP // 8):
        for r in range(4):
            a.imad(op, c, 2, 8 * j + (r & 1))       # token
            a.imad(op, op, t2, g)
            a.imad(op, rt, 16, op)
            a.shl(op, op, 2)
            a.iadd(op, op, t)
            a.red_add(op, acc + 4 * j + r, imm=4 * 8 * (r >> 1))
    a.exit()

    # ---- producer ----
    a.label("producer")
    for i in range(ST):
        a.mov(mb, MB + 8 * i)
        with a.guard(l0):
            a.mbar_init(mb, 1)
        a.mov(mb, MB + 8 * (ST + i))
        with a.guard(l0):
            a.mbar_init(mb, 4 * WG)
    a.fence()
    a.bar()
    # first k tile of the slice; weight source W + ((kt0 + it) NRT + rb 4 WG) 1536
    a.ldc(wsrc, "W")
    a.imad(t, s_, ktc * NRT, 0)
    a.imad(t, rb, 4 * WG, t)
    a.imad(t, t, TB, 0)
    a.iadd(wsrc, wsrc, t)
    a.imad(col, s_, ktc * 64, 0)
    a.mov(it, 0)
    a.label("ploop")
    a.and_(st, it, ST - 1)
    a.shr(par, it, ST.bit_length() - 1)
    a.and_(par, par, 1)
    a.setp_lt(p, it, ST)
    a.brnz(p, "pfree")
    a.imad(mbe, st, 8, MB + 8 * ST)
    a.xor(t, par, 1)
    a.mbar_wait(mbe, t)
    a.label("pfree")
    a.imad(sa, st, SB, 0)
    a.imad(mb, st, 8, MB)
    if mcast > 1:
        # this CTA loads rows [r0, r0 + NP / mcast) of the box into every CTA of the cluster
        rank = a.reg()
        a.s2r(rank, "clusterrank")
        a.imad(row0, rank, NP // mcast, 0)
        a.imad(t, row0, 128, sa)
    with a.guard(l0):
        a.mbar_arrive_tx(mb, XB + WB)
        if mcast > 1:
            a.tma_load_2d(t, tmap_x, col, row0, mb, mcast=(1 << mcast) - 1)
        else:
            a.tma_load_2d(sa, tmap_x, col, zero, mb)
    a.iadd(t, sa, XB)
    with a.guard(l0):
        a.tma_load_1d(t, wsrc, WB, mb)
    a.iadd(wsrc, wsrc, NRT * TB)
    a.iadd(col, col, 64)
    a.iadd(it, it, 1)
    a.setp_lt(p, it, ktc)
    a.brnz(p, "ploop")
    a.exit()
    return a.build(), MB + 16 * ST


def run(n, k, m, WG, ST, split, mcast, seed=0, verbose=True, bf16=False):
    rng = np.random.default_rng(seed)
    wbits = to_bf16_bits(rng.standard_normal((n, k), dtype=np.float32) / np.sqrt(k))
    NP = (m + 7) // 8 * 8
    x = from_bf16_bits(to_bf16_bits(rng.standard_normal((NP, k), dtype=np.float32)))
    x[m:] = 0
    NRT, KT = n // 16, k // 64
    if bf16:
        d, bp, esc = 0, 0, []
        t = wbits.reshape(n, KT, 8, 8).transpose(1, 0, 2, 3)  # [KT][row][chunk][8]
        wdat = np.empty_like(t)
        for r in range(8):                                    # 128 B swizzle: chunk ^ row % 8
            wdat[:, r::8, np.arange(8) ^ r] = t[:, r::8]
    else:
        packed, b, esc = pack_wg(wbits)
        d, bp = b & 7, b & ~7
        wdat = packed.transpose(1, 0, 2, 3, 4).copy()        # k-major tiles [KT][NRT]
    sim = Sim(H200())
    pW, px, po = sim.mem.alloc(wdat.nbytes), sim.mem.alloc(x.size * 2), sim.mem.alloc(NP * n * 4)
    sim.mem.write(pW, wdat)
    sim.mem.write(px, to_bf16_bits(x))
    sim.mem.write(po, np.zeros(NP * n, np.float32))
    tm = sim.tensor_map(px, NP, k, 2, NP // mcast, 64, swizzle=128)
    nrb = NRT // (4 * WG)
    ktc = KT // split
    prog, smem = kernel(k, m, WG, ST, ktc, bp, tm, mcast, NRT, bf16)
    t0 = time.time()
    st = sim.launch(prog, nrb * split, (4 * WG + 1) * 32,
                    {"W": pW, "out": po, "N": n, "inv_nrb": (1 << 32) // nrb + 1, "neg_nrb": (1 << 32) - nrb},
                    smem_bytes=smem, cluster=mcast)
    wall = time.time() - t0
    out = sim.mem.read(po, np.float32, NP * n).reshape(NP, n)[:m].astype(np.float64) * 2.0 ** (2 * d)
    for r, kk, tb, db in esc:
        out[:, r] += x[:m, kk].astype(np.float64) * (from_bf16_bits(np.array([tb], np.uint16))[0]
                                                    - from_bf16_bits(np.array([db], np.uint16))[0] * 2.0 ** (2 * d))
    ref = x[:m].astype(np.float64) @ from_bf16_bits(wbits).astype(np.float64).T
    err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    flops = 2.0 * m * n * k
    if verbose:
        print(f"{'bf16' if bf16 else 'w12'} wgmma n={n} k={k} m={m} WG={WG} ST={ST} split={split} mcast={mcast}: {st.time_us:.2f} us, "
              f"{flops / st.time_us / 1e6:.0f} TFLOPS, {st.hbm_bytes / 1e6:.1f} MB HBM at {st.hbm_tbs:.2f} TB/s, "
              f"issue {100 * st.issue_util:.0f}%, err {err:.1e} (wall {wall:.0f}s)", flush=True)
    return st, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2304)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--wg", type=int, default=1)
    ap.add_argument("--st", type=int, default=4)
    ap.add_argument("--split", type=int, default=4)
    ap.add_argument("--mcast", type=int, default=1)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    st, _ = run(args.n, args.k, args.m, args.wg, args.st, args.split, args.mcast, bf16=args.bf16)
    print("  " + st.report().replace("\n", "\n  "))


if __name__ == "__main__":
    main()
