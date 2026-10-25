"""Tensor-core data types, sparsity and timing. python3 -m h200sim.test_tensor"""

import numpy as np

from . import Asm, H200, Sim, to_bf16_bits
from . import tensor as T


def rand_raw(dt, shape, rng):
    if dt == "f16":
        return rng.standard_normal(shape).astype(np.float16).view(np.uint16)
    if dt == "bf16":
        return to_bf16_bits(rng.standard_normal(shape).astype(np.float32))
    if dt == "tf32":
        return rng.integers(0, 2**32, shape, dtype=np.uint32) & np.uint32(0x3FFFFFFF) | np.uint32(0x20000000)
    if dt in ("e4m3", "e5m2"):
        lut = T.E4M3 if dt == "e4m3" else T.E5M2
        good = np.nonzero(np.isfinite(lut))[0]
        return rng.choice(good, shape).astype(np.uint8)
    if dt == "s8":
        return rng.integers(-100, 100, shape).astype(np.int8).view(np.uint8)
    return rng.integers(0, 200, shape, dtype=np.uint8)


def _expected(dt, acc, A, B, C):
    if dt in ("e4m3", "e5m2"):
        return T.fp8_accumulate(A, B.T, C)
    if acc == "s32":
        return A.astype(np.int64) @ B.astype(np.int64) + C
    return A.astype(np.float32) @ B.astype(np.float32) + C


def run_mma(dt, k, acc, c_fill=None, seed=0, A=None, B=None):
    sim = Sim(H200())
    rng = np.random.default_rng(seed)
    spec = T.DTYPES[dt]
    epr = 32 // spec.bits
    na, nb, nc = k // (2 * epr), k // (4 * epr), 2 if acc == "f16" else 4
    if A is None:
        A, B = rand_raw(dt, (16, k), rng), rand_raw(dt, (k, 8), rng)
    fa = T.pack_a_fragment(dt, A, k).T.copy()
    fb = T.pack_b_fragment(dt, B, k).T.copy()
    fc = np.zeros((32, nc), np.uint32)
    if c_fill is not None:
        fc[:] = c_fill
    pa, pb, pc = (sim.mem.alloc(x.nbytes) for x in (fa, fb, fc))
    for p, x in ((pa, fa), (pb, fb), (pc, fc)):
        sim.mem.write(p, x)
    po = sim.mem.alloc(32 * nc * 4)
    a = Asm("mma")
    lane, addr, t = a.reg(), a.reg(), a.reg()
    ra, rb, rc, rd = a.reg(na), a.reg(nb), a.reg(nc), a.reg(nc)
    a.s2r(lane, "laneid")
    for reg, n, name in ((ra, na, "a"), (rb, nb, "b"), (rc, nc, "c")):
        a.ldc(t, name)
        a.imad(addr, lane, 4 * n, t)
        a.ldg(reg, addr, 0, 4 * n)
    a.hmma(rd, ra, rb, rc, dtype=dt, acc=acc, k=k)
    a.ldc(t, "out")
    a.imad(addr, lane, 4 * nc, t)
    a.stg(addr, rd, 0, 4 * nc)
    sim.launch(a.build(), 1, 32, {"a": pa, "b": pb, "c": pc, "out": po})
    regs = sim.mem.read(po, np.uint32, 32 * nc).reshape(32, nc).T.copy()
    got = T.c_fragment(regs, acc)
    Cm = T.c_fragment(fc.T.copy(), acc)
    want = _expected(dt, acc, T.decode(dt, A), T.decode(dt, B), Cm)
    return got, want


def test_mma_all_types():
    small = lambda shape, r: (r.integers(0x20, 0x40, shape) | (r.integers(0, 2, shape) << 7)).astype(np.uint8)
    rng = np.random.default_rng(5)
    got, want = run_mma("e4m3", 16, "f16", A=small((16, 16), rng), B=small((16, 8), rng))   # |values| <= 2
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-2, err_msg="e4m3 k16 acc f16")
    cases = [("f16", 16, "f32"), ("f16", 8, "f32"), ("f16", 16, "f16"), ("bf16", 16, "f32"), ("bf16", 8, "f32"),
             ("tf32", 8, "f32"), ("tf32", 4, "f32"), ("e4m3", 32, "f32"), ("e5m2", 32, "f32"),
             ("s8", 32, "s32"), ("u8", 32, "s32"), ("s8", 16, "s32")]
    for dt, k, acc in cases:
        got, want = run_mma(dt, k, acc)
        tol = dict(rtol=2e-3, atol=2e-2) if acc == "f16" else dict(rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(got, want, err_msg=f"{dt} k{k} acc {acc}", **tol)


def test_f16_accumulator_rounds_to_half():
    # C = 2048 (f16), A.B = 1: f32 accumulate gives 2049; the f16 accumulator can only hold 2048
    one = np.float16(1.0).view(np.uint16)
    A = np.zeros((16, 16), np.uint16); B = np.zeros((16, 8), np.uint16)
    A[:, 0], B[0, :] = one, one
    got32, _ = run_mma("f16", 16, "f32", c_fill=np.float32(2048).view(np.uint32), A=A, B=B)
    got16, _ = run_mma("f16", 16, "f16", c_fill=np.uint32(0x68006800), A=A, B=B)
    assert (got32 == 2049).all() and (got16 == 2048).all()


def test_fp8_accumulation_truncates_small_products():
    """Products far below the largest one's exponent vanish (DeepSeek's FP8 accumulation finding)."""
    enc = lambda x: T.encode_fp8(np.array([x], np.float32), T.E4M3)[0]
    A = np.zeros((16, 32), np.uint8); B = np.zeros((32, 8), np.uint8)
    A[:, 0], B[0, :] = enc(256.0), enc(1.0)
    A[:, 1:], B[1:, :] = enc(2.0 ** -6), enc(1.0)         # 31 products of 2^-6: exact sum 0.484
    got, want = run_mma("e4m3", 32, "f32", A=A, B=B)
    exact = 256.0 + 31 * 2.0 ** -6
    assert (got == 256.0).all() and np.allclose(want, got) and exact != 256.0


def run_wgmma(dt, acc, n, rs=False, sparse=False, swizzle=0, seed=1, mn=False, sel=0):
    sim = Sim(H200())
    rng = np.random.default_rng(seed)
    spec = T.DTYPES[dt]
    eb = spec.bits // 8
    kc = spec.k_wg
    ke = 2 * kc if sparse else kc
    Ar, Br = rand_raw(dt, (64, kc), rng), rand_raw(dt, (n, ke), rng)
    metaw = None
    if sparse:
        gw, kept, chunks = T._sparse_geometry(dt, ke)
        if dt == "tf32":
            idx = rng.integers(0, 2, (64, chunks, 1))
        else:
            pairs = np.array(sorted(T._VALID_24))
            idx = pairs[rng.integers(0, len(pairs), (64, chunks))]
        cd = T.decode(dt, Ar)
        Ad = np.zeros((64, ke), cd.dtype)
        for r in range(64):
            for c in range(chunks):
                if dt == "tf32":
                    Ad[r, 2 * c + idx[r, c, 0]] = cd[r, c]
                else:
                    Ad[r, 4 * c + idx[r, c, 0]] = cd[r, 2 * c]
                    Ad[r, 4 * c + idx[r, c, 1]] = cd[r, 2 * c + 1]
        # each warp of the warpgroup supplies the metadata of its 16 rows, in the mma.sp register layout
        metaw = np.concatenate([T.encode_sparse_meta(dt, idx[16 * wi:16 * wi + 16], ke, sel) for wi in range(4)])
    else:
        Ad = T.decode(dt, Ar)
    Bd = T.decode(dt, Br)
    pA, pB = sim.mem.alloc(Ar.nbytes), sim.mem.alloc(Br.nbytes)
    sim.mem.write(pA, Ar.T.copy() if mn else Ar); sim.mem.write(pB, Br.T.copy() if mn else Br)
    nacc = n // 4 if acc == "f16" else n // 2
    po = sim.mem.alloc(128 * nacc * 4)
    a_bytes, b_bytes = 64 * kc * eb, n * ke * eb
    a_off, b_off = 0, (a_bytes + 127) // 128 * 128
    mb_off = b_off + b_bytes + 16
    smem = mb_off + 16
    if mn:      # stored K x M and K x N: the tile is "transposed" in shared memory
        tA = sim.tensor_map(pA, kc, 64, eb, kc, 64)
        tB = sim.tensor_map(pB, ke, n, eb, ke, n)
    else:
        tA = sim.tensor_map(pA, 64, kc, eb, 64, kc)
        tB = sim.tensor_map(pB, n, ke, eb, n, ke)
    if sparse:
        pM = sim.mem.alloc(metaw.nbytes); sim.mem.write(pM, metaw)
    if rs:
        fr = np.zeros((128, 4), np.uint32)
        for wi in range(4):
            fr[wi * 32:(wi + 1) * 32] = T.pack_a_fragment(dt, Ar[16 * wi:16 * wi + 16], kc).T
        pF = sim.mem.alloc(fr.nbytes); sim.mem.write(pF, fr)

    a = Asm("wgmma_t")
    wid, tid, p, t, addr, zero, sa, sb, mb, rmeta = (a.reg() for _ in range(10))
    ra, acc_r = a.reg(4), a.reg(nacc)
    a.s2r(wid, "warpid"); a.s2r(tid, "tid"); a.mov(zero, 0)
    a.mov(mb, mb_off); a.mov(sa, a_off); a.mov(sb, b_off)
    a.setp_lt(p, 0, wid)
    a.brnz(p, "go")
    a.mbar_init(mb, 1)
    a.fence()
    a.mbar_arrive_tx(mb, (0 if rs else a_bytes) + b_bytes)
    if not rs:
        a.tma_load_2d(sa, tA, zero, zero, mb)
    a.tma_load_2d(sb, tB, zero, zero, mb)
    a.label("go")
    a.bar()
    if rs:
        a.ldc(t, "frag"); a.imad(addr, tid, 16, t); a.ldg(ra, addr, 0, 16)
    if sparse:
        a.ldc(t, "meta"); a.imad(addr, tid, 4, t); a.ldg(rmeta, addr, 0, 4)     # this lane's metadata word
    a.mov(zero, 0)
    a.mbar_wait(mb, zero)
    a.wgmma_fence()
    lda, ldb = (64 * eb, n * eb) if mn else (kc * eb, ke * eb)
    a.wgmma(acc_r, ra if rs else sa, sb, n, lda, ldb, accumulate=False, swizzle=swizzle,
            a_in_regs=rs, dtype=dt, acc=acc, sparse=sparse, meta=rmeta if sparse else None, sel=sel,
            a_mn=mn and not rs, b_mn=mn)
    a.wgmma_commit()
    a.wgmma_wait(0)
    a.ldc(t, "out"); a.imad(addr, tid, nacc * 4, t)
    for r in range(nacc):
        a.stg(addr, acc_r + r, r * 4)
    params = {"out": po, "frag": pF if rs else 0, "meta": pM if sparse else 0}
    st = sim.launch(a.build(), 1, 128, params, smem_bytes=smem)
    regs = sim.mem.read(po, np.uint32, 128 * nacc).reshape(128, nacc)
    nreg = 2 if acc == "f16" else 4
    got = np.zeros((64, n), np.float32 if acc != "s32" else np.int32)
    for wi in range(4):
        for j in range(n // 8):
            blk = regs[wi * 32:(wi + 1) * 32, nreg * j:nreg * j + nreg].T.copy()
            got[16 * wi:16 * wi + 16, 8 * j:8 * j + 8] = T.c_fragment(blk, acc)
    want = np.zeros((64, n), got.dtype)
    for wi in range(4):
        sl = slice(16 * wi, 16 * wi + 16)
        want[sl] = _expected(dt, acc, Ad[sl], Bd.T, np.zeros((16, n), got.dtype))
    return got, want, st


def test_wgmma_types_ss_rs_and_sparse():
    for dt, acc in (("bf16", "f32"), ("f16", "f32"), ("f16", "f16"), ("tf32", "f32"), ("e4m3", "f32"),
                    ("e5m2", "f32"), ("s8", "s32"), ("u8", "s32")):
        for rs in (False, True):
            for sparse in (False, True):
                sel = 1 if sparse and dt in ("bf16", "f16", "tf32") else 0       # thread pair T2,T3 where allowed
                got, want, _ = run_wgmma(dt, acc, 32, rs=rs, sparse=sparse, sel=sel)
                tol = dict(rtol=3e-3, atol=3e-2) if acc == "f16" else dict(rtol=1e-5, atol=1e-4)
                np.testing.assert_allclose(got, want, err_msg=f"{dt}/{acc} rs={rs} sparse={sparse}", **tol)


def test_wgmma_transposed_mn_major_operands():
    for dt in ("bf16", "f16"):
        for rs in (False, True):
            got, want, _ = run_wgmma(dt, "f32", 32, rs=rs, mn=True)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4, err_msg=f"{dt} rs={rs}")


def test_tma_3d_box_load_store_with_clipping():
    sim = Sim(H200())
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, 6, 32)).astype(np.float32)           # [d2][d1][d0], d0 fastest
    pX = sim.mem.alloc(X.nbytes); sim.mem.write(pX, X)
    pO = sim.mem.alloc(X.nbytes); sim.mem.write(pO, np.zeros_like(X))
    tin = sim.tensor_map_nd(pX, (32, 6, 4), (32 * 4, 6 * 32 * 4), (32, 4, 2), 4)
    tout = sim.tensor_map_nd(pO, (32, 6, 4), (32 * 4, 6 * 32 * 4), (32, 4, 2), 4)
    box_bytes = 32 * 4 * 2 * 4
    a = Asm("tma3d")
    sa, mb, c0, c1, c2 = (a.reg() for _ in range(5))
    a.mov(sa, 0); a.mov(mb, box_bytes); a.mov(c0, 0); a.mov(c1, 4); a.mov(c2, 3)
    a.mbar_init(mb, 1)
    a.mbar_arrive_tx(mb, box_bytes)                        # the whole box counts, even the clipped part
    a.tma_load_nd(sa, tin, (c0, c1, c2), mb)
    a.mbar_wait(mb, 0)
    a.tma_store_nd(sa, tout, (c0, c1, c2))
    a.cp_commit(); a.cp_wait(0)
    sim.launch(a.build(), 1, 32, {}, smem_bytes=box_bytes + 16)
    got = sim.mem.read(pO, np.float32, X.size).reshape(X.shape)
    want = np.zeros_like(X)
    want[3:4, 4:6, :] = X[3:4, 4:6, :]                     # d1 rows 6,7 and d2 slab 4 are outside the tensor
    np.testing.assert_array_equal(got, want)


def test_wgmma_timing_matches_measured_table():
    """H800 Table IX/X, m64nNk16 FP16: latency (= cycles per instruction in a chain)."""
    want = {  # n: (dense SS, dense RS, sparse SS, sparse RS)
        256: (128, 128, 144, 128), 128: (64, 64, 80, 64), 64: (32, 32, 48, 32),
        32: (24, 16, 32, 18), 16: (20, 13, 24, 18), 8: (18, 13, 20, 16)}
    for n, row in want.items():
        for (rs, sparse), target in zip(((False, False), (True, False), (False, True), (True, True)), row):
            assert abs(T.wgmma_cycles(n, rs, sparse) - target) < 0.5, (n, rs, sparse)


def test_wgmma_chain_runs_at_modeled_rate():
    """16 back-to-back m64n256 wgmmas on one sub-core take ~16 x 128 cycles; random data derates them."""
    for random_data, ratio in ((False, 1.0), (True, 0.913)):
        sim = Sim(H200(tensor_data_random=random_data, model_icache=False, model_ccache=False))
        sim.log_issue = True
        sim.mem.alloc(16)
        a = Asm("chain")
        sa, sb, t = a.reg(), a.reg(), a.reg()
        acc = a.reg(128)
        a.mov(sa, 0); a.mov(sb, 0)
        for _ in range(16):
            a.wgmma(acc, sa, sb, 256, 32, 32, dtype="bf16", acc="f32")
        a.wgmma_commit()
        a.wgmma_wait(0)
        a.s2r(t, "clock")
        sim.launch(a.build(), 1, 32, {}, smem_bytes=256 * 32 + 64 * 32 + 64)
        first = min(e[0] for e in sim.issue_log if e[6] == "WGMMA")
        end = [e[0] for e in sim.issue_log if e[6] == "S2R"][-1]
        per = (end - first) / 16
        assert abs(per - 128 / ratio) < 0.06 * 128 / ratio, (random_data, per)


def test_mma_timing_fraction_of_peak():
    lat, occ = T.mma_timing("bf16", 16, "f32")
    assert lat == 24.1 and abs(occ - 4096 / (1024 * 490.7 / 756.5)) < 1e-6      # 65 % of the dense peak
    assert abs(T.mma_timing("s8", 32, "s32")[1] - 8192 / (2048 * 977.9 / 1513.0)) < 1e-6


def _sparse_case(dt, k, acc, sel, seed=0):
    """Random structured-sparse A (as indices), compress it, run mma.sp, compare with the dense product."""
    rng = np.random.default_rng(seed)
    gw, kept, chunks = T._sparse_geometry(dt, k)
    if dt == "tf32":                                     # 1:2: one kept element in each pair
        idx = rng.integers(0, 2, (16, chunks, 1))
    else:                                                # 2:4: two kept elements, idx0 < idx1
        pairs = sorted(T._VALID_24)
        idx = np.array(pairs)[rng.integers(0, len(pairs), (16, chunks))]
    comp = rand_raw(dt, (16, k // 2), rng)
    B = rand_raw(dt, (k, 8), rng)
    meta = T.encode_sparse_meta(dt, idx, k, sel)
    A_dense = np.zeros((16, k), np.float32 if dt != "s8" else np.int32)
    cd = T.decode(dt, comp)
    for r in range(16):
        for c in range(chunks):
            if dt == "tf32":
                A_dense[r, 2 * c + idx[r, c, 0]] = cd[r, c]
            else:
                A_dense[r, 4 * c + idx[r, c, 0]] = cd[r, 2 * c]
                A_dense[r, 4 * c + idx[r, c, 1]] = cd[r, 2 * c + 1]
    sim = Sim(H200())
    bits = T.DTYPES[dt].bits; epr = 32 // bits
    na, nb, nc = (k // 2) // (2 * epr), k // (4 * epr), 4
    fa = T.pack_a_fragment(dt, comp, k // 2).T.copy(); fb = T.pack_b_fragment(dt, B, k).T.copy()
    fc = np.zeros((32, nc), np.uint32)
    ptrs = [sim.mem.alloc(x.nbytes) for x in (fa, fb, fc, meta)]
    for p, x in zip(ptrs, (fa, fb, fc, meta)):
        sim.mem.write(p, x)
    po = sim.mem.alloc(32 * nc * 4)
    a = Asm("mmasp")
    lane, addr, t, rm = a.reg(), a.reg(), a.reg(), a.reg()
    ra, rb, rc, rd = a.reg(na), a.reg(nb), a.reg(nc), a.reg(nc)
    a.s2r(lane, "laneid")
    for reg, n, name in ((ra, na, "a"), (rb, nb, "b"), (rc, nc, "c")):
        a.ldc(t, name); a.imad(addr, lane, 4 * n, t); a.ldg(reg, addr, 0, 4 * n)
    a.ldc(t, "m"); a.imad(addr, lane, 4, t); a.ldg(rm, addr, 0, 4)
    a.hmma_sp(rd, ra, rb, rc, rm, sel=sel, dtype=dt, acc=acc, k=k)
    a.ldc(t, "out"); a.imad(addr, lane, 4 * nc, t); a.stg(addr, rd, 0, 4 * nc)
    sim.launch(a.build(), 1, 32, {"a": ptrs[0], "b": ptrs[1], "c": ptrs[2], "m": ptrs[3], "out": po})
    got = T.c_fragment(sim.mem.read(po, np.uint32, 32 * nc).reshape(32, nc).T.copy(), acc)
    want = _expected(dt, acc, A_dense, T.decode(dt, B), np.zeros((16, 8), got.dtype))
    return got, want


def test_mma_sp_sparse_mma_sync():
    for dt, k, acc, sels in (("f16", 32, "f32", (0, 1)), ("bf16", 32, "f32", (0, 1)), ("f16", 16, "f32", (0, 1, 2, 3)),
                             ("tf32", 16, "f32", (0, 1)), ("tf32", 8, "f32", (0, 1, 2, 3)), ("s8", 64, "s32", (0,)),
                             ("s8", 32, "s32", (0, 1)),
                             ("e4m3", 64, "f32", (0,))):
        for sel in sels:
            got, want = _sparse_case(dt, k, acc, sel)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3, err_msg=f"{dt} k{k} sel{sel}")


def test_mma_sp_is_faster_than_dense_per_flop():
    lat, occ = T.mma_sp_timing("bf16", 32, "f32")
    dlat, docc = T.mma_timing("bf16", 16, "f32")
    assert lat == 24.0 and 1.2 < (occ / docc) < 1.6          # twice the effective FLOPs in ~1.4x the cycles


def test_tma_im2col_convolution_taps_padding_and_wrap():
    sim = Sim(H200())
    rng = np.random.default_rng(9)
    N, H, W, C = 2, 5, 6, 8
    X = rng.standard_normal((N, H, W, C)).astype(np.float32)
    pX = sim.mem.alloc(X.nbytes); sim.mem.write(pX, X)
    ppc = 9                                                # crosses a row boundary (W = 6)
    tm = sim.tensor_map_im2col(pX, (C, W, H, N), (C * 4, W * C * 4, H * W * C * 4), lower=(-1, -1), upper=(-1, -1),
                               channels_per_pixel=C, pixels_per_column=ppc, elem_bytes=4)
    pO = sim.mem.alloc(ppc * C * 4)
    nbytes = ppc * C * 4

    def run_tap(c0, w0, h0, n0, ow, oh):
        a = Asm("im2col")
        sa, mb, cs = a.reg(), a.reg(), [a.reg() for _ in range(4)]
        os_ = [a.reg(), a.reg()]
        a.mov(sa, 0); a.mov(mb, nbytes)
        for r, v in zip(cs, (c0, w0, h0, n0)):
            a.mov(r, v)
        for r, v in zip(os_, (ow, oh)):
            a.mov(r, v)
        a.mbar_init(mb, 1); a.mbar_arrive_tx(mb, nbytes)
        a.tma_load_im2col(sa, tm, cs, os_, mb)
        a.mbar_wait(mb, 0)
        g = a.reg(); a.ldc(g, "o")
        a.tma_store_1d(g, sa, nbytes)
        a.cp_commit(); a.cp_wait(0)
        sim.launch(a.build(), 1, 32, {"o": pO}, smem_bytes=nbytes + 16)
        return sim.mem.read(pO, np.float32, ppc * C).reshape(ppc, C)

    def reference(w0, h0, n0, ow, oh):
        out = np.zeros((ppc, C), np.float32)
        a_w, a_h, n = w0, h0, n0
        for p in range(ppc):
            w, h = a_w + ow, a_h + oh
            if 0 <= w < W and 0 <= h < H and 0 <= n < N:
                out[p] = X[n, h, w]
            a_w += 1
            if a_w > W - 1 - 1:                              # dim + upper - 1 = W - 2 for pad 1, 3x3
                a_w = -1; a_h += 1
                if a_h > H - 1 - 1:
                    a_h = -1; n += 1
        return out

    for (w0, h0, n0, ow, oh) in ((-1, -1, 0, 1, 1), (-1, -1, 0, 0, 0), (-1, -1, 0, 2, 2), (3, 0, 0, 1, 2), (2, 2, 1, 0, 1)):
        np.testing.assert_array_equal(run_tap(0, w0, h0, n0, ow, oh), reference(w0, h0, n0, ow, oh),
                                      err_msg=str((w0, h0, n0, ow, oh)))
    # centre tap of an unpadded-looking window is just the image: the first row of pixels equals X[0, 0, :, :]
    np.testing.assert_array_equal(run_tap(0, -1, -1, 0, 1, 1)[:W], X[0, 0])


def test_wgmma_matches_the_nsight_compute_data_in_a_chinese_analysis():
    """Zhihu, 'NVIDIA Hopper架构TensorCore分析(4)' (H100 PCIe, Nsight Compute, zero data): shared-memory wavefronts
    per wgmma and Tensor Core utilisation for m64nNk16 in SS and RS mode."""
    wavefronts = {"SS": {256: 80, 128: 48, 64: 32, 32: 24, 16: 20, 8: 18},
                  "RS": {256: 64, 128: 32, 64: 16, 32: 8, 16: 4, 8: 2}}
    util = {"SS": {256: 97.3, 128: 97.1, 64: 96.7, 32: 64.6, 16: 38.8, 8: 21.2},
            "RS": {256: 97.3, 128: 97.1, 64: 96.9, 32: 96.0, 16: 59.0, 8: 29.5}}
    for mode, rs in (("SS", False), ("RS", True)):
        for n in (256, 128, 64, 32, 16, 8):
            assert T.wgmma_smem_wavefronts(n, rs) == wavefronts[mode][n], (mode, n)      # exact
            model = 100.0 * (n / 2) / T.wgmma_cycles(n, rs, False)
            assert abs(model - util[mode][n]) < 4.5, (mode, n, model, util[mode][n])     # within 4.5 points (Nsight includes ~3 % overhead)


def test_tma_im2col_w_modes_halo_and_offset():
    """PTX ISA 5.5.5: im2col::w loads Pixels-per-Column pixels along W plus wHalo halo pixels at the end;
    im2col::w::128 loads 128 pixels with wHalo more after every 32; wOffset shifts the first pixel; positions
    outside the tensor read as zero; H and D stay fixed; im2col_no_offs::w rejects negative coordinates."""
    rng = np.random.default_rng(13)
    N, H, W, C = 2, 3, 150, 8
    X = rng.standard_normal((N, H, W, C)).astype(np.float32)
    sim = Sim(H200())
    pX = sim.mem.alloc(X.nbytes); sim.mem.write(pX, X)
    ppc = 40
    tm = sim.tensor_map_im2col(pX, (C, W, H, N), (C * 4, W * C * 4, H * W * C * 4), lower=(0, 0), upper=(0, 0),
                               channels_per_pixel=C, pixels_per_column=ppc, elem_bytes=4, traversal=(1, 1))
    rowb = C * 4

    def run(mode, coords, whalo=0, woffset=0):
        rows = tm_rows(mode, whalo)
        nbytes = rows * rowb
        pO = sim.mem.alloc(nbytes)
        a = Asm("w")
        sa, mb, cs = a.reg(), a.reg(), [a.reg() for _ in range(4)]
        a.mov(sa, 0); a.mov(mb, nbytes + 8)
        for r, v in zip(cs, coords):
            a.mov(r, v)
        a.mbar_init(mb, 1); a.mbar_arrive_tx(mb, nbytes)
        a.tma_load_im2col_w(sa, tm, cs, mb, mode=mode, whalo=whalo, woffset=woffset)
        a.mbar_wait(mb, 0)
        g = a.reg(); a.ldc(g, "o")
        a.tma_store_1d(g, sa, nbytes)
        a.cp_commit(); a.cp_wait(0)
        sim.launch(a.build(), 1, 32, {"o": pO}, smem_bytes=nbytes + 16)
        return sim.mem.read(pO, np.float32, rows * C).reshape(rows, C)

    def tm_rows(mode, whalo):
        return 4 * (32 + whalo) if mode == "w128" else ppc + whalo

    def ref(positions, h, n):
        out = np.zeros((len(positions), C), np.float32)
        for r, w in enumerate(positions):
            if 0 <= w < W:
                out[r] = X[n, h, w]
        return out

    # im2col::w: 40 pixels + 2 halo pixels, first pixel shifted by wOffset 3, from W = -2 (partly outside)
    got = run("w", (0, -2, 1, 1), whalo=2, woffset=3)
    np.testing.assert_array_equal(got, ref([1 + k for k in range(42)], 1, 1))
    # im2col::w::128: 128 pixels with 2 halo pixels after every 32 -> 136 rows
    got = run("w128", (0, 5, 2, 0), whalo=2)
    pos = [5 + 32 * g + i for g in range(4) for i in range(34)]
    np.testing.assert_array_equal(got, ref(pos, 2, 0))
    # near the right edge the box runs off the tensor and the rest is zero
    got = run("w", (0, 130, 0, 0))
    np.testing.assert_array_equal(got, ref([130 + k for k in range(40)], 0, 0))
    assert (got[20:] == 0).all()
    # im2col_no_offs::w: allowed with non-negative coordinates, an error with a negative one
    np.testing.assert_array_equal(run("w_no_offs", (0, 10, 1, 0)), ref([10 + k for k in range(40)], 1, 0))
    try:
        run("w_no_offs", (0, -1, 1, 0))
        assert False, "negative coordinates must be rejected"
    except RuntimeError as e:
        assert "negative" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
