"""python3 -m h200sim.test_sim"""

import numpy as np

from . import Asm, H200, Sim, to_bf16_bits, from_bf16_bits
from .kernels import gemv_bf16


def test_gemv_matches_numpy():
    sim = Sim(H200())
    n, k = 64, 4096
    rng = np.random.default_rng(1)
    W = from_bf16_bits(to_bf16_bits(rng.standard_normal((n, k), dtype=np.float32)))
    x = from_bf16_bits(to_bf16_bits(rng.standard_normal(k, dtype=np.float32)))
    pW, px, po = sim.mem.alloc(W.size * 2), sim.mem.alloc(x.size * 2), sim.mem.alloc(n * 4)
    sim.mem.write(pW, to_bf16_bits(W)); sim.mem.write(px, to_bf16_bits(x))
    st = sim.launch(gemv_bf16(k, 2, 4), n // 4, 128, {"W": pW, "x": px, "out": po})
    out = sim.mem.read(po, np.float32, n)
    ref = W.astype(np.float64) @ x.astype(np.float64)
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-5
    assert st.hbm_bytes >= W.size * 2  # every weight byte came from HBM


def test_hmma_fragment_layout():
    """Pack A, B in the PTX m16n8k16 fragment layout on the host, run HMMA, unpack D."""
    sim = Sim(H200())
    rng = np.random.default_rng(2)
    A = from_bf16_bits(to_bf16_bits(rng.standard_normal((16, 16), dtype=np.float32)))
    B = from_bf16_bits(to_bf16_bits(rng.standard_normal((16, 8), dtype=np.float32)))
    Ab, Bb = to_bf16_bits(A), to_bf16_bits(B)
    frag = np.zeros((32, 8), np.uint32)   # a0..a3, b0, b1, 2 pad words per lane
    for lane in range(32):
        g, t = lane // 4, lane % 4
        pk = lambda lo, hi: np.uint32(lo) | (np.uint32(hi) << 16)
        frag[lane, 0] = pk(Ab[g, 2 * t], Ab[g, 2 * t + 1])
        frag[lane, 1] = pk(Ab[g + 8, 2 * t], Ab[g + 8, 2 * t + 1])
        frag[lane, 2] = pk(Ab[g, 2 * t + 8], Ab[g, 2 * t + 9])
        frag[lane, 3] = pk(Ab[g + 8, 2 * t + 8], Ab[g + 8, 2 * t + 9])
        frag[lane, 4] = pk(Bb[2 * t, g], Bb[2 * t + 1, g])
        frag[lane, 5] = pk(Bb[2 * t + 8, g], Bb[2 * t + 9, g])
    pin = sim.mem.alloc(frag.nbytes); sim.mem.write(pin, frag)
    pout = sim.mem.alloc(32 * 16)

    a = Asm("mma")
    lane, addr, t = a.reg(), a.reg(), a.reg()
    ra, rb, rc, rd = a.reg(4), a.reg(2), a.reg(4), a.reg(4)
    a.s2r(lane, "laneid")
    a.ldc(t, "in")
    a.imad(addr, lane, 32, t)
    a.ldg(ra, addr, 0, 16)
    a.ldg(rb, addr, 16, 8)
    for r in range(4):
        a.mov(rc + r, 0)
    a.hmma(rd, ra, rb, rc)
    a.ldc(t, "out")
    a.imad(addr, lane, 16, t)
    a.stg(addr, rd, 0, 16)
    sim.launch(a.build(), 1, 32, {"in": pin, "out": pout})

    D = np.zeros((16, 8), np.float32)
    got = sim.mem.read(pout, np.float32, 128).reshape(32, 4)
    for lane in range(32):
        g, t = lane // 4, lane % 4
        D[g, 2 * t], D[g, 2 * t + 1] = got[lane, 0], got[lane, 1]
        D[g + 8, 2 * t], D[g + 8, 2 * t + 1] = got[lane, 2], got[lane, 3]
    np.testing.assert_allclose(D, A @ B, rtol=1e-5, atol=1e-5)


def test_barrier_and_shared_memory():
    """Each thread writes tid to smem, syncs, reads its neighbour's value."""
    sim = Sim(H200())
    po = sim.mem.alloc(256 * 4)
    a = Asm("bar")
    tid, addr, v, nb, o, t = (a.reg() for _ in range(6))
    a.s2r(tid, "tid")
    a.shl(addr, tid, 2)
    a.sts(addr, tid)
    a.bar()
    a.iadd(nb, tid, 1)
    a.and_(nb, nb, 255)
    a.shl(nb, nb, 2)
    a.lds(v, nb)
    a.ldc(t, "out")
    a.iadd(o, addr, t)
    a.stg(o, v)
    sim.launch(a.build(), 1, 256, {"out": po}, smem_bytes=1024)
    got = sim.mem.read(po, np.uint32, 256)
    np.testing.assert_array_equal(got, (np.arange(256) + 1) % 256)


def _run1(sim, a, block, params, smem=0, grid=1, cluster=1):
    return sim.launch(a.build(), grid, block, params, smem_bytes=smem, cluster=cluster)


def test_divergence_reconverges():
    """lane odd -> 2*lane, even -> lane+100; both paths run with their own masks, then rejoin."""
    sim = Sim(H200())
    po = sim.mem.alloc(128)
    a = Asm("div")
    lane, odd, v, addr, t = (a.reg() for _ in range(5))
    a.s2r(lane, "laneid")
    a.and_(odd, lane, 1)
    a.brnz(odd, "odd", reconv="join")
    a.iadd(v, lane, 100)            # fall-through: even lanes
    a.bra("join")
    a.label("odd")
    a.shl(v, lane, 1)
    a.label("join")
    a.iadd(v, v, 1000)              # all lanes again
    a.ldc(t, "out")
    a.imad(addr, lane, 4, t)
    a.stg(addr, v)
    _run1(sim, a, 32, {"out": po})
    lane = np.arange(32)
    np.testing.assert_array_equal(sim.mem.read(po, np.uint32, 32), np.where(lane % 2, 2 * lane, lane + 100) + 1000)


def test_cp_async_lands_only_after_wait():
    """cp.async data is not visible until the group completes: reading early returns the old smem."""
    sim = Sim(H200())
    src = sim.mem.alloc(512); sim.mem.write(src, np.arange(1, 129, dtype=np.uint32))
    po = sim.mem.alloc(512)
    a = Asm("cpasync")
    lane, soff, goff, early, late, t, o = (a.reg() for _ in range(7))
    a.s2r(lane, "laneid")
    a.shl(soff, lane, 4)
    a.ldc(t, "src")
    a.iadd(goff, soff, t)
    a.ldgsts(soff, goff, 16)
    a.cp_commit()
    a.lds(early, soff)              # before the wait: stale zero
    a.cp_wait(0)
    a.lds(late, soff)
    a.ldc(t, "out")
    a.shl(o, lane, 2)
    a.iadd(o, o, t)
    a.stg(o, early)
    a.stg(o, late, 128)
    _run1(sim, a, 32, {"src": src, "out": po}, smem=512)
    got = sim.mem.read(po, np.uint32, 64)
    assert (got[:32] == 0).all()
    np.testing.assert_array_equal(got[32:], np.arange(32) * 4 + 1)


def test_tma_1d_mbarrier_and_store():
    sim = Sim(H200())
    n = 1024
    src = sim.mem.alloc(n); sim.mem.write(src, np.arange(n // 4, dtype=np.uint32))
    dst = sim.mem.alloc(n)
    a = Asm("tma1d")
    sa, ga, mb, gd = (a.reg() for _ in range(4))
    a.mov(mb, n)                     # mbarrier lives after the data
    a.mov(sa, 0)
    a.mbar_init(mb, 1)
    a.mbar_arrive_tx(mb, n)
    a.ldc(ga, "src")
    a.tma_load_1d(sa, ga, n, mb)
    a.mbar_wait(mb, 0)
    a.ldc(gd, "dst")
    a.tma_store_1d(gd, sa, n)
    a.cp_commit()
    a.cp_wait(0)
    st = _run1(sim, a, 32, {"src": src, "dst": dst}, smem=n + 16)
    np.testing.assert_array_equal(sim.mem.read(dst, np.uint32, n // 4), np.arange(n // 4))
    assert st.cycles > sim.cfg.lat_hbm   # a cold load really paid DRAM latency


def test_cluster_dsmem_and_multicast():
    sim = Sim(H200())
    src = sim.mem.alloc(256); sim.mem.write(src, np.arange(64, dtype=np.uint32) + 7)
    po = sim.mem.alloc(2 * 32 * 4 * 2)
    a = Asm("cluster")
    rank, lane, mb, sa, ga, v, peer, t, o, cta = (a.reg() for _ in range(10))
    a.s2r(rank, "clusterrank")
    a.s2r(lane, "laneid")
    a.s2r(cta, "ctaid")
    a.mov(mb, 256)
    a.mbar_init(mb, 1)
    a.mbar_arrive_tx(mb, 256)       # every CTA expects the multicast payload
    a.cluster_sync()                # both CTAs initialised before anyone sends
    a.setp_eq(t, rank, 0)
    a.brnz(t, "send", reconv="recv")
    a.bra("recv")
    a.label("send")
    a.mov(sa, 0)
    a.ldc(ga, "src")
    a.tma_load_1d(sa, ga, 256, mb, mcast=0b11)    # one global read, lands in both CTAs
    a.label("recv")
    a.mbar_wait(mb, 0)
    # DSMEM: read the peer CTA's copy of word `lane`
    a.xor(peer, rank, 1)
    a.shl(t, lane, 2)
    a.lds_cl(v, t, peer)
    a.ldc(o, "out")
    a.imad(t, cta, 32, lane)
    a.imad(t, t, 4, o)
    a.stg(t, v)
    st = _run1(sim, a, 32, {"src": src, "out": po}, smem=512, grid=2, cluster=2)
    got = sim.mem.read(po, np.uint32, 64)
    np.testing.assert_array_equal(got, np.concatenate([np.arange(32) + 7] * 2))
    assert st.l2_miss * 32 <= 512   # the multicast read global memory once, not twice


def test_gemm_tma_wgmma():
    from .kernels import gemm_tma_wgmma
    sim = Sim(H200())
    n, k, bn = 128, 512, 32
    rng = np.random.default_rng(3)
    x = from_bf16_bits(to_bf16_bits(rng.standard_normal((64, k), dtype=np.float32)))
    W = from_bf16_bits(to_bf16_bits(rng.standard_normal((n, k), dtype=np.float32) / 16))
    px, pW, po = sim.mem.alloc(x.size * 2), sim.mem.alloc(W.size * 2), sim.mem.alloc(64 * n * 4)
    sim.mem.write(px, to_bf16_bits(x)); sim.mem.write(pW, to_bf16_bits(W))
    ta = sim.tensor_map(px, 64, k, 2, 64, 64, swizzle=128)
    tb = sim.tensor_map(pW, n, k, 2, bn, 64, swizzle=128)
    prog, smem = gemm_tma_wgmma(n, k, bn, ta, tb)
    st = sim.launch(prog, n // bn, 128, {"out": po}, smem_bytes=smem)
    out = sim.mem.read(po, np.float32, 64 * n).reshape(64, n)
    np.testing.assert_allclose(out, x @ W.T, rtol=1e-4, atol=1e-4)
    assert st.op_counts["WGMMA"] == 4 * 4 * (k // 64) * (n // bn)


def test_tma_swizzle_layout():
    """A 128 B-swizzled TMA box lands with 16 B chunk c of row r at chunk c ^ (r % 8)."""
    sim = Sim(H200())
    tile = np.arange(8 * 64, dtype=np.uint16).reshape(8, 64)          # 8 rows x 128 B
    src = sim.mem.alloc(tile.nbytes); sim.mem.write(src, tile)
    po = sim.mem.alloc(1024)
    tm = sim.tensor_map(src, 8, 64, 2, 8, 64, swizzle=128)
    a = Asm("swz")
    sa, mb, z, lane, addr, v, t, o = (a.reg() for _ in range(8))
    a.mov(sa, 0); a.mov(mb, 1024); a.mov(z, 0)
    a.mbar_init(mb, 1)
    a.mbar_arrive_tx(mb, 1024)
    a.tma_load_2d(sa, tm, z, z, mb)
    a.mbar_wait(mb, 0)
    a.s2r(lane, "laneid")
    a.ldc(t, "out")
    for it in range(8):
        a.imad(addr, lane, 4, it * 128)
        a.lds(v, addr)
        a.iadd(o, addr, t)
        a.stg(o, v)
    _run1(sim, a, 32, {"out": po}, smem=1024 + 16)
    got = sim.mem.read(po, np.uint8, 1024)
    want = np.zeros(1024, np.uint8)
    flat = tile.view(np.uint8).ravel()
    for i in range(1024):
        want[i ^ (((i >> 7) & 7) << 4)] = flat[i]
    np.testing.assert_array_equal(got, want)
    assert not np.array_equal(got, flat)      # it really is permuted


def test_wgmma_a_from_registers():
    sim = Sim(H200())
    n = 16
    rng = np.random.default_rng(4)
    A = from_bf16_bits(to_bf16_bits(rng.standard_normal((64, 16), dtype=np.float32)))
    B = from_bf16_bits(to_bf16_bits(rng.standard_normal((n, 16), dtype=np.float32)))
    Ab = to_bf16_bits(A)
    frag = np.zeros((128, 4), np.uint32)
    pk = lambda lo, hi: np.uint32(lo) | (np.uint32(hi) << 16)
    for tid in range(128):
        wi, lane = tid // 32, tid % 32
        g, t = lane // 4, lane % 4
        r = 16 * wi
        frag[tid] = [pk(Ab[r + g, 2 * t], Ab[r + g, 2 * t + 1]), pk(Ab[r + g + 8, 2 * t], Ab[r + g + 8, 2 * t + 1]),
                     pk(Ab[r + g, 2 * t + 8], Ab[r + g, 2 * t + 9]), pk(Ab[r + g + 8, 2 * t + 8], Ab[r + g + 8, 2 * t + 9])]
    pa = sim.mem.alloc(frag.nbytes); sim.mem.write(pa, frag)
    pb = sim.mem.alloc(B.size * 2); sim.mem.write(pb, to_bf16_bits(B))
    po = sim.mem.alloc(128 * (n // 2) * 4)
    a = Asm("wgmma_regs")
    tid, t, addr, v, bsm = (a.reg() for _ in range(5))
    ra, acc = a.reg(4), a.reg(n // 2)
    a.s2r(tid, "tid")
    a.ldc(t, "A")
    a.imad(addr, tid, 16, t)
    a.ldg(ra, addr, 0, 16)
    a.ldc(t, "B")                       # B tile: 128 words, one per thread, into shared memory
    a.imad(addr, tid, 4, t)
    a.ldg(v, addr, 0, 4)
    a.shl(bsm, tid, 2)
    a.sts(bsm, v)
    a.fence()                           # fence.proxy.async: wgmma reads shared memory through the async proxy
    a.bar()
    a.mov(bsm, 0)
    a.wgmma_fence()
    a.wgmma(acc, ra, bsm, n, 0, 32, accumulate=False, a_in_regs=True)
    a.wgmma_commit()
    a.wgmma_wait(0)
    a.ldc(t, "out")
    a.imad(addr, tid, (n // 2) * 4, t)
    for r in range(n // 2):
        a.stg(addr, acc + r, r * 4)
    _run1(sim, a, 128, {"A": pa, "B": pb, "out": po}, smem=512)
    assert not sim.hazards
    got = sim.mem.read(po, np.float32, 128 * (n // 2)).reshape(128, n // 2)
    D = np.zeros((64, n), np.float32)
    for tid in range(128):
        wi, lane = tid // 32, tid % 32
        g, t = lane // 4, lane % 4
        for j in range(n // 8):
            for r in range(4):
                D[16 * wi + g + 8 * (r >> 1), 8 * j + 2 * t + (r & 1)] = got[tid, 4 * j + r]
    np.testing.assert_allclose(D, A @ B.T, rtol=1e-5, atol=1e-5)


def test_tma_store_2d_reduce_and_prefetch():
    sim = Sim(H200())
    rows, cols = 16, 32
    po = sim.mem.alloc(rows * cols * 4)
    sim.mem.write(po, np.full((rows, cols), 5.0, np.float32))
    tm = sim.tensor_map(po, rows, cols, 4, 8, 8)
    a = Asm("tma_st2d")
    lane, one, addr, c0, r0, c1, r1, z = (a.reg() for _ in range(8))
    v = a.reg(2)
    a.s2r(lane, "laneid")
    a.mov(z, 0); a.mov(c0, 28); a.mov(r0, 12); a.mov(c1, 8); a.mov(r1, 4)
    a.tma_prefetch_2d(tm, z, z)
    a.mov(v, 0x3F800000); a.mov(v + 1, 0x3F800000)      # 1.0f
    a.shl(addr, lane, 3)
    a.sts(addr, v, 0, 8)                                # 8x8 box of ones (64 floats)
    a.fence()
    a.tma_store_2d(z, tm, c1, r1)                       # plain store, fully inside
    a.tma_store_2d(z, tm, c0, r0, reduce=True)          # reduce-add, clipped at the tensor corner
    a.cp_commit()
    a.cp_wait(0)
    st = _run1(sim, a, 32, {}, smem=256)
    got = sim.mem.read(po, np.float32, rows * cols).reshape(rows, cols)
    want = np.full((rows, cols), 5.0, np.float32)
    want[4:12, 8:16] = 1.0
    want[12:16, 28:32] = 6.0
    np.testing.assert_array_equal(got, want)
    assert st.l2_miss > 0                                # the prefetch really read the tensor


def test_exit_inside_divergent_region():
    sim = Sim(H200())
    po = sim.mem.alloc(128)
    sim.mem.write(po, np.full(32, 0xDEADBEEF, np.uint32))
    a = Asm("divexit")
    lane, p, v, t, addr = (a.reg() for _ in range(5))
    a.s2r(lane, "laneid")
    a.setp_lt(p, lane, 8)
    a.brnz(p, "die", reconv="join")
    a.iadd(v, lane, 500)
    a.bra("join")
    a.label("die")
    a.exit()                                            # lanes 0..7 retire
    a.label("join")
    a.ldc(t, "out")
    a.imad(addr, lane, 4, t)
    a.stg(addr, v)
    _run1(sim, a, 32, {"out": po})
    got = sim.mem.read(po, np.uint32, 32)
    assert (got[:8] == 0xDEADBEEF).all()
    np.testing.assert_array_equal(got[8:], np.arange(8, 32) + 500)


def test_dirty_lines_written_back_on_eviction():
    sim = Sim(H200(l2_bytes=64 * 1024))                 # 32 sets x 16 ways x 128 B
    po = sim.mem.alloc(256 * 1024)
    a = Asm("wb")
    lane, addr, i, p, t = (a.reg() for _ in range(5))
    a.s2r(lane, "laneid")
    a.ldc(t, "out")
    a.imad(addr, lane, 4, t)
    a.mov(i, 0)
    a.label("loop")
    a.stg(addr, lane)
    a.iadd(addr, addr, 128)
    a.iadd(i, i, 1)
    a.setp_lt(p, i, 2048)
    a.brnz(p, "loop")
    st = _run1(sim, a, 32, {"out": po})
    assert st.hbm_write_bytes >= 150_000                # ~256 KB written, only 64 KB fits


def test_64bit_pointers_above_4gib():
    base = 1 << 40
    sim = Sim(H200(va_base=base))
    src = sim.mem.alloc(128); sim.mem.write(src, np.arange(32, dtype=np.uint32) * 3)
    dst = sim.mem.alloc(128)
    assert src >= base
    a = Asm("wide")
    lane, v = a.reg(), a.reg()
    ps, pd = a.reg(2), a.reg(2)
    a.s2r(lane, "laneid")
    a.ldc64(ps, "src")
    a.ldc64(pd, "dst")
    a.imad_wide(ps, lane, 4, ps)
    a.imad_wide(pd, lane, 4, pd)
    a.ldg(v, ps, 0, 4, wide=True)
    a.stg(pd, v, 0, 4, wide=True)
    _run1(sim, a, 32, {"src": src, "dst": dst})
    np.testing.assert_array_equal(sim.mem.read(dst, np.uint32, 32), np.arange(32) * 3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
