"""Scalar / control extensions. python3 -m h200sim.test_ops"""

import numpy as np

from . import Asm, H200, Sim, to_bf16_bits, from_bf16_bits
from . import tensor as T


def run(build, block=32, grid=1, smem=0, nout=32, params=None, cfg=None, init=None, cluster=1, sim=None):
    """build(a, lane, out_ptr_reg) emits code; returns the uint32 output words (one per output slot)."""
    sim = sim or Sim(cfg or H200())
    po = sim.mem.alloc(4 * nout)
    if init is not None:
        sim.mem.write(po, init)
    a = Asm("t")
    lane, optr = a.reg(), a.reg()
    a.s2r(lane, "laneid")
    a.ldc(optr, "out")
    build(a, lane, optr)
    p = {"out": po}
    p.update(params or {})
    st = sim.launch(a.build(), grid, block, p, smem_bytes=smem, cluster=cluster)
    return sim.mem.read(po, np.uint32, nout), sim, st


def store_lane(a, lane, optr, val, offset_words=0):
    t = a.reg()
    a.imad(t, lane, 4, optr)
    a.stg(t, val, offset_words * 4)


def f32(x): return np.asarray(x, np.float32).view(np.uint32)


def test_predicates_guards_and_select():
    def build(a, lane, optr):
        p, q, r = a.preg(), a.preg(), a.preg()
        v, w_ = a.reg(), a.reg()
        a.mov(v, 1000)
        a.isetp("lt", p, lane, 16)
        a.isetp("ge", q, lane, 8)
        a.psetp("and", r, p, q)                 # lanes 8..15
        with a.guard(r):
            a.mov(v, 7)                         # only lanes 8..15
        with a.guard(p, negate=True):
            a.iadd(v, v, 1)                     # lanes 16..31
        a.sel(w_, v, lane, q)                   # q ? v : lane
        store_lane(a, lane, optr, w_)
    out, _, _ = run(build)
    lane = np.arange(32)
    v = np.where((lane >= 8) & (lane < 16), 7, 1000) + (lane >= 16)
    np.testing.assert_array_equal(out, np.where(lane >= 8, v, lane))


def test_guarded_branch_diverges_and_reconverges():
    def build(a, lane, optr):
        p = a.preg()
        v = a.reg()
        a.mov(v, 0)
        a.isetp("lt", p, lane, 5)
        with a.guard(p):
            a.bra("small", reconv="end")
        a.iadd(v, lane, 100)                    # lanes >= 5
        a.bra("end")
        a.label("small")
        a.iadd(v, lane, 500)                    # lanes < 5
        a.label("end")
        a.iadd(v, v, 1)
        store_lane(a, lane, optr, v)
    out, _, _ = run(build)
    lane = np.arange(32)
    np.testing.assert_array_equal(out, np.where(lane < 5, lane + 500, lane + 100) + 1)


def test_guarded_exit_and_vote_elect_activemask():
    def build(a, lane, optr):
        p, e = a.preg(), a.preg()
        ball, anyv, allv, am, t = (a.reg() for _ in range(5))
        a.isetp("lt", p, lane, 12)
        a.vote_ballot(ball, p)
        a.vote_any(anyv, p)
        a.vote_all(allv, p)
        with a.guard(p):
            a.exit()                            # lanes 0..11 retire
        a.activemask(am)                        # only lanes 12..31 remain
        a.elect(e)
        v = a.reg()
        a.mov(v, 0)
        with a.guard(e):
            a.mov(v, 1)
        a.imad(t, lane, 16, optr)
        a.stg(t, ball, 0); a.stg(t, am, 4); a.stg(t, v, 8); a.stg(t, allv, 12)
    out, _, _ = run(build, nout=128, init=np.full(128, 0xDEADBEEF, np.uint32))
    out = out.reshape(32, 4)
    assert (out[:12] == 0xDEADBEEF).all()                              # retired lanes never stored
    assert (out[12:, 0] == 0xFFF).all()                                # ballot taken before the exit
    assert (out[12:, 1] == 0xFFFFF000).all()                           # activemask: lanes 12..31
    assert out[12, 2] == 1 and (out[13:, 2] == 0).all()                # elect picks the lowest live lane
    assert (out[12:, 3] == 0).all()                                    # not all lanes had p


def test_fp64_and_packed_half():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(32); B = rng.standard_normal(32); C = rng.standard_normal(32)
    sim = Sim(H200())
    pin = sim.mem.alloc(32 * 8 * 3)
    sim.mem.write(pin, np.stack([A, B, C], 1))
    H = rng.standard_normal((32, 3)).astype(np.float16).view(np.uint16)
    Hb = to_bf16_bits(rng.standard_normal((32, 3)).astype(np.float32))
    ph = sim.mem.alloc(32 * 12)
    hp = np.zeros((32, 3), np.uint32)
    hp[:] = H.astype(np.uint32) | (H[:, ::-1].astype(np.uint32) << 16)
    sim.mem.write(ph, hp)
    pb = sim.mem.alloc(32 * 12)
    bp = Hb.astype(np.uint32) | (Hb[:, ::-1].astype(np.uint32) << 16)
    sim.mem.write(pb, bp)

    def build(a, lane, optr):
        t, addr = a.reg(), a.reg()
        x, y, z, r = a.reg(2), a.reg(2), a.reg(2), a.reg(2)
        a.ldc(t, "in"); a.imad(addr, lane, 24, t)
        a.ldg(x, addr, 0, 8); a.ldg(y, addr, 8, 8); a.ldg(z, addr, 16, 8)
        a.dfma(r, x, y, z)
        a.imad(t, lane, 8, optr); a.stg(t, r, 0, 8)
        h, hb = a.reg(3), a.reg(3)
        o1, o2 = a.reg(), a.reg()
        a.ldc(t, "h"); a.imad(addr, lane, 12, t)
        a.ldg(h, addr, 0, 4); a.ldg(h + 1, addr, 4, 4); a.ldg(h + 2, addr, 8, 4)
        a.hfma2(o1, h, h + 1, h + 2)
        a.ldc(t, "b"); a.imad(addr, lane, 12, t)
        a.ldg(hb, addr, 0, 4); a.ldg(hb + 1, addr, 4, 4); a.ldg(hb + 2, addr, 8, 4)
        a.hfma2(o2, hb, hb + 1, hb + 2, bf16=True)
        a.imad(t, lane, 8, optr)
        a.stg(t, o1, 512); a.stg(t, o2, 516)
    out, sim, _ = run(build, nout=192, params={"in": pin, "h": ph, "b": pb}, sim=sim)
    d = np.frombuffer(out[:64].tobytes(), np.float64)
    np.testing.assert_allclose(d, A * B + C, rtol=1e-14)
    words = out[128:].reshape(32, 2)
    # f16x2: both halves are an FMA rounded once to f16
    for half, shift in ((0, 0), (1, 16)):
        a_, b_, c_ = [((hp[:, i] >> shift) & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64) for i in range(3)]
        got = ((words[:, 0] >> shift) & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
        np.testing.assert_array_equal(got, (a_ * b_ + c_).astype(np.float16).astype(np.float64))
    for shift in (0, 16):
        a_, b_, c_ = [from_bf16_bits(((bp[:, i] >> shift) & 0xFFFF).astype(np.uint16)).astype(np.float64) for i in range(3)]
        got = from_bf16_bits(((words[:, 1] >> shift) & 0xFFFF).astype(np.uint16))
        np.testing.assert_array_equal(got, from_bf16_bits(to_bf16_bits((a_ * b_ + c_).astype(np.float32))))


def test_conversions_and_mufu():
    def build(a, lane, optr):
        t = a.reg()
        fv, iv, back, pk_b, pk_h, pk_8, up, rcp, ex2 = (a.reg() for _ in range(9))
        a.iadd(iv, lane, -10)                    # -10 .. 21
        a.i2f(fv, iv)
        a.fmul(fv, fv, 1.7)
        a.f2i(back, fv)
        a.f2fp(pk_b, fv, fv, "bf16")
        a.f2fp(pk_h, fv, fv, "f16")
        a.f2fp(pk_8, fv, fv, "e4m3")
        a.cvt_f16x2_fp8(up, pk_8, "e4m3")
        a.mufu("rcp", rcp, fv)
        lf = a.reg(); a.i2f(lf, lane)
        a.mufu("ex2", ex2, lf)
        a.imad(t, lane, 32, optr)
        for i, r in enumerate((back, pk_b, pk_h, up, rcp, ex2)):
            a.stg(t, r, 4 * i)
    out, _, _ = run(build, nout=32 * 8)
    o = out.reshape(32, 8)
    lane = np.arange(32); fv = (lane - 10).astype(np.float32) * np.float32(1.7)
    np.testing.assert_array_equal(o[:, 0].view(np.int32), np.trunc(fv).astype(np.int32))
    np.testing.assert_array_equal(o[:, 1] >> 16, to_bf16_bits(fv).astype(np.uint32))
    np.testing.assert_array_equal(o[:, 2] & 0xFFFF, fv.astype(np.float16).view(np.uint16))
    up = o[:, 3].astype(np.uint32)
    codes = T.encode_fp8(fv, T.E4M3)
    np.testing.assert_array_equal((up & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32), T.E4M3[codes])
    with np.errstate(divide="ignore"):
        np.testing.assert_allclose(o[:, 4].view(np.float32), 1 / fv, rtol=1e-6)
    np.testing.assert_allclose(o[:, 5].view(np.float32), 2.0 ** lane, rtol=1e-6)


def test_minmax_dpx_and_bit_ops():
    def build(a, lane, optr):
        t = a.reg()
        x, y, z = a.reg(), a.reg(), a.reg()
        a.iadd(x, lane, -16)                      # signed -16..15
        a.imnmx(y, x, 3, "max")
        a.imnmx(z, x, 3, "min", unsigned=True)
        d1, d2, p1, l3, pc, br, fl, ab, mh = (a.reg() for _ in range(9))
        # s16x2: a = (lane-16 , 7) packed, b = (2, -3), c = (0, 10): max(a+b, c) per half, relu
        pa, pb_, pcc = a.reg(), a.reg(), a.reg()
        a.shl(pa, 7, 16); a.and_(t, x, 0xFFFF); a.or_(pa, pa, t)
        a.mov(pb_, 0xFFFD0002)                    # (-3 << 16) | 2
        a.mov(pcc, 0x000A0000)                    # (10 << 16) | 0
        a.viaddmnmx(d1, pa, pb_, pcc, "max", "s16x2", relu=True)
        a.vimnmx3(d2, x, 5, 9, "max", "s32")
        a.prmt(p1, 0x04030201, 0x08070605, 0x7531)    # pick bytes 1,3,5,7 -> 0x08060402... order low->high
        a.lop3(l3, x, 0x0F0F0F0F, 0xFFFF0000, 0xE8)   # majority(a, b, c)
        a.popc(pc, x); a.brev(br, lane); a.flo(fl, lane); a.iabs(ab, x)
        a.imul_hi(mh, x, 0x40000000, unsigned=False)
        a.imad(t, lane, 48, optr)
        for i, r in enumerate((y, z, d1, d2, p1, l3, pc, br, fl, ab, mh)):
            a.stg(t, r, 4 * i)
    out, _, _ = run(build, nout=32 * 12)
    o = out.reshape(32, 12)
    lane = np.arange(32); x = (lane - 16).astype(np.int32)
    np.testing.assert_array_equal(o[:, 0].view(np.int32), np.maximum(x, 3))
    np.testing.assert_array_equal(o[:, 1], np.minimum(x.view(np.uint32), 3))
    np.testing.assert_array_equal(o[:, 2] & 0xFFFF, np.maximum(np.maximum(x + 2, 0), 0).astype(np.uint32) & 0xFFFF)
    assert (o[:, 2] >> 16 == 10).all()
    np.testing.assert_array_equal(o[:, 3].view(np.int32), np.maximum(np.maximum(x, 5), 9))
    xu = x.view(np.uint32)
    maj = (xu & 0x0F0F0F0F) | (xu & 0xFFFF0000) | (0x0F0F0F0F & 0xFFFF0000)                # lut 0xE8 = majority
    np.testing.assert_array_equal(o[:, 5], maj)
    np.testing.assert_array_equal(o[:, 6], [bin(int(v) & 0xFFFFFFFF).count("1") for v in x])
    np.testing.assert_array_equal(o[:, 7], [int(f"{i:032b}"[::-1], 2) for i in lane])
    np.testing.assert_array_equal(o[:, 8], [(int(i).bit_length() - 1) & 0xFFFFFFFF for i in lane])
    np.testing.assert_array_equal(o[:, 9].view(np.int32), np.abs(x))
    np.testing.assert_array_equal(o[:, 10].view(np.int32), (x.astype(np.int64) * 0x40000000) >> 32)


def test_prmt_selector_semantics():
    def build(a, lane, optr):
        r1, r2 = a.reg(), a.reg()
        a.prmt(r1, 0x44332211, 0x88776655, 0x7531)    # bytes 1,3,5,7 of {b:a}
        a.prmt(r2, 0x000000F0, 0x0, 0x8888 | 0x0000)  # sign-replicate byte 0 of a
        store_lane(a, lane, optr, r1)
        store_lane(a, lane, optr, r2, 32)
    out, _, _ = run(build, nout=64)
    assert out[0] == 0x88664422        # byte pool a=[11,22,33,44] b=[55,66,77,88]: picks 22,44,66,88 low->high
    assert out[32] == 0xFFFFFFFF


def test_shuffles_and_redux():
    def build(a, lane, optr):
        up, dn, idx, wdn, red, mx = (a.reg() for _ in range(6))
        a.shfl_up(up, lane, 3)
        a.shfl_down(dn, lane, 2)
        a.shfl_idx(idx, lane, 5)
        a.shfl_down(wdn, lane, 1, width=8)
        a.redux(red, lane, "add"); a.redux(mx, lane, "max")
        t = a.reg(); a.imad(t, lane, 24, optr)
        for i, r in enumerate((up, dn, idx, wdn, red, mx)):
            a.stg(t, r, 4 * i)
    out, _, _ = run(build, nout=32 * 6)
    o = out.reshape(32, 6); lane = np.arange(32)
    np.testing.assert_array_equal(o[:, 0], np.where(lane >= 3, lane - 3, lane))
    np.testing.assert_array_equal(o[:, 1], np.where(lane + 2 < 32, lane + 2, lane))
    assert (o[:, 2] == 5).all()
    np.testing.assert_array_equal(o[:, 3], np.where((lane % 8) + 1 < 8, lane + 1, lane))
    assert (o[:, 4] == lane.sum()).all() and (o[:, 5] == 31).all()


def test_atomics_global_and_shared():
    def build(a, lane, optr):
        old, old2, oldf, cas, t, zero, so = (a.reg() for _ in range(7))
        a.mov(zero, 0)
        a.atom(old, optr, 1, "add")                   # every lane adds 1 to out[0]: old values 0..31 in lane order
        a.mov(t, 0x3F800000)
        a.imad(so, zero, 1, optr)
        a.atom(oldf, optr, t, "add", fmt="f32", imm=4)
        a.mov(t, 5)
        a.atom(cas, optr, t, "cas", cmp=zero, imm=8)  # only the first lane's compare (0) matches
        a.atoms(old2, zero, 3, "add")
        w = a.reg()
        a.imad(w, lane, 16, optr)
        a.stg(w, old, 16); a.stg(w, cas, 20); a.stg(w, old2, 24)
    out, _, _ = run(build, nout=4 + 32 * 4, smem=16)
    assert out[0] == 32
    assert np.frombuffer(out[1:2].tobytes(), np.float32)[0] == 32.0
    assert out[2] == 5
    rec = out[4:].reshape(32, 4)                       # base is out + 16 B
    np.testing.assert_array_equal(rec[:, 0], np.arange(32))                    # atomic add returned old values
    assert rec[0, 1] == 0 and (rec[1:, 1] == 5).all()                          # cas: first lane saw 0, the rest 5
    np.testing.assert_array_equal(rec[:, 2], np.arange(32) * 3)                # shared atomic


def test_constant_bank_and_serialization():
    sim = Sim(H200())
    sim.write_constant(0, np.arange(256, dtype=np.uint32) * 11)

    def build(a, lane, optr):
        same, diff, addr = a.reg(), a.reg(), a.reg()
        a.ldc_mem(same, addr)                          # R? uninitialised = 0: uniform address 0
        a.shl(addr, lane, 2)
        a.ldc_mem(diff, addr)                          # 32 distinct addresses
        t = a.reg(); a.imad(t, lane, 8, optr)
        a.stg(t, same, 0); a.stg(t, diff, 4)
    out, sim, _ = run(build, nout=64, sim=sim)
    o = out.reshape(32, 2)
    assert (o[:, 0] == 0).all()
    np.testing.assert_array_equal(o[:, 1], np.arange(32) * 11)


def test_ldmatrix_stmatrix_match_mma_fragments():
    rng = np.random.default_rng(3)
    A = to_bf16_bits(rng.standard_normal((16, 16)).astype(np.float32))        # 16x16 b16, 32 B rows
    sim = Sim(H200())
    pa = sim.mem.alloc(512); sim.mem.write(pa, A)
    po2 = sim.mem.alloc(512)

    def build(a, lane, optr):
        t, addr, src = a.reg(), a.reg(), a.reg()
        frag = a.reg(4)
        # stage the tile into shared memory, one 16 B chunk per lane x 2
        a.ldc(src, "tile")
        a.shl(t, lane, 4); a.iadd(addr, src, t)
        a.reg(); r4 = a.reg(4)
        a.ldg(r4, addr, 0, 16)
        a.sts(t, r4, 0, 16)
        a.bar()
        # ldmatrix.x4: lane -> row (lane % 16), column block (lane / 16) * 8 elements
        a.and_(addr, lane, 15); a.shl(addr, addr, 5)
        a.shr(t, lane, 4); a.shl(t, t, 4); a.iadd(addr, addr, t)
        a.ldmatrix(frag, addr, 4)
        a.imad(t, lane, 16, optr)
        a.stg(t, frag, 0, 16)
        # stmatrix the same registers back to a different shared region and dump it
        a.and_(addr, lane, 15); a.shl(addr, addr, 5)
        a.shr(t, lane, 4); a.shl(t, t, 4); a.iadd(addr, addr, t); a.iadd(addr, addr, 512)
        a.stmatrix(addr, frag, 4)
        a.bar()
        a.shl(t, lane, 4); a.iadd(addr, t, 512)
        a.lds(r4, addr, 0, 16)
        a.ldc(src, "dump"); a.iadd(addr, src, t)
        a.stg(addr, r4, 0, 16)
    out, sim, _ = run(build, nout=128, params={"tile": pa, "dump": po2}, smem=1024 + 64, sim=sim)
    want = T.pack_a_fragment("bf16", A, 16).T                                  # (32 lanes, 4 regs)
    np.testing.assert_array_equal(out.reshape(32, 4), want)
    np.testing.assert_array_equal(sim.mem.read(po2, np.uint16, 256).reshape(16, 16), A)


def test_uniform_datapath():
    def build(a, lane, optr):
        u1, u2, u3, up = a.ureg(), a.ureg(), a.ureg(), a.ureg()
        a.uldc(u1, "out")
        a.uiadd(u2, u1, 64)                          # uniform address arithmetic
        a.ushl(u3, u1, 0)
        a.uisetp("lt", up, u1, 0)                    # signed compare: addresses are positive -> 0
        v = a.reg(); a.iadd(v, lane, 1000)
        t = a.reg(); a.imad(t, lane, 4, u2)          # uniform register as a source operand
        a.stg(t, v, 0)
        r = a.reg()
        u4 = a.ureg(); a.r2ur(u4, lane)              # R2UR takes the first active lane: lane 0 -> 0
        a.uiadd(u2, u2, u4)
        a.ldg(r, u2, 0, 4)                           # uniform-register address: first lane's slot
        a.imad(t, lane, 4, optr); a.stg(t, r, 0)
        a.brnz(up, "skip"); a.mov(r, 7)
        a.label("skip")
        a.stg(t, r, 128)
    out, _, _ = run(build, nout=64)
    assert (out[:32] == 1000).all()                  # every lane read word 0 of the u2 region (written by lane 0 = 1000)
    assert (out[32:] == 7).all()                     # the uniform predicate was false -> fell through


def test_named_barriers_between_warp_groups():
    def build(a, lane, optr):
        wid, p, t, v, zero = (a.reg() for _ in range(5))
        a.s2r(wid, "warpid")
        a.mov(zero, 0)
        a.setp_eq(p, wid, 0)
        a.brnz(p, "producer")
        a.bar(1, 64)                                  # warp 1 waits until warp 0 has arrived on barrier 1
        a.lds(v, zero)
        a.imad(t, lane, 4, optr); a.stg(t, v, 0)
        a.exit()
        a.label("producer")
        a.nanosleep(500)                              # the consumer reaches the barrier long before this
        a.mov(v, 4242); a.sts(zero, v)
        a.bar_arrive(1, 64)
    out, _, st = run(build, block=64, smem=16)
    assert (out[:32] == 4242).all()
    assert st.cycles > 500


def test_setmaxnreg_moves_registers_between_warpgroups():
    def build_ok(a, lane, optr):
        wid, p, t = a.reg(), a.reg(), a.reg()
        a.s2r(wid, "warpid")
        a.setp_lt(p, wid, 4)
        a.brnz(p, "consumer")
        a.setmaxnreg(24, inc=False)                   # producer warpgroup gives registers back
        a.exit()
        a.label("consumer")
        a.setmaxnreg(104, inc=True)                   # consumer warpgroup takes them
        hi = a.reg(); a.mov(hi, 0)
        a.nxt = 100                                   # allocate a register far above the launch budget
        big = a.reg(); a.mov(big, 99); a.iadd(big, big, 1)
        a.imad(t, lane, 4, optr); a.stg(t, big, 0)
    sim = Sim(H200())
    po = sim.mem.alloc(128)
    a = Asm("nreg")
    lane, optr = a.reg(), a.reg()
    a.s2r(lane, "laneid"); a.ldc(optr, "out")
    build_ok(a, lane, optr)
    st = sim.launch(a.build(launch_regs=64), 1, 256, {"out": po})
    assert (sim.mem.read(po, np.uint32, 32) == 100).all()
    # using the high register before the increase is an error
    sim2 = Sim(H200())
    b = Asm("bad")
    b.nxt = 100
    r = b.reg(); b.mov(r, 1)
    try:
        sim2.launch(b.build(launch_regs=64), 1, 32, {})
        assert False, "expected the register-limit check to fire"
    except RuntimeError as e:
        assert "setmaxnreg" in str(e)


def test_fp64_tensor_core_dmma():
    rng = np.random.default_rng(11)
    A, B, C = rng.standard_normal((8, 4)), rng.standard_normal((4, 8)), rng.standard_normal((8, 8))
    g, t = np.arange(32) >> 2, np.arange(32) & 3
    sim = Sim(H200())
    fa = A[g, t]; fb = B[t, g]
    fc = np.stack([C[g, 2 * t], C[g, 2 * t + 1]], 1)
    pa, pb, pc = (sim.mem.alloc(x.nbytes) for x in (fa, fb, fc))
    for p, x in ((pa, fa), (pb, fb), (pc, fc)):
        sim.mem.write(p, x)

    def build(a, lane, optr):
        t_, addr = a.reg(), a.reg()
        ra, rb, rc, rd = a.reg(2), a.reg(2), a.reg(4), a.reg(4)
        a.ldc(t_, "a"); a.imad(addr, lane, 8, t_); a.ldg(ra, addr, 0, 8)
        a.ldc(t_, "b"); a.imad(addr, lane, 8, t_); a.ldg(rb, addr, 0, 8)
        a.ldc(t_, "c"); a.imad(addr, lane, 16, t_); a.ldg(rc, addr, 0, 8); a.ldg(rc + 2, addr, 8, 8)
        a.dmma(rd, ra, rb, rc)
        a.imad(addr, lane, 16, optr); a.stg(addr, rd, 0, 8); a.stg(addr, rd + 2, 8, 8)
    out, _, _ = run(build, nout=128, params={"a": pa, "b": pb, "c": pc}, sim=sim)
    D = np.zeros((8, 8))
    d = np.frombuffer(out.tobytes(), np.float64).reshape(32, 2)
    D[g, 2 * t], D[g, 2 * t + 1] = d[:, 0], d[:, 1]
    np.testing.assert_allclose(D, A @ B + C, rtol=1e-13)


def test_unit_throughputs():
    """FP64: one warp instruction / 2 cycles per sub-core; MUFU: one / 8 cycles."""
    for build_op, gap in (("dfma", 2), ("mufu", 8)):
        sim = Sim(H200(model_icache=False, model_ccache=False)); sim.log_issue = True
        a = Asm("tp")
        regs = [a.reg(2) for _ in range(12)]
        for i in range(8):
            if build_op == "dfma":
                a.dfma(regs[i + 3], regs[0], regs[1], regs[2])
            else:
                a.mufu("rcp", regs[i + 3], regs[0])
        sim.launch(a.build(), 1, 32, {})
        c = [e[0] for e in sim.issue_log if e[6] in ("DFMA", "MUFU_RCP")]
        assert set(np.diff(c)) == {gap}, (build_op, c)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
