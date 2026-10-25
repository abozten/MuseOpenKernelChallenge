"""Example kernels written for the simulator ISA."""

from .isa import Asm


def gemv_bf16(k: int, unroll: int = 8, warps_per_cta: int = 4):
    """out[n] = sum_k W[n, k] * x[k]. W is bf16 [N, K]; x is bf16 [K]; out is fp32 [N].

    One warp per output row. Each lane loads 16 B (8 bf16) of W and of x per step, `unroll`
    steps in flight before the FMAs (the same streaming shape as the cuBLAS-style GEMV).
    A row is reduced with a butterfly shuffle; every lane then stores the same sum, which
    coalesces to one sector.
    params: W, x, out
    """
    assert k % (256 * unroll) == 0
    iters = k // (256 * unroll)
    step = 512  # bytes of one warp-wide 16 B load
    a = Asm("gemv_bf16")
    lane, wid, cta, t = a.reg(), a.reg(), a.reg(), a.reg()
    row, wp, xp, i, p = a.reg(), a.reg(), a.reg(), a.reg(), a.reg()
    acc = [a.reg(), a.reg()]
    W = [a.reg(4) for _ in range(unroll)]
    X = [a.reg(4) for _ in range(unroll)]
    tmp = [a.reg() for _ in range(8)]

    a.s2r(lane, "laneid")
    a.s2r(wid, "warpid")
    a.s2r(cta, "ctaid")
    a.imad(row, cta, warps_per_cta, wid)                  # row = cta * warps + warp
    a.ldc(t, "W")
    a.imad(wp, row, k * 2, t)                             # &W[row, 0]
    a.shl(t, lane, 4)
    a.iadd(wp, wp, t)                                     # + lane * 16 B
    a.ldc(xp, "x")
    a.iadd(xp, xp, t)
    a.mov(acc[0], 0)
    a.mov(acc[1], 0)
    a.mov(i, 0)

    a.label("loop")
    for u in range(unroll):
        a.ldg(W[u], wp, u * step, 16, cache="cs")         # stream weights past L1
    for u in range(unroll):
        a.ldg(X[u], xp, u * step, 16, cache="ca")         # x is reused: keep in L1
    n = 0
    for u in range(unroll):
        for j in range(4):
            tw0, tx0, tw1, tx1 = tmp[(n % 2) * 4:(n % 2) * 4 + 4]
            n += 1
            a.shl(tw0, W[u] + j, 16)                      # low bf16 -> fp32
            a.shl(tx0, X[u] + j, 16)
            a.and_(tw1, W[u] + j, 0xFFFF0000)             # high bf16 -> fp32
            a.and_(tx1, X[u] + j, 0xFFFF0000)
            a.ffma(acc[0], tw0, tx0, acc[0])
            a.ffma(acc[1], tw1, tx1, acc[1])
    a.iadd(wp, wp, unroll * step)
    a.iadd(xp, xp, unroll * step)
    a.iadd(i, i, 1)
    a.setp_lt(p, i, iters)
    a.brnz(p, "loop")

    a.fadd(acc[0], acc[0], acc[1])
    for m in (16, 8, 4, 2, 1):
        a.shfl_xor(t, acc[0], m)
        a.fadd(acc[0], acc[0], t)
    a.ldc(t, "out")
    a.shl(wp, row, 2)
    a.iadd(wp, wp, t)
    a.stg(wp, acc[0])
    return a.build()


def gemm_tma_wgmma(n: int, k: int, bn: int, tmap_a: int, tmap_b: int, stages: int = 4, bk: int = 64,
                   swizzle: int = 128):
    """out[64, N] (fp32) = x[64, K] @ W[N, K]^T, bf16 inputs, the Hopper way.

    One CTA = one warpgroup (4 warps) computing a 64 x bn tile. Warp 0 is the TMA producer:
    it fills `stages` smem stages (A tile 64 x bk, B tile bn x bk), each guarded by an
    mbarrier. Every warp waits on the stage's barrier, runs bk/16 wgmmas, waits for them,
    syncs the CTA, and warp 0 refills the stage. The accumulator (bn/2 regs/thread) is
    written straight from registers. With swizzle=128 (bk=64) the tensor maps must be created
    with swizzle=128 so TMA and wgmma agree on the shared-memory layout.
    params: out
    """
    assert k % bk == 0 and n % bn == 0 and stages & (stages - 1) == 0 and bn % 8 == 0
    kt_total = k // bk
    a_bytes, b_bytes = 64 * bk * 2, bn * bk * 2
    sb = a_bytes + b_bytes
    mb0 = stages * sb
    a = Asm("gemm_tma_wgmma")
    wid, cta, lane, g, tt = a.reg(), a.reg(), a.reg(), a.reg(), a.reg()
    t, p, q, kt, stage, par = a.reg(), a.reg(), a.reg(), a.reg(), a.reg(), a.reg()
    sa, sbb, mb, aa, bb, nk, col, rown, zero = (a.reg() for _ in range(9))
    acc = a.reg(bn // 2)

    a.s2r(wid, "warpid")
    a.s2r(cta, "ctaid")
    a.s2r(lane, "laneid")
    a.mov(zero, 0)
    a.imad(rown, cta, bn, 0)                              # first W row of this CTA's tile

    def issue(stage_const, kt_const):
        a.mov(mb, mb0 + stage_const * 8)
        a.mov(sa, stage_const * sb)
        a.mov(col, kt_const * bk)
        a.mbar_arrive_tx(mb, sb)
        a.tma_load_2d(sa, tmap_a, col, zero, mb)
        a.iadd(sbb, sa, a_bytes)
        a.tma_load_2d(sbb, tmap_b, col, rown, mb)

    a.setp_lt(p, 0, wid)                                  # p = warp != 0
    a.brnz(p, "start")
    for s in range(stages):
        a.mov(mb, mb0 + s * 8)
        a.mbar_init(mb, 1)
    a.fence()
    for s in range(min(stages, kt_total)):
        issue(s, s)
    a.label("start")
    a.bar()                                               # barriers initialised before anyone waits
    a.mov(kt, 0)

    a.label("loop")
    a.and_(stage, kt, stages - 1)
    a.shr(par, kt, stages.bit_length() - 1)
    a.and_(par, par, 1)
    a.imad(sa, stage, sb, 0)
    a.iadd(sbb, sa, a_bytes)
    a.imad(mb, stage, 8, mb0)
    a.mbar_wait(mb, par)
    a.wgmma_fence()
    for kk in range(bk // 16):
        a.iadd(aa, sa, kk * 32)
        a.iadd(bb, sbb, kk * 32)
        a.wgmma(acc, aa, bb, bn, bk * 2, bk * 2, accumulate=True, swizzle=swizzle)
    a.wgmma_commit()
    a.wgmma_wait(0)
    a.bar()                                               # everyone is done reading this stage
    a.iadd(nk, kt, stages)
    a.setp_eq(q, wid, 0)
    a.setp_lt(p, nk, kt_total)
    a.and_(q, q, p)
    a.brnz(q, "refill")
    a.bra("next")
    a.label("refill")
    a.mbar_arrive_tx(mb, sb)
    a.shl(col, nk, 6 if bk == 64 else bk.bit_length() - 1)
    a.tma_load_2d(sa, tmap_a, col, zero, mb)
    a.tma_load_2d(sbb, tmap_b, col, rown, mb)
    a.label("next")
    a.iadd(kt, kt, 1)
    a.setp_lt(p, kt, kt_total)
    a.brnz(p, "loop")

    # epilogue: thread (warp w, lane) owns rows 16w + g (+8), columns 8j + 2t (+1)
    a.shr(g, lane, 2)
    a.and_(tt, lane, 3)
    a.and_(t, wid, 3)
    a.imad(t, t, 16, g)                                   # row
    a.imad(t, t, n, rown)                                 # row * N + cta * bn
    a.imad(t, tt, 2, t)                                   # + 2t
    a.shl(t, t, 2)
    a.ldc(p, "out")
    a.iadd(t, t, p)
    for j in range(bn // 8):
        for r in range(4):
            a.stg(t, acc + 4 * j + r, (8 * (r >> 1) * n + 8 * j + (r & 1)) * 4)
    return a.build(), mb0 + stages * 8
