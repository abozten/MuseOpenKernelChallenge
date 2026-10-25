"""Compiler control bits vs the hardware scoreboard. python3 -m h200sim.test_control"""

import numpy as np

from . import Asm, H200, Sim, from_bf16_bits, to_bf16_bits
from .kernels import gemv_bf16


def _issue(mode, build, params=None, setup=None, cfg=None):
    sim = Sim(cfg or H200(model_icache=False, model_ccache=False), dep_mode=mode)
    sim.log_issue = True
    a = Asm("t")
    ctx = setup(sim) if setup else None
    build(a, ctx)
    sim.launch(a.build(), 1, 32, params(ctx) if params else {})
    return sim


def test_dependent_and_independent_chains_match_the_scoreboard():
    def chain(a, _):
        x = [a.reg() for _ in range(8)]
        for i in range(6):
            a.ffma(x[0], x[0], x[1], x[2])             # each depends on the previous: 4 cycles apart
        for i in range(6):
            a.fadd(x[3 + i % 4], x[1], x[2])           # independent, sources in different banks: back to back
    for mode in ("scoreboard", "control"):
        sim = _issue(mode, chain)
        c = [e[0] for e in sim.issue_log if e[6] in ("FFMA", "FADD")]
        d = np.diff(c)
        assert list(d[:5]) == [4] * 5, (mode, d)
        assert list(d[6:]) == [1] * 5, (mode, d)


def test_annotation_follows_fixed_latency_and_counters():
    from .control import annotate
    a = Asm("ann")
    x, y, p = a.reg(), a.reg(), a.reg()
    a.ldc(p, "p")                    # variable latency -> counter
    a.ldg(x, p, 0, 4)                # variable latency -> counter, waits for the pointer
    a.fadd(y, x, x)                  # consumer: waits for the load's counter
    a.fadd(y, y, y)                  # fixed-latency dependent: stall of the previous op covers 4 cycles
    prog = a.build()
    annotate(prog)
    ldc, ldg, add1, add2 = prog.instrs[:4]
    assert ldc.inc_raw is not None and ldg.inc_raw is not None and ldc.inc_raw != ldg.inc_raw
    assert ldc.inc_raw in ldg.wait and ldg.inc_raw in add1.wait
    assert add1.stall == 4           # the next instruction needs add1's result


def test_shared_dependence_counters_create_false_waits():
    """Six counters cannot name every in-flight load: a consumer of a fast load also waits for a slow one that
    shares its counter. The hardware scoreboard only waits for the load it needs."""
    def build(a, ctx):
        fast, slow_base = ctx
        p = a.reg(); a.ldc(p, "p")
        regs = [a.reg() for _ in range(12)]
        a.ldg(regs[0], p, 0, 4, cache="ca")                 # warmed in L1: returns in ~35 cycles
        a.ldc(p, "q")
        for i in range(1, 11):
            a.ldg(regs[i], p, 4096 * i, 4, cache="cs")      # ten cold DRAM loads (> 5 counters, so they share)
        a.iadd(regs[11], regs[0], 1)                        # consumer of the FAST load

    def setup(sim):
        fast = sim.mem.alloc(64)
        slow = sim.mem.alloc(4096 * 12)
        return fast, slow
    gaps = {}
    for mode in ("scoreboard", "control"):
        sim = Sim(H200(model_icache=False, model_ccache=False), dep_mode=mode)
        sim.log_issue = True
        fast, slow = setup(sim)
        a = Asm("share")
        build(a, (fast, slow))
        # warm the L1 line of `fast` with a first launch, then time the second
        sim.launch(a.build(), 1, 32, {"p": fast, "q": slow})
        sim.issue_log.clear()
        sim.launch(a.build(), 1, 32, {"p": fast, "q": slow})
        iss = {e[5]: e[0] for e in sim.issue_log}
        ldgs = [e[0] for e in sim.issue_log if e[6] == "LDG"]
        gaps[mode] = [e[0] for e in sim.issue_log if e[6] == "IADD"][0] - ldgs[0]
    assert gaps["control"] > gaps["scoreboard"] + 300, gaps


def test_gemv_is_correct_and_close_in_both_modes():
    n, k = 64, 4096
    rng = np.random.default_rng(1)
    W = from_bf16_bits(to_bf16_bits(rng.standard_normal((n, k), dtype=np.float32) / 64))
    x = from_bf16_bits(to_bf16_bits(rng.standard_normal(k, dtype=np.float32)))
    res = {}
    for mode in ("scoreboard", "control"):
        sim = Sim(H200(), dep_mode=mode)
        pW, px, po = sim.mem.alloc(W.size * 2), sim.mem.alloc(x.size * 2), sim.mem.alloc(n * 4)
        sim.mem.write(pW, to_bf16_bits(W)); sim.mem.write(px, to_bf16_bits(x))
        st = sim.launch(gemv_bf16(k, 2, 4), n // 4, 128, {"W": pW, "x": px, "out": po})
        out = sim.mem.read(po, np.float32, n)
        ref = W.astype(np.float64) @ x.astype(np.float64)
        assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-5
        res[mode] = st.cycles
    ratio = res["control"] / res["scoreboard"]
    assert 0.85 < ratio < 1.35, res          # same machine, two ways of tracking dependences


def test_yield_bit_switches_warps_or_leaves_a_bubble():
    def prog():
        a = Asm("y")
        r = [a.reg() for _ in range(8)]
        for i in range(12):
            a.fadd(r[2 + i % 4], r[0], r[1])
            a.yield_hint()
        return a.build()

    def gaps(block):
        sim = Sim(H200(model_icache=False, model_ccache=False), dep_mode="control")
        sim.log_issue = True
        sim.launch(prog(), 1, block, {})
        # sub-core 0 hosts warp 0 (and warp 4 when the block has 8 warps)
        c = sorted(e[0] for e in sim.issue_log if e[2] == 0 and e[6] == "FADD")
        return np.diff(c)
    one = gaps(32)                      # a lone warp that yields every instruction: a bubble after each
    two = gaps(32 * 5)                  # warps 0 and 4 share sub-core 0 and cover each other's bubbles
    assert set(one) == {2}, one
    assert set(two) == {1}, two


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
