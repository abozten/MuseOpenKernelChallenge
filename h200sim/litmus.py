"""Litmus tests for the memory model.

Each test runs two threads in two CTAs (on different SMs) with random start delays, over addresses placed so
that one location is in the writer's near L2 NUMA node and the other in the far one (so stores can land out of
program order). `outcomes()` collects the set of observed result tuples; `ALLOWED` lists what the PTX memory
model permits for each variant (Lustig, Sahasrabuddhe, Giroux, "A Formal Analysis of the NVIDIA PTX Memory
Consistency Model", ASPLOS 2019). The simulator is an operational model, so the check is:
  * soundness  -- it never produces an outcome the PTX model forbids;
  * liveness   -- it does exhibit the weak outcomes of the unsynchronised variants.
Loads read memory where their request lands (about half way through their latency), which is what lets load
buffering and store buffering show up; fence.acq_rel only orders (it does not stall), fence.sc waits.
"""

import random

import numpy as np

from .config import H200
from .core import Sim
from .isa import Asm

W, Wc = (0, 0), None


def _addresses(sim, both_near=False):
    base = sim.mem.alloc(1 << 15)
    far = next(o for o in range(0, 1 << 15, 128) if sim.memsys._partition((base + o) >> 5) == 1)
    nears = [o for o in range(0, 1 << 15, 128) if sim.memsys._sub((base + o) >> 5) == 0]
    if both_near:
        return base + nears[1], base + nears[0]
    return base + far, base + nears[0]


def _fence(a, kind):
    if kind == "sc":
        a.membar("gpu", "sc")
    elif kind == "acq_rel":
        a.membar("gpu", "acq_rel")


def build(test, variant):
    """Kernel with both threads (CTA 0 and CTA 1). Parameters: x, y, out, d0, d1."""
    a = Asm(f"{test}.{variant}")
    cta, px, py, po, p, v, r0, r1, t = (a.reg() for _ in range(9))
    d = a.reg()
    a.s2r(cta, "ctaid"); a.ldc(px, "x"); a.ldc(py, "y"); a.ldc(po, "out"); a.ldc(d, "d0")
    a.setp_eq(p, cta, 0)
    a.brnz(p, "t0")
    a.ldc(d, "d1")
    a.label("t1")
    a.nanosleep(d)
    if test == "MP":                       # T1: r_flag = y; r_data = x
        ld_flag = dict(sem="acquire") if variant == "relacq" else {}
        a.ldg(r0, py, 0, 4, cache="cv", **ld_flag)
        _fence(a, variant if variant in ("sc", "acq_rel") else None)
        a.ldg(r1, px, 0, 4, cache="cv")
    elif test == "SB":                     # T1: y = 1; fence; r1 = x
        a.mov(v, 1); a.stg(py, v)
        _fence(a, variant)
        a.ldg(r1, px, 0, 4, cache="cv"); a.mov(r0, 0)
    elif test == "CORR":                   # T1: r0 = x; r1 = x
        a.ldg(r0, px, 0, 4, cache="cv"); a.ldg(r1, px, 0, 4, cache="cv")
    elif test == "LB":                     # T1: r1 = x; y = 1
        a.ldg(r1, px, 0, 4, cache="cv"); a.mov(v, 1); a.stg(py, v); a.mov(r0, 0)
    a.imad(t, cta, 8, po); a.stg(t, r0, 0); a.stg(t, r1, 4)
    a.exit()
    a.label("t0")
    a.nanosleep(d)
    if test == "MP":                       # T0: x = 1; fence; y = 1
        a.mov(v, 1); a.stg(px, v)
        _fence(a, variant if variant in ("sc", "acq_rel") else None)
        a.stg(py, v, sem="release" if variant == "relacq" else "weak")
        a.mov(r0, 0); a.mov(r1, 0)
    elif test == "SB":                     # T0: x = 1; fence; r0 = y
        a.mov(v, 1); a.stg(px, v)
        _fence(a, variant)
        a.ldg(r0, py, 0, 4, cache="cv"); a.mov(r1, 0)
    elif test == "CORR":                   # T0: x = 1
        a.mov(v, 1); a.stg(px, v); a.mov(r0, 0); a.mov(r1, 0)
    elif test == "LB":                     # T0: r0 = y; x = 1
        a.ldg(r0, py, 0, 4, cache="cv"); a.mov(v, 1); a.stg(px, v); a.mov(r1, 0)
    a.imad(t, cta, 8, po); a.stg(t, r0, 0); a.stg(t, r1, 4)
    return a.build()


def result(test, out):
    """The litmus outcome tuple from the two threads' recorded registers."""
    r = out.reshape(2, 2)                  # [cta][slot]
    if test == "MP":
        return int(r[1][0]), int(r[1][1])  # (flag, data) seen by T1
    if test == "SB":
        return int(r[0][0]), int(r[1][1])  # (T0's read of y, T1's read of x)
    if test == "CORR":
        return int(r[1][0]), int(r[1][1])  # (first, second) read of x by T1
    if test == "LB":
        return int(r[0][0]), int(r[1][1])  # (T0's read of y, T1's read of x)


def run_once(test, variant, d0, d1, swap=False):
    sim = Sim(H200(model_icache=False, model_ccache=False))
    far, near = _addresses(sim, both_near=(test == "LB"))
    x, y = (near, far) if swap else (far, near)
    out = sim.mem.alloc(16)
    sim.mem.write(out, np.zeros(4, np.uint32))
    warm = Asm("warm")                      # touch both locations so translations and lines are resident
    px, py, v = warm.reg(), warm.reg(), warm.reg()
    warm.ldc(px, "x"); warm.ldc(py, "y")
    warm.ldg(v, px, 0, 4, cache="cv"); warm.ldg(v, py, 0, 4, cache="cv")
    sim.launch(warm.build(), 2, 32, {"x": x, "y": y})
    sim.launch(build(test, variant), 2, 32, {"x": x, "y": y, "out": out, "d0": d0, "d1": d1})
    return result(test, sim.mem.read(out, np.uint32, 4))


def outcomes(test, variant, trials=24, seed=0):
    rng = random.Random(seed)
    seen = {}
    for k in range(trials):
        d0 = rng.randrange(0, 300)                       # T0's delay; T1's is T0's plus a relative offset that sweeps
        d1 = max(0, d0 + rng.randrange(-150, 500))       # across the windows where reordering is possible
        o = run_once(test, variant, d0, d1, swap=bool(k & 1))
        seen[o] = seen.get(o, 0) + 1
    return seen


ALL4 = {(0, 0), (0, 1), (1, 0), (1, 1)}
# what the PTX memory model allows
ALLOWED = {
    ("MP", "weak"): ALL4,
    ("MP", "relacq"): ALL4 - {(1, 0)},          # flag seen => data seen
    ("MP", "sc"): ALL4 - {(1, 0)},
    ("MP", "acq_rel"): ALL4 - {(1, 0)},
    ("SB", "weak"): ALL4,
    ("SB", "sc"): ALL4 - {(0, 0)},              # fence.sc restores sequential consistency for SB
    ("SB", "acq_rel"): ALL4,                    # PTX allows (0,0): acq_rel orders, it does not forbid store buffering
    ("CORR", "weak"): ALL4 - {(1, 0)},          # per-location coherence
    ("LB", "weak"): ALL4,                       # PTX allows (1,1); this implementation never produces it
}
# weak outcomes the variants must be able to show
# (load buffering's (1,1) is allowed by PTX but never produced here: a store becomes visible later than a load
# reads, so a load cannot observe a program-later store of another thread. That is stronger than the model.)
WITNESS = {("MP", "weak"): (1, 0), ("SB", "weak"): (0, 0), ("SB", "acq_rel"): (0, 0)}
