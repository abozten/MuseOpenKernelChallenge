"""Checks the model against numbers measured on real Hopper/Ampere parts (see README sources).
python3 -m h200sim.test_calibration"""

import numpy as np

from . import Asm, H200, Sim


def _issue_cycles(sim, op="LDG"):
    return [e[0] for e in sim.issue_log if e[6] == op]


def test_memory_queue_matches_table1():
    """Ampere core study, Table 1: a sub-core issues 5 memory instructions back to back, the 6th
    waits for the queue (13), then one per 4 cycles; four active sub-cores share a 2-cycle port."""
    for active, expect_gaps in ((1, [0, 1, 2, 3, 4, 11, 15, 19]), (4, [0, 1, 2, 3, 4, 11, 13, 15, 17])):
        sim = Sim(H200(model_icache=False, model_ccache=False))   # cold fetch would stagger the sub-cores
        sim.log_issue = True
        mem = sim.mem.alloc(8192)
        a = Asm("memq")
        ptr = a.reg()
        dst = [a.reg() for _ in range(10)]
        a.ldc(ptr, "p")
        for i in range(10):
            a.ldg(dst[i], ptr, i * 128, 4)
        sim.launch(a.build(), 1, 32 * active, {"p": mem})
        for sp in range(active):
            c = [e[0] for e in sim.issue_log if e[6] == "LDG" and e[2] == sp]
            base = c[0]
            if active == 1:
                assert [x - base for x in c[:8]] == expect_gaps, (active, c)
        if active == 4:
            sixth = sorted(e[0] for e in sim.issue_log if e[6] == "LDG" and e[5] == 6)   # pc after the 6th LDG
            d = np.diff(sixth)
            assert (d == 2).all(), sixth         # sub-cores interleave on the shared 2-cycle port


def _chain(bypass_l1, n=8):
    """Dependent loads v = [v] (the word holds its own address)."""
    a = Asm("chase")
    ptr = a.reg()
    a.ldc(ptr, "p")
    for _ in range(n + 1):                    # the first is a warm-up and loads the pointer
        a.ldg(ptr, ptr, 0, 4, cache="cs" if bypass_l1 else "ca")
    return a.build()


def _hop(sim):
    c = _issue_cycles(sim)
    return float(np.mean(np.diff(c)[1:]))      # drop the hop out of the warm-up load


def test_l1_hit_latency():
    sim = Sim(H200())
    sim.log_issue = True
    p = sim.mem.alloc(256)
    sim.mem.write(p, np.array([p], np.uint32))
    sim.launch(_chain(False), 1, 32, {"p": p})
    assert 31 <= _hop(sim) <= 36, _hop(sim)          # measured 32-33 (H800)


def test_l2_near_far_and_dram_latency():
    results = {}
    for part_name, want_part in (("near", 0), ("far", 1)):
        for level in ("hit", "miss"):
            sim = Sim(H200())
            sim.log_issue = True
            base = sim.mem.alloc(1 << 16)
            ok = (lambda o: sim.memsys._sub((base + o) >> 5) == 0) if want_part == 0 else \
                 (lambda o: sim.memsys._partition((base + o) >> 5) == 1)     # near = SM 0's own sub-NUMA node
            hops = [o for o in range(0, 1 << 16, 128) if ok(o)][:12]
            # sector hops[i] holds the address of hops[i+1]; the last one points at itself
            for h, nxt in zip(hops, hops[1:] + hops[-1:]):
                sim.mem.write(base + h, np.array([base + nxt], np.uint32))
            p = base + hops[0]
            prog = _chain(True, 8)
            if level == "hit":
                sim.launch(prog, 1, 32, {"p": p})           # first run fills L2 (SM 0, partition 0 is near)
                sim.issue_log.clear()
            sim.launch(prog, 1, 32, {"p": p})
            results[(part_name, level)] = _hop(sim)
    # measured on H100 SXM5 (Liu et al.): L2 295 local / 470 remote, DRAM 554 local / 728 remote
    assert 288 <= results[("near", "hit")] <= 310, results
    assert 465 <= results[("far", "hit")] <= 490, results
    assert 548 <= results[("near", "miss")] <= 575, results
    assert 720 <= results[("far", "miss")] <= 755, results


def test_l2_prefetch_qualifier_pulls_neighbours():
    """ld.global.nc.L1::no_allocate.L2::256B: a miss also fetches the other sectors of its 256 B window,
    so the neighbour is an L2 hit (~258) instead of a DRAM miss (~555) when it is read next."""
    got = {}
    for pf in (0, 256):
        sim = Sim(H200())
        sim.log_issue = True
        base = sim.mem.alloc(1 << 14)
        hop = [o for o in range(0, 1 << 14, 256) if sim.memsys._sub((base + o) >> 5) == 0
               and sim.memsys._sub((base + o + 96) >> 5) == 0][0]
        p0, p1 = base + hop, base + hop + 96                    # same aligned 256 B window, different sectors
        sim.mem.write(p0, np.array([p1], np.uint32))
        sim.mem.write(p1, np.array([p1], np.uint32))
        a = Asm("pf")
        ptr = a.reg(); a.ldc(ptr, "p")
        a.ldg(ptr, ptr, 0, 4, cache="cs", l2pf=pf)
        a.ldg(ptr, ptr, 0, 4, cache="cs")
        sim.launch(a.build(), 1, 32, {"p": p0})
        c = _issue_cycles(sim)
        got[pf] = c[1] - c[0]                                   # the first load's full latency
        got[(pf, "second")] = sim.memsys.l2_prefetch
        got[(pf, "hits")] = sim.memsys.l2_hit
        got[(pf, "bytes")] = sim.memsys.hbm_bytes
    assert got[0] > 500 and got[(0, "second")] == 0
    assert got[(256, "second")] == 6                            # 8 sectors in 256 B: demand miss + its 64 B burst mate + 6 more
    assert got[(0, "bytes")] == 128 and got[(256, "bytes")] == 256   # two 64 B bursts by default; one 256 B window with the qualifier
    assert got[(0, "hits")] == 0 and got[(256, "hits")] == 1    # the second load now finds its sector in L2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
