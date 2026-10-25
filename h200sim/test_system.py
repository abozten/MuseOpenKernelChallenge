"""Launch chains (PDL), TLBs and memory ordering. python3 -m h200sim.test_system"""

import numpy as np

from . import Asm, H200, Sim
from .core import Launch


def _producer(early_trigger):
    a = Asm("A")
    ctaid, lane, t, v, optr = (a.reg() for _ in range(5))
    a.s2r(ctaid, "ctaid"); a.s2r(lane, "laneid"); a.ldc(optr, "out")
    if early_trigger:
        a.griddep_launch()                      # the dependents may be scheduled now
    a.nanosleep(3000)                           # long body
    a.iadd(v, ctaid, 1000)
    a.imad(t, ctaid, 32, lane); a.imad(t, t, 4, optr)
    a.stg(t, v)
    return a.build()


def _consumer():
    a = Asm("B")
    ctaid, lane, t, v, src, dst = (a.reg() for _ in range(6))
    a.s2r(ctaid, "ctaid"); a.s2r(lane, "laneid"); a.ldc(src, "in"); a.ldc(dst, "out")
    a.nanosleep(1500)                           # preamble that does not need A's results
    a.griddep_wait()
    a.imad(t, ctaid, 32, lane); a.imad(t, t, 4, src)
    a.ldg(v, t, 0, 4)
    a.imad(t, ctaid, 32, lane); a.imad(t, t, 4, dst)
    a.stg(t, v)
    return a.build()


def _run_chain(pdl):
    sim = Sim(H200())
    n = 132
    pa = sim.mem.alloc(n * 128)
    pb = sim.mem.alloc(n * 128)
    st = sim.launch_chain([Launch(_producer(pdl), n, 32, {"out": pa}),
                           Launch(_consumer(), n, 32, {"in": pa, "out": pb}, pdl=pdl)])
    got = sim.mem.read(pb, np.uint32, n * 32).reshape(n, 32)
    return st, got


def test_pdl_overlaps_the_secondary_and_keeps_results_correct():
    base, got0 = _run_chain(False)
    pdl, got1 = _run_chain(True)
    want = (np.arange(132) + 1000)[:, None] * np.ones((1, 32), np.uint32)
    np.testing.assert_array_equal(got0, want)
    np.testing.assert_array_equal(got1, want)              # griddep_wait made A's stores visible
    a, b = pdl.kernels
    assert b.begin_cycle < a.end_cycle                      # B was resident while A was still running
    ba, bb = base.kernels
    assert bb.begin_cycle >= ba.end_cycle + 2000            # without PDL: A ends, then a ~1.3 us graph launch gap
    assert pdl.cycles < base.cycles - 3000                  # the gap and B's preamble are hidden


def test_streams_run_kernels_concurrently():
    def sleeper(cycles, tag):
        a = Asm(tag)
        a.nanosleep(cycles)
        t = a.reg(); a.s2r(t, "clock")
        return a.build()

    def total(streams):
        sim = Sim(H200())
        st = sim.launch_chain([Launch(sleeper(4000, "A"), 66, 32, {}, stream=streams[0]),
                               Launch(sleeper(4000, "B"), 66, 32, {}, stream=streams[1])])
        return st
    same, diff = total((0, 0)), total((0, 1))
    assert same.kernels[1].begin_cycle >= same.kernels[0].end_cycle        # one stream: B waits for A (+ launch gap)
    assert diff.kernels[1].begin_cycle < diff.kernels[0].end_cycle         # two streams: B starts immediately
    assert diff.cycles < same.cycles - 3000                                # and the SMs are shared, not queued


def test_griddep_wait_without_a_predecessor_is_a_no_op():
    sim = Sim(H200())
    out = sim.mem.alloc(128)
    a = Asm("solo")
    a.griddep_wait()
    t = a.reg(); a.s2r(t, "laneid"); a.shl(t, t, 2); a.ldc(a.reg(), "out")
    st = sim.launch(a.build(), 1, 32, {"out": out})
    assert st.cycles > 0


def _two_cta_sim(cfg=None):
    return Sim(cfg or H200())


def _far_near_addrs(sim, want_far_for_sm0=True):
    """A data sector in the partition far from SM 0 and a flag sector in its near partition."""
    base = sim.mem.alloc(1 << 15)
    far = next(o for o in range(0, 1 << 15, 128) if sim.memsys._partition((base + o) >> 5) == 1)
    near = next(o for o in range(0, 1 << 15, 128) if sim.memsys._sub((base + o) >> 5) == 0)
    return base + far, base + near


def _message_passing(fence, samples=48, release=False):
    """Count samples where the consumer saw flag == 1 but the data still held its old value."""
    sim = _two_cta_sim()
    data, flag = _far_near_addrs(sim)
    out = sim.mem.alloc(8)
    a = Asm("mp")
    cta, v, p, pd, pf, o, anomalies, dz = (a.reg() for _ in range(8))
    fs, ds = [a.reg() for _ in range(samples)], [a.reg() for _ in range(samples)]
    a.s2r(cta, "ctaid"); a.ldc(pd, "data"); a.ldc(pf, "flag"); a.ldc(o, "out")
    a.setp_eq(p, cta, 0)
    a.brnz(p, "producer")
    for k in range(samples):                         # consumer: independent flag / data sample pairs over time
        a.ldg(fs[k], pf, 0, 4, cache="cv")
        a.ldg(ds[k], pd, 0, 4, cache="cv")
        a.nanosleep(20)
    a.mov(anomalies, 0)
    for k in range(samples):
        a.setp_eq(dz, ds[k], 0)
        a.and_(dz, dz, fs[k])                        # flag seen (1) and data still 0
        a.iadd(anomalies, anomalies, dz)
    a.stg(o, anomalies)
    a.exit()
    a.label("producer")
    a.nanosleep(300)
    a.mov(v, 42); a.stg(pd, v)
    if fence:
        a.membar("gpu")                              # data must land before the flag is written
    a.mov(v, 1); a.stg(pf, v, sem="release" if release else "weak")
    sim.launch(a.build(), 2, 32, {"data": data, "flag": flag, "out": out})
    return int(sim.mem.read(out, np.uint32, 1)[0])


def test_message_passing_needs_a_fence():
    """The data store goes to the far L2 partition and lands later than the flag store, so a consumer can
    see the flag with stale data. A membar.gpu between the producer's stores forbids it."""
    assert _message_passing(fence=False) > 0          # flag observed while the data was still old
    assert _message_passing(fence=True) == 0


def test_release_store_replaces_the_fence():
    assert _message_passing(fence=False, release=False) > 0
    assert _message_passing(fence=False, release=True) == 0       # st.release waits for the data store to land


def test_acquire_load_invalidates_l1():
    """The consumer caches the data (old value) in L1 first. Reading the flag with ld.acquire invalidates the
    L1 and orders the later load, so it sees the new data; a relaxed flag read leaves the stale L1 copy."""
    def run(sem):
        sim = _two_cta_sim()
        data, flag = _far_near_addrs(sim)
        out = sim.mem.alloc(8)
        a = Asm("acq")
        cta, v, p, pd, pf, o, f, d0, d1 = (a.reg() for _ in range(9))
        a.s2r(cta, "ctaid"); a.ldc(pd, "data"); a.ldc(pf, "flag"); a.ldc(o, "out")
        a.setp_eq(p, cta, 0)
        a.brnz(p, "producer")
        a.ldg(d0, pd, 0, 4, cache="ca")                # caches 0 in L1
        a.label("poll")
        a.ldg(f, pf, 0, 4, cache="cv", sem=sem)
        a.setp_eq(p, f, 0)
        a.brnz(p, "poll")
        a.ldg(d1, pd, 0, 4, cache="ca")                # after the flag
        a.stg(o, d1)
        a.exit()
        a.label("producer")
        a.nanosleep(4000)
        a.mov(v, 42); a.stg(pd, v)
        a.mov(v, 1); a.stg(pf, v, sem="release")
        sim.launch(a.build(), 2, 32, {"data": data, "flag": flag, "out": out})
        return int(sim.mem.read(out, np.uint32, 1)[0])
    assert run("acquire") == 42
    assert run("relaxed") == 0


def test_l1_is_not_coherent():
    for staleness, want in ((True, 0), (False, 7)):
        sim = _two_cta_sim(H200(model_l1_staleness=staleness))
        x = sim.mem.alloc(64)
        out = sim.mem.alloc(16)
        a = Asm("l1")
        cta, v, cached, fresh, p, px, o = (a.reg() for _ in range(7))
        a.s2r(cta, "ctaid"); a.ldc(px, "x"); a.ldc(o, "out")
        a.setp_eq(p, cta, 0)
        a.brnz(p, "writer")
        a.ldg(cached, px, 0, 4, cache="ca")           # fills this SM's L1 with the old value (its cold read lands ~cycle 2000)
        a.nanosleep(5000)
        a.ldg(cached, px, 0, 4, cache="ca")           # still in L1: the other SM's store is invisible here
        a.ldg(fresh, px, 0, 4, cache="cv")            # volatile / .cv goes to L2
        a.stg(o, cached); a.stg(o, fresh, 4)
        a.exit()
        a.label("writer")
        a.nanosleep(3000)                              # write after the consumer's first read
        a.mov(v, 7); a.stg(px, v)
        sim.launch(a.build(), 2, 32, {"x": x, "out": out})
        got = sim.mem.read(out, np.uint32, 2)
        assert got[0] == want and got[1] == 7, (staleness, got)


def test_membar_waits_for_stores_to_land():
    sim = Sim(H200(model_icache=False, model_ccache=False))
    sim.log_issue = True
    base = sim.mem.alloc(256)
    a = Asm("mb")
    p, v, t = a.reg(), a.reg(), a.reg()
    a.ldc(p, "p"); a.mov(v, 5)
    a.stg(p, v)
    a.membar("gpu")
    a.s2r(t, "clock")
    sim.launch(a.build(), 1, 32, {"p": base})
    c = {e[6]: e[0] for e in sim.issue_log}
    assert c["MEMBAR"] - c["STG"] >= sim.cfg.lat_l2 // 2          # held back until the store reached L2
    sim2 = Sim(H200(model_icache=False, model_ccache=False)); sim2.log_issue = True
    b = Asm("mb0")
    p, t = b.reg(), b.reg()
    b.ldc(p, "p"); b.membar("cta"); b.s2r(t, "clock")
    sim2.launch(b.build(), 1, 32, {"p": sim2.mem.alloc(256)})
    c2 = {e[6]: e[0] for e in sim2.issue_log}
    assert c2["S2R"] - c2["MEMBAR"] <= 2                          # CTA scope costs nothing


def test_async_proxy_needs_a_fence():
    for fence, strict in ((False, False), (True, False)):
        sim = Sim(H200())
        dst = sim.mem.alloc(256)
        a = Asm("proxy")
        lane, g, v = a.reg(), a.reg(), a.reg()
        a.s2r(lane, "laneid"); a.shl(lane, lane, 2); a.mov(v, 9); a.sts(lane, v)
        a.bar()
        if fence:
            a.fence()                               # fence.proxy.async
        a.ldc(g, "dst"); a.mov(v, 0)
        a.tma_store_1d(g, v, 128)
        a.cp_commit(); a.cp_wait(0)
        sim.launch(a.build(), 1, 32, {"dst": dst}, smem_bytes=256)
        assert bool(sim.hazards) == (not fence), sim.hazards
        assert (sim.mem.read(dst, np.uint32, 32) == 9).all()
    sim = Sim(H200(strict_proxy_fence=True))
    a = Asm("strict")
    lane, g, v = a.reg(), a.reg(), a.reg()
    a.s2r(lane, "laneid"); a.shl(lane, lane, 2); a.sts(lane, v); a.ldc(g, "dst"); a.mov(v, 0)
    a.tma_store_1d(g, v, 128)
    try:
        sim.launch(a.build(), 1, 32, {"dst": sim.mem.alloc(256)}, smem_bytes=256)
        assert False
    except RuntimeError as e:
        assert "fence.proxy.async" in str(e)


def test_nvlink_peer_memory_latency_and_bandwidth():
    # latency: a dependent chain through peer memory pays the NVLink round trip every hop
    sim = Sim(H200()); sim.log_issue = True
    peer = sim.map_peer(1 << 12)
    for i in range(4):
        sim.mem.write(peer + 256 * i, np.array([peer + 256 * ((i + 1) % 4)], np.uint32))
    a = Asm("nvl_chain"); ptr = a.reg(); a.ldc(ptr, "p")
    for _ in range(9):
        a.ldg(ptr, ptr, 0, 4, cache="cs")
    sim.launch(a.build(), 1, 32, {"p": peer})
    c = [e[0] for e in sim.issue_log if e[6] == "LDG"]
    hop = float(np.mean(np.diff(c)[1:]))
    assert abs(hop - sim.cfg.nvlink_latency_ns * sim.cfg.clock_ghz) < 40, hop

    # bandwidth: many warps streaming from the peer saturate ~80 % of 450 GB/s
    sim = Sim(H200())
    n_ctas, warps, loads = 66, 8, 16
    size = n_ctas * warps * loads * 512
    peer = sim.map_peer(size)
    a = Asm("nvl_bw")
    cta, wid, lane, base, t = (a.reg() for _ in range(5))
    dst = [a.reg(4) for _ in range(loads)]
    a.s2r(cta, "ctaid"); a.s2r(wid, "warpid"); a.s2r(lane, "laneid"); a.ldc(base, "p")
    a.imad(t, cta, warps, wid)                       # global warp id
    a.imad(t, t, loads * 512, base)
    a.imad(t, lane, 16, t)
    for i in range(loads):
        a.ldg(dst[i], t, i * 512, 16, cache="cs")
    st = sim.launch(a.build(), n_ctas, warps * 32, {"p": peer})
    gbs = size / (st.cycles / (sim.cfg.clock_ghz * 1e9)) / 1e9
    want = sim.cfg.nvlink_gbs_per_dir * sim.cfg.nvlink_efficiency
    assert 0.6 * want < gbs <= 1.02 * want, (gbs, want)       # latency and ramp-up keep it a little below the link rate
    assert sim.memsys.nvlink_bytes == size and sim.memsys.hbm_bytes == 0


def test_ecc_single_bit_corrected_double_bit_aborts_disabled_corrupts():
    from .memory import EccError

    def chase(cfg, bits):
        sim = Sim(cfg); sim.log_issue = True
        base = sim.mem.alloc(1 << 14)
        hops = [base + 4096 * i for i in range(4)]
        for h, n in zip(hops, hops[1:] + hops[:1]):
            sim.mem.write(h, np.array([n], np.uint32))
        if bits:
            sim.inject_ecc_error(hops[2], bits)
        a = Asm("ecc"); ptr = a.reg(); a.ldc(ptr, "p")
        for _ in range(5):
            a.ldg(ptr, ptr, 0, 4, cache="cs")
        sim.launch(a.build(), 1, 32, {"p": hops[0]})
        c = [e[0] for e in sim.issue_log if e[6] == "LDG"]
        return sim, np.diff(c)
    clean_sim, clean = chase(H200(), 0)
    sim, hit = chase(H200(), 1)
    assert sim.memsys.ecc_corrected == 1
    assert (hit - clean).max() >= H200().ecc_correct_cycles - 5 and (hit - clean).min() >= -5   # one slow hop
    try:
        chase(H200(), 2)
        assert False, "double-bit error must abort"
    except EccError as e:
        assert "XID 48" in str(e)
    s2 = Sim(H200(ecc_enabled=False))
    x = s2.mem.alloc(64); s2.mem.write(x, np.zeros(16, np.uint32))
    s2.inject_ecc_error(x, 3)
    assert s2.mem.read(x, np.uint32, 1)[0] == 0b111                 # no ECC: silent corruption


def _power(cap, toggle=1.0, tc_util=0.0, issue_util=0.0, hbm_util=0.0, l2_util=0.0, cyc=100_000):
    import types
    sim = Sim(H200(power_cap_w=cap, tensor_data_toggle=toggle))
    c = sim.cfg
    slots = cyc * c.num_sms * c.smsp_per_sm
    sim.memsys.hbm_bytes = int(hbm_util * cyc * c.hbm_bytes_per_cycle)
    sim.memsys.l2_hit = int(l2_util * cyc * c.l2_bytes_per_cycle / 32)
    k = types.SimpleNamespace(tc_busy=tc_util * slots, issued=issue_util * slots)
    return sim._power(k, cyc, slots)


def test_power_model_matches_the_h200_power_capping_study():
    """arXiv 2604.11391 (H200): a streaming kernel needs a >= 550 W cap to keep full bandwidth, and under a
    200 W cap the SM clock falls to its 345 MHz floor while the board still draws ~250 W."""
    stream = dict(hbm_util=1.0, l2_util=0.557)                     # 4.4 TB/s of 7.9 TB/s L2
    p700 = _power(700, **stream)
    assert 545 <= p700["power_w"] <= 560 and p700["clock_scale"] == 1.0
    assert _power(550, **stream)["clock_scale"] > 0.99             # the knee is at ~550 W
    assert _power(450, **stream)["clock_scale"] < 0.95             # below it bandwidth work slows
    low = _power(200, **stream)
    assert abs(low["clock_scale"] - 345 / 1980) < 1e-6             # the 345 MHz floor
    assert 240 <= low["power_w"] <= 262                            # "about 250 W" at a 200 W cap
    assert low["time_us_dvfs"] > 5 * p700["time_us_dvfs"]


def test_tensor_power_depends_on_the_data_and_a_power_virus_throttles():
    gemm = dict(tc_util=0.9, issue_util=0.3, hbm_util=0.3, l2_util=0.3)
    rnd, zero = _power(700, 1.0, **gemm), _power(700, 0.0, **gemm)
    assert rnd["power_w"] > zero["power_w"] + 100                  # random operands burn far more than zeros
    assert 0.9 < rnd["clock_scale"] < 1.0 and rnd["power_w"] <= 700.5      # a saturated random-data GEMM sits at the limit
    assert zero["clock_scale"] == 1.0 and zero["power_w"] < 600
    capped = _power(450, 1.0, **gemm)
    assert 0.5 < capped["clock_scale"] < 0.9 and capped["time_us_dvfs"] > rnd["time_us_dvfs"] * 1.1
    hot = _power(700, 1.0, tc_util=1.0, issue_util=1.0, hbm_util=1.0, l2_util=1.0)
    assert hot["clock_scale"] < 1.0 and hot["power_w"] <= 700.5


def test_ecc_and_power_stats_are_reported():
    sim = Sim(H200())
    a = Asm("k"); t = a.reg(); a.s2r(t, "clock")
    st = sim.launch(a.build(), 1, 32, {})
    assert st.power_w >= sim.cfg.power_static_w and st.clock_scale == 1.0 and st.ecc_corrected == 0


def _send_recv(hosts, nbytes=64 * 1024):
    """GPU 0 writes `nbytes` into GPU 1's memory then sets a flag (st.release); GPU 1 spins on the flag, then
    sums the data it finds in its own memory. Returns (checksum, cycle GPU 1 finished, node)."""
    from .node import Node
    node = Node(2, hosts=hosts)
    data = node.alloc(1, nbytes)
    flag = node.alloc(1, 128)
    count = node.alloc(1, 128)
    out = node.alloc(1, 128)
    src = (np.arange(nbytes // 4, dtype=np.uint32) % 251)
    node.write(data, np.zeros(nbytes // 4, np.uint32))
    # GPU 0: one warp per 512 B chunk of the payload -- but its source values are computed, not loaded
    nctas = nbytes // 128
    a = Asm("send")
    cta, lane, v, addr, pd, pf, pc, t, old = (a.reg() for _ in range(9))
    p0, last = a.preg(), a.preg()
    a.s2r(cta, "ctaid"); a.s2r(lane, "laneid"); a.ldc(pd, "data"); a.ldc(pf, "flag"); a.ldc(pc, "count")
    a.imad(t, cta, 32, lane)                                   # global word index (one word per lane)
    a.imad(addr, t, 4, pd)
    a.stg(addr, t)                                             # payload word i holds the value i
    a.membar("gpu")                                            # this CTA's payload is visible before it is counted
    a.isetp("eq", p0, lane, 0)
    with a.guard(p0):
        a.atom(old, pc, 1, "add")                              # count finished CTAs on the receiver
        a.isetp("eq", last, old, nctas - 1)
    with a.guard(last):                                        # the last CTA publishes the flag
        a.mov(v, 1)
        a.stg(pf, v, sem="release")
    send = a.build()
    # GPU 1: spin on the flag, then read the data locally and write a checksum
    b = Asm("recv")
    lane, f, d, p, pd, pf, po, t, acc = (b.reg() for _ in range(9))
    b.s2r(lane, "laneid"); b.ldc(pd, "data"); b.ldc(pf, "flag"); b.ldc(po, "out")
    b.label("poll")
    b.ldg(f, pf, 0, 4, cache="cv", sem="acquire")
    b.setp_eq(p, f, 0)
    b.brnz(p, "poll")
    b.mov(acc, 0)
    for k in range(nbytes // 128):                             # 32 lanes x 4 B = 128 B per load
        b.imad(t, lane, 4, pd)
        b.ldg(d, t, k * 128, 4, cache="cs")
        b.iadd(acc, acc, d)
    b.imad(t, lane, 4, po)
    b.stg(t, acc)
    recv = b.build()
    res = node.run({0: [Launch(send, nctas, 32, {"data": data, "flag": flag, "count": count})],
                    1: [Launch(recv, 1, 32, {"data": data, "flag": flag, "out": out})]})
    got = node.read(out, np.uint32, 32)
    return got, res, node


def test_two_gpus_exchange_data_over_nvlink_and_infiniband():
    nbytes = 64 * 1024
    want = np.array([sum(range(l, nbytes // 4, 32)) for l in range(32)], dtype=np.uint64) & 0xFFFFFFFF
    nv, res_nv, node = _send_recv([0, 0], nbytes)                  # same host: NVLink via NVSwitch
    ib, res_ib, _ = _send_recv([0, 1], nbytes)                     # different hosts: InfiniBand
    np.testing.assert_array_equal(nv, want)                         # GPU 1 saw all of GPU 0's writes (release / acquire)
    np.testing.assert_array_equal(ib, want)
    assert node.sims[0].memsys.nvlink_bytes >= nbytes               # the traffic crossed the link
    nv_us, ib_us = res_nv[1].time_us, res_ib[1].time_us
    assert ib_us > nv_us                                            # InfiniBand is slower: lower bandwidth, higher latency
    cfg = node.cfg
    assert nv_us * 1e3 > cfg.nvlink_latency_ns                      # at least one link latency


def test_nvlink_egress_is_shared_by_readers_of_one_gpu():
    """Two GPUs streaming from GPU 1's memory share GPU 1's egress, so each gets about half of the link."""
    from .node import Node
    size = 1 << 20

    def reader(src_gpu_addr):
        a = Asm("rd")
        cta, wid, lane, base, t = (a.reg() for _ in range(5))
        dst = [a.reg(4) for _ in range(8)]
        a.s2r(cta, "ctaid"); a.s2r(wid, "warpid"); a.s2r(lane, "laneid"); a.ldc(base, "p")
        a.imad(t, cta, 8, wid)
        a.imad(t, t, 8 * 512, base)
        a.imad(t, lane, 16, t)
        for i in range(8):
            a.ldg(dst[i], t, i * 512, 16, cache="cs")
        return a.build()

    def run(readers):
        node = Node(3)
        buf = node.alloc(1, size)
        node.write(buf, np.zeros(size // 4, np.uint32))
        launches = {g: [Launch(reader(buf), size // (8 * 8 * 512), 8 * 32, {"p": buf})] for g in readers}
        res = node.run(launches)
        return {g: res[g].kernels[0].cycles for g in readers}, node
    (alone,), node = (lambda r: (list(r[0].values()), r[1]))(run([0]))
    both, _ = run([0, 2])
    link_cycles = size / node.sims[0].memsys.nvlink_bpc                 # one more megabyte through GPU 1's egress
    # the second reader adds one megabyte of traffic on the shared port, not another independent link
    assert abs((max(both.values()) - alone) - link_cycles) < 0.15 * link_cycles, (both, alone, link_cycles)


def test_tlb_levels_are_shared_like_the_ampere_reverse_engineering():
    """Zhang et al. (A100): the L1 TLB is shared by the two SMs of a TPC, the L2 TLB by a GPC, the L3 by all SMs.
    Three CTAs on SM 0, 1 (one TPC) and 2 (next TPC, same GPC) touch one page in turn."""
    sim = Sim(H200())
    page = sim.mem.alloc(1 << 22)
    page = (page + (1 << 21) - 1) // (1 << 21) * (1 << 21)
    a = Asm("tlb3")
    cta, d, p, v = a.reg(), a.reg(), a.reg(), a.reg()
    a.s2r(cta, "ctaid"); a.imad(d, cta, 3000, 0); a.ldc(p, "p")
    a.nanosleep(d)
    a.ldg(v, p, 0, 4, cache="cs")
    sim.launch(a.build(), 3, 32, {"p": page})
    ms = sim.memsys
    assert ms.tlb_walks == 1                       # only the first access walks the page table
    assert ms.tlb_l1_miss == 2                     # SM 1 shares SM 0's L1 TLB; SM 2 (another TPC) misses it
    assert sim.sms[0].tlb is sim.sms[1].tlb and sim.sms[1].tlb is not sim.sms[2].tlb


def _stream_from(node, readers, owner, size=1 << 19):
    """Each reader GPU streams `size` bytes out of `owner`'s memory; returns {reader: cycles}."""
    def reader():
        a = Asm("rd")
        cta, wid, lane, base, t = (a.reg() for _ in range(5))
        dst = [a.reg(4) for _ in range(8)]
        a.s2r(cta, "ctaid"); a.s2r(wid, "warpid"); a.s2r(lane, "laneid"); a.ldc(base, "p")
        a.imad(t, cta, 8, wid); a.imad(t, t, 8 * 512, base); a.imad(t, lane, 16, t)
        for i in range(8):
            a.ldg(dst[i], t, i * 512, 16, cache="cs")
        return a.build()
    buf = node.alloc(owner, size)
    node.write(buf, np.zeros(size // 4, np.uint32))
    res = node.run({g: [Launch(reader(), size // (8 * 8 * 512), 8 * 32, {"p": buf})] for g in readers})
    return {g: res[g].kernels[0].cycles for g in readers}


def test_switch_fabric_queues_limit_throughput():
    from .node import Node
    base = _stream_from(Node(2), [0], 1)[0]
    slow_switch = _stream_from(Node(2, cfg=H200(nvswitch_tbps=0.05)), [0], 1)[0]       # 4 chips x 50 GB/s = 200 GB/s
    assert slow_switch > 1.4 * base                                                      # the NVSwitch, not the link, limits
    # four GPUs on two hosts all read across the (oversubscribed) InfiniBand level
    hosts = [0, 0, 1, 1]
    loose = _stream_from(Node(4, hosts=hosts, cfg=H200(ib_oversubscription=1.0)), [0, 1], 2)
    tight = _stream_from(Node(4, hosts=hosts, cfg=H200(ib_oversubscription=8.0)), [0, 1], 2)
    assert max(tight.values()) > 1.5 * max(loose.values())


def _ring(sim, hop_addrs):
    for h, nxt in zip(hop_addrs, hop_addrs[1:] + hop_addrs[:1]):
        sim.mem.write(h, np.array([nxt], np.uint32))


def _tlb_prog(n):
    a = Asm("tlb")
    ptr = a.reg(); a.ldc(ptr, "p")
    for _ in range(n):
        a.ldg(ptr, ptr, 0, 4, cache="cs")
    return a.build()


def _hop_mean(sim, skip=1):
    c = [e[0] for e in sim.issue_log if e[6] == "LDG"]
    return float(np.mean(np.diff(c)[skip:]))


def test_tlb_reach_and_page_walks():
    page = 1 << 21
    sim = Sim(H200())
    sim.log_issue = True
    base = sim.mem.alloc(40 * page)
    base = (base + page - 1) // page * page
    part0 = lambda o: sim.memsys._sub((base + o) >> 5) == 0
    def pick(page_idx):
        return next(base + page_idx * page + o for o in range(0, 4096, 128) if part0(page_idx * page + o))
    many = [pick(i) for i in range(20)]                   # 20 pages: more than the 16-entry L1 TLB
    _ring(sim, many)
    prog = _tlb_prog(41)
    sim.launch(prog, 1, 32, {"p": many[0]})               # cold: page walks, fills L2 and the L2 TLB
    cold = [e[0] for e in sim.issue_log if e[6] == "LDG"]
    assert sim.memsys.tlb_walks == 20
    assert max(np.diff(cold)) >= sim.cfg.tlb_walk_cycles + sim.cfg.lat_hbm      # a cold hop pays walk + DRAM
    sim.launch(prog, 1, 32, {"p": many[0]})               # warm L2 + L2 TLB; the 16-entry L1 TLB thrashes
    thrash = _hop_mean(sim)
    sect = [base + o for o in range(0, 4096, 128) if part0(o)][:20]      # 20 sectors in ONE page
    _ring(sim, sect)
    sim.launch(prog, 1, 32, {"p": sect[0]}); sim.launch(prog, 1, 32, {"p": sect[0]})
    onepage = _hop_mean(sim)
    assert sim.cfg.lat_l2 - 5 <= onepage <= sim.cfg.lat_l2 + 15, onepage
    assert 80 <= thrash - onepage <= 100, (thrash, onepage)             # L1 TLB miss + L2 TLB hit = +90


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
