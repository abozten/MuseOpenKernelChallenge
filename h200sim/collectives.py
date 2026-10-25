"""NVSHMEM / NCCL-style collectives written as ordinary kernels over a Node.

There is no library: an all-gather is every GPU pushing its chunk into every peer's output with stores
(`put`), an all-reduce is a reduce-scatter by remote `red.add` into the owner's shard followed by an all-gather
of the reduced shards, and a barrier is a counter incremented with a remote atomic after a `fence.sc` and
spun on with `ld.acquire`. What gets timed is what the hardware does: peer stores over NVLink / InfiniBand,
remote atomics, fences and acquire loads. Bandwidths are reported the NCCL way: algbw = bytes / time and
busbw = algbw x 2(n-1)/n for all-reduce, (n-1)/n for all-gather.
"""

import numpy as np

from .core import Launch
from .isa import Asm
from .node import Node

WARPS = 8
PIECE = WARPS * 512          # bytes one CTA moves per peer


def _barrier(a, n, nctas, counter_regs, own_counter, tmp):
    """After the data movement: fence, signal every peer's counter once per CTA, wait for all n * nctas signals."""
    lane, p0, old, v = tmp
    a.membar("gpu", "sc")                                 # this CTA's stores are visible before it is counted
    a.isetp("eq", p0, lane, 0)
    with a.guard(p0):
        for r in counter_regs:
            a.atom(old, r, 1, "add")
    a.label("wait" + str(id(counter_regs)))
    a.ldg(v, own_counter, 0, 4, cache="cv", sem="acquire")
    a.isetp("lt", p0, v, n * nctas)
    with a.guard(p0):
        a.bra("wait" + str(id(counter_regs)))


def all_gather(node: Node, chunk_bytes: int):
    """Every GPU g holds `chunk_bytes` of data; afterwards every GPU holds all n chunks. Returns (results, busbw GB/s)."""
    n = node.n
    assert chunk_bytes % PIECE == 0
    nctas = chunk_bytes // PIECE
    src = [node.alloc(g, chunk_bytes) for g in range(n)]
    dst = [node.alloc(g, chunk_bytes * n) for g in range(n)]
    cnt = [node.alloc(g, 128) for g in range(n)]
    rng = np.random.default_rng(0)
    data = [rng.integers(0, 2 ** 31, chunk_bytes // 4, dtype=np.uint32) for _ in range(n)]
    for g in range(n):
        node.write(src[g], data[g])
    launches = {}
    for g in range(n):
        a = Asm(f"allgather{g}")
        cta, wid, lane, off, v, p0, old, own = (a.reg() for _ in range(8))
        s_ptr, d_ptr = a.reg(), a.reg()
        val = a.reg(4)
        peers = [a.reg() for _ in range(n)]
        cnts = [a.reg() for _ in range(n)]
        a.s2r(cta, "ctaid"); a.s2r(wid, "warpid"); a.s2r(lane, "laneid")
        a.imad(off, cta, WARPS, wid)
        a.imad(off, off, 512, 0)
        a.imad(off, lane, 16, off)                         # byte offset of this lane's 16 B inside the chunk
        a.ldc(s_ptr, "src")
        a.iadd(s_ptr, s_ptr, off)
        a.ldg(val, s_ptr, 0, 16, cache="cs")
        for p in range(n):
            a.ldc(peers[p], f"dst{p}"); a.ldc(cnts[p], f"cnt{p}")
            a.iadd(peers[p], peers[p], off)
            a.stg(peers[p], val, g * chunk_bytes, 16)       # put into peer p's output at this GPU's slot
        a.ldc(own, f"cnt{g}")
        _barrier(a, n, nctas, cnts, own, (lane, p0, old, v))
        params = {"src": src[g], **{f"dst{p}": dst[p] for p in range(n)}, **{f"cnt{p}": cnt[p] for p in range(n)}}
        launches[g] = [Launch(a.build(), nctas, WARPS * 32, params)]
    res = node.run(launches)
    cycles = max(res[g].kernels[0].cycles for g in range(n))
    seconds = cycles / (node.cfg.clock_ghz * 1e9)
    ok = all(np.array_equal(node.read(dst[g] + p * chunk_bytes, np.uint32, chunk_bytes // 4), data[p])
             for g in range(n) for p in range(n))
    busbw = (n - 1) / n * (chunk_bytes * n) / seconds / 1e9
    return {"correct": ok, "cycles": cycles, "us": seconds * 1e6, "busbw_gbs": busbw}


def all_reduce_sum(node: Node, nbytes: int):
    """fp32 sum over n GPUs: reduce-scatter by remote red.add into the owner's shard, barrier, all-gather."""
    n = node.n
    shard = nbytes // n
    assert shard % PIECE == 0
    nctas_shard = shard // PIECE
    nctas = nctas_shard * n                                 # CTAs 0..n-1 of each shard walk its pieces
    src = [node.alloc(g, nbytes) for g in range(n)]
    acc = [node.alloc(g, shard) for g in range(n)]          # each GPU's reduced shard (zero-initialised)
    out = [node.alloc(g, nbytes) for g in range(n)]
    cnt = [node.alloc(g, 256) for g in range(n)]
    rng = np.random.default_rng(1)
    data = [rng.integers(-8, 8, nbytes // 4).astype(np.float32) for _ in range(n)]
    for g in range(n):
        node.write(src[g], data[g])
        node.write(acc[g], np.zeros(shard // 4, np.float32))
    launches = {}
    for g in range(n):
        a = Asm(f"allreduce{g}")
        cta, wid, lane, off, owner, p0, old, v, ptr = (a.reg() for _ in range(9))
        val = a.reg(4)
        cnts = [a.reg() for _ in range(n)]
        accp = [a.reg() for _ in range(n)]
        outp = [a.reg() for _ in range(n)]
        own = a.reg()
        a.s2r(cta, "ctaid"); a.s2r(wid, "warpid"); a.s2r(lane, "laneid")
        # phase 1: CTA c reduces piece (c % nctas_shard) of shard (c // nctas_shard) -- each lane adds 4 floats remotely
        for p in range(n):
            a.ldc(cnts[p], f"cnt{p}"); a.ldc(accp[p], f"acc{p}"); a.ldc(outp[p], f"out{p}")
        a.ldc(own, f"cnt{g}")
        a.ldc(ptr, "src")
        # global byte offset within this GPU's source buffer for (cta, warp, lane)
        a.imad(off, cta, WARPS, wid); a.imad(off, off, 512, 0); a.imad(off, lane, 16, off)
        a.iadd(ptr, ptr, off)
        a.ldg(val, ptr, 0, 16, cache="cs")
        # shard owner of this CTA: cta // nctas_shard, computed by compare chain (n is small)
        pin, q = a.preg(), a.preg()
        for p in range(n):
            lo, hi = p * nctas_shard, (p + 1) * nctas_shard
            a.isetp("ge", pin, cta, lo)
            a.isetp("lt", q, cta, hi)
            a.psetp("and", pin, pin, q)
            with a.guard(pin):
                t = a.reg()
                a.iadd(t, accp[p], 0)
                a.iadd(t, t, off)
                a.iadd(t, t, -p * shard)
                for j in range(4):
                    a.red_add(t, val + j, j * 4)
        _barrier(a, n, nctas, cnts, own, (lane, p0, old, v))
        # phase 2: the CTAs that own shard g's pieces push my reduced shard to every peer's output
        mine = a.preg(); m2 = a.preg()
        a.isetp("ge", mine, cta, g * nctas_shard); a.isetp("lt", m2, cta, (g + 1) * nctas_shard); a.psetp("and", mine, mine, m2)
        with a.guard(mine):
            t = a.reg()
            a.iadd(t, accp[g], off); a.iadd(t, t, -g * shard)
            a.ldg(val, t, 0, 16, cache="cv", sem="acquire")
            for p in range(n):
                d = a.reg()
                a.iadd(d, outp[p], off)
                a.stg(d, val, 0, 16)
        a.membar("gpu", "sc")
        launches[g] = [Launch(a.build(), nctas, WARPS * 32,
                              {"src": src[g], **{f"acc{p}": acc[p] for p in range(n)},
                               **{f"out{p}": out[p] for p in range(n)}, **{f"cnt{p}": cnt[p] for p in range(n)}})]
    res = node.run(launches)
    cycles = max(res[g].kernels[0].cycles for g in range(n))
    seconds = cycles / (node.cfg.clock_ghz * 1e9)
    want = sum(data)
    ok = all(np.allclose(node.read(out[g], np.float32, nbytes // 4), want) for g in range(n))
    busbw = 2 * (n - 1) / n * nbytes / seconds / 1e9
    return {"correct": ok, "cycles": cycles, "us": seconds * 1e6, "busbw_gbs": busbw}


# ---------------------------------------------------------------------------------------------------
# Ring algorithms (the structure NCCL uses): the buffer is split into n chunks, each chunk into `nch` slices, and
# every slice is carried by its own CTA (a "channel") around the ring independently, so the channels pipeline.
#
#   reduce-scatter: n-1 steps; at step s GPU g adds its partial of chunk (g - s) mod n into the next GPU's buffer
#                   (remote red.add), after waiting for the previous GPU's step s-1 to land;
#   all-gather:     n-1 more steps; GPU g forwards chunk (g + 1 - t) mod n with plain stores.
#
# A step ends with: every warp fences its stores (fence.sc), the CTA syncs, and one thread increments the next
# GPU's per-channel flag; the next step starts when this GPU's own flag shows the previous GPU has done the same.
# ---------------------------------------------------------------------------------------------------

def _ring_kernel(g, n, nch, slice_bytes, chunk_bytes, reduce):
    a = Asm(f"ring{g}")
    cta, wid, lane, off, v, ws, wn, fs, fn = (a.reg() for _ in range(9))
    val = a.reg(4)
    p0, pw0, pl, ps = a.preg(), a.preg(), a.preg(), a.preg()
    old, t = a.reg(), a.reg()
    a.s2r(cta, "ctaid"); a.s2r(wid, "warpid"); a.s2r(lane, "laneid")
    a.ldc(ws, "w_self"); a.ldc(wn, "w_next"); a.ldc(fs, "flag_self"); a.ldc(fn, "flag_next")
    a.imad(off, cta, slice_bytes, 0)                       # this channel's slice inside a chunk
    a.imad(t, wid, 512, 0); a.iadd(off, off, t)
    a.imad(t, lane, 16, 0); a.iadd(off, off, t)
    a.iadd(fs, fs, 0); a.imad(t, cta, 4, 0)
    a.iadd(fs, fs, t); a.iadd(fn, fn, t)                   # per-channel flag words
    a.isetp("eq", pw0, wid, 0)
    a.isetp("eq", pl, lane, 0)
    a.psetp("and", p0, pw0, pl)
    iters = slice_bytes // (WARPS * 512)

    def wait_flag(target, label):
        a.label(label)
        with a.guard(pw0):                                 # warp 0 spins on the flag; the other warps wait at the barrier
            a.ldg(v, fs, 0, 4, cache="cv", sem="acquire")
            a.isetp("lt", ps, v, target)
        with a.guard(ps):
            a.bra(label)
        a.bar()

    def send(chunk, kind):
        for it in range(iters):
            base = chunk * chunk_bytes + it * WARPS * 512
            a.iadd(t, ws, off); a.iadd(t, t, base)
            a.ldg(val, t, 0, 16, cache="cv")
            a.iadd(t, wn, off); a.iadd(t, t, base)
            if kind == "add":
                for j in range(4):
                    a.red_add(t, val + j, 4 * j)
            else:
                a.stg(t, val, 0, 16)
        a.membar("gpu", "sc")                              # my stores are visible before the flag is
        a.bar()
        with a.guard(p0):
            a.atom(old, fn, 1, "add")

    step = 0
    if reduce:
        for s in range(n - 1):
            if s > 0:
                wait_flag(s, f"w{step}"); step += 1
            send((g - s) % n, "add")
        wait_flag(n - 1, f"w{step}"); step += 1            # my own chunk is fully reduced
    for t_ in range(n - 1):
        send((g + 1 - t_) % n, "store")
        wait_flag((n - 1 if reduce else 0) + t_ + 1, f"w{step}"); step += 1
    return a.build()


def _ring_run(node: Node, nbytes: int, nch: int, reduce: bool):
    n = node.n
    chunk = nbytes // n
    slice_bytes = chunk // nch
    assert slice_bytes % (WARPS * 512) == 0 and chunk % nch == 0
    wbuf = [node.alloc(g, nbytes) for g in range(n)]
    flag = [node.alloc(g, 4 * nch + 128) for g in range(n)]
    rng = np.random.default_rng(2)
    if reduce:
        data = [rng.integers(-8, 8, nbytes // 4).astype(np.float32) for _ in range(n)]
    else:                                                  # all-gather: GPU g owns chunk (g+1) % n
        data = [rng.integers(0, 2 ** 20, nbytes // 4, dtype=np.uint32) for _ in range(n)]
    for g in range(n):
        node.write(flag[g], np.zeros(nch + 32, np.uint32))
        if reduce:
            node.write(wbuf[g], data[g])
        else:
            w = np.zeros(nbytes // 4, np.uint32)
            lo = ((g + 1) % n) * chunk // 4
            w[lo:lo + chunk // 4] = data[g][lo:lo + chunk // 4]
            node.write(wbuf[g], w)
    launches = {g: [Launch(_ring_kernel(g, n, nch, slice_bytes, chunk, reduce), nch, WARPS * 32,
                           {"w_self": wbuf[g], "w_next": wbuf[(g + 1) % n], "flag_self": flag[g],
                            "flag_next": flag[(g + 1) % n]})] for g in range(n)}
    res = node.run(launches)
    cycles = max(res[g].kernels[0].cycles for g in range(n))
    seconds = cycles / (node.cfg.clock_ghz * 1e9)
    if reduce:
        want = sum(data)
        ok = all(np.allclose(node.read(wbuf[g], np.float32, nbytes // 4), want) for g in range(n))
        busbw = 2 * (n - 1) / n * nbytes / seconds / 1e9
    else:
        want = np.zeros(nbytes // 4, np.uint32)
        for g in range(n):
            lo = ((g + 1) % n) * chunk // 4
            want[lo:lo + chunk // 4] = data[g][lo:lo + chunk // 4]
        ok = all(np.array_equal(node.read(wbuf[g], np.uint32, nbytes // 4), want) for g in range(n))
        busbw = (n - 1) / n * nbytes / seconds / 1e9
    return {"correct": ok, "cycles": cycles, "us": seconds * 1e6, "busbw_gbs": busbw}


def ring_all_gather(node: Node, nbytes: int, nch: int = 4):
    """NCCL-style ring all-gather over `nch` channels; GPU g starts owning chunk (g+1) % n."""
    return _ring_run(node, nbytes, nch, reduce=False)


def ring_all_reduce_sum(node: Node, nbytes: int, nch: int = 4):
    """NCCL-style ring all-reduce (reduce-scatter then all-gather) of fp32 over `nch` channels."""
    return _ring_run(node, nbytes, nch, reduce=True)
