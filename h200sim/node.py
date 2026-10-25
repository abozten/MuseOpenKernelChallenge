"""A multi-GPU node: several full GPU simulators over one shared memory array.

Every GPU is a complete `Sim` (its own SMs, caches, TLBs, HBM). Device memory of all GPUs lives in one
array, split into per-GPU regions, so a load, store, atomic or TMA copy to a peer address really reads or
writes the peer's memory. Peer accesses cross a link:

  * inside a host: NVLink through an NVSwitch, modeled as non-blocking -- each GPU has an egress and an
    ingress port (450 GB/s per direction x efficiency), and a flow occupies the sender's egress and the
    receiver's ingress, so two GPUs reading from one peer share that peer's egress;
  * between hosts: InfiniBand, one NIC per GPU (50 GB/s x efficiency, larger latency).

The GPUs advance in global time order: the run loop of each GPU is a generator that yields the cycle of its
next iteration, and the node always resumes the earliest one, so stores landing on one GPU are visible to
the others at the right time.
"""

import heapq

from .config import H200
from .core import ChainStats, Launch, Sim
from .memory import GlobalMemory


class Node:
    def __init__(self, n_gpus: int = 2, cfg: H200 | None = None, hosts=None, region_bytes: int = 64 << 20,
                 dep_mode: str = "scoreboard"):
        self.cfg = cfg or H200()
        self.n = n_gpus
        self.hosts = list(hosts) if hosts is not None else [0] * n_gpus
        assert len(self.hosts) == n_gpus
        backing = GlobalMemory(n_gpus * region_bytes, self.cfg.va_base)
        self.views = []
        for g in range(n_gpus):
            v = GlobalMemory(0, self.cfg.va_base, shared=backing, gpu=g, lo=g * region_bytes,
                             hi=(g + 1) * region_bytes)
            self.views.append(v)
        for v in self.views:
            v.views = self.views
        self.sims = [Sim(self.cfg, dep_mode=dep_mode, mem=self.views[g], node=self, gpu_id=g) for g in range(n_gpus)]
        self.time = 0
        self.switch_free = [0.0] * self.cfg.nvswitch_chips      # NVSwitch chip queues
        self.ib_fabric_free = 0.0                               # oversubscribed InfiniBand level-1 queue

    # ---- memory ----
    def alloc(self, gpu: int, nbytes: int) -> int:
        """Allocate device memory on `gpu`; the address is valid on every GPU of the node."""
        return self.views[gpu].alloc(nbytes)

    def write(self, addr: int, arr) -> None:
        self.views[0].write(addr, arr)

    def read(self, addr: int, dtype, count: int):
        return self.views[0].read(addr, dtype, count)

    # ---- running ----
    def run(self, launches: dict, mode: str = "graph") -> dict:
        """launches: {gpu: [Launch, ...]}. All GPUs start together; returns {gpu: ChainStats}."""
        gens, nxt, out = {}, [], {}
        self.switch_free = [0.0] * self.cfg.nvswitch_chips
        self.ib_fabric_free = 0.0
        for g, ls in launches.items():
            sim = self.sims[g]
            sim.cycle = self.time
            gens[g] = sim._run_chain(ls, mode)
            heapq.heappush(nxt, (next(gens[g]), g))
        while nxt:
            _, g = heapq.heappop(nxt)
            try:
                heapq.heappush(nxt, (gens[g].send(None), g))
            except StopIteration as done:
                out[g] = done.value
        self.time = max(self.sims[g].cycle for g in launches)
        return out
