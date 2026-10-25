"""Global memory contents plus the L1 / L2 / HBM timing model (32-byte sectors)."""

from collections import OrderedDict

import numpy as np

from .frontend import LRUSet

SECTOR = 32


class TLBLevel:
    """A TLB level whose entries each cover `subs` consecutive pages through sub-entries (Zhang et al.):
    LRU over entries, one valid bit per page inside an entry."""

    def __init__(self, entries: int, subs: int):
        self.cap, self.subs = entries, subs
        self.d: OrderedDict[int, int] = OrderedDict()

    def hit(self, page: int) -> bool:
        g = page // self.subs
        m = self.d.get(g)
        if m is not None and (m >> (page % self.subs)) & 1:
            self.d.move_to_end(g)
            return True
        return False

    def insert(self, page: int) -> None:
        g = page // self.subs
        self.d[g] = self.d.get(g, 0) | (1 << (page % self.subs))
        self.d.move_to_end(g)
        while len(self.d) > self.cap:
            self.d.popitem(last=False)

    def clear(self) -> None:
        self.d.clear()


class EccError(RuntimeError):
    """An uncorrectable (double-bit) DRAM error."""


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (uint16), round to nearest even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


class L2Cache:
    """Set-associative, 128 B lines tagged per line, 32 B sectors valid individually, LRU per set."""

    def __init__(self, nbytes: int, ways: int, line: int):
        self.ways = ways
        self.nsets = nbytes // (line * ways)
        self.sets: dict[int, OrderedDict] = {}

    def _set(self, line: int) -> int:
        return ((line * 2654435761) >> 5) % self.nsets

    def lookup(self, s: int):
        od = self.sets.get(self._set(s >> 2))
        if od is None:
            return None
        ent = od.get(s >> 2)
        if ent is None:
            return None
        od.move_to_end(s >> 2)
        return ent[s & 3]

    def fill(self, s: int, ready: float, dirty: bool = False) -> list:
        """Install a sector; returns the dirty sectors of any line evicted to make room."""
        line = s >> 2
        od = self.sets.setdefault(self._set(line), OrderedDict())
        ent = od.get(line)
        evicted = []
        if ent is None:
            if len(od) >= self.ways:
                ln, old = od.popitem(last=False)
                evicted = [ln * 4 + i for i in range(4) if (old[4] >> i) & 1]
            ent = od[line] = [None] * 4 + [0]
        ent[s & 3] = ready
        if dirty:
            ent[4] |= 1 << (s & 3)
        od.move_to_end(line)
        return evicted

    def clear(self) -> None:
        self.sets.clear()


class GlobalMemory:
    """Backing store. Device pointers are `base + offset`. In a multi-GPU Node every GPU gets a view with
    its own allocation region over one shared array, so peer memory is real memory."""

    def __init__(self, nbytes: int, base: int = 0, shared: "GlobalMemory | None" = None, gpu: int = 0,
                 lo: int = 0, hi: int | None = None):
        if shared is None:
            self.bytes = np.zeros(nbytes, np.uint8)
            self.u32 = self.bytes.view(np.uint32)
            self.views = None
        else:
            self.bytes, self.u32, self.views = shared.bytes, shared.u32, shared.views
        self.remote: list = []              # (lo, hi) ranges mapped as peer memory (single-GPU Sim.map_peer)
        self.base, self.gpu, self.lo = base, gpu, lo
        self.hi = self.bytes.size if hi is None else hi
        self.top = lo + 256  # keep the first 256 B of a region unmapped-looking

    def alloc(self, nbytes: int) -> int:
        off = (self.top + 255) & ~255
        self.top = off + nbytes
        if self.top > self.hi:
            raise MemoryError("simulated global memory exhausted; raise H200.mem_bytes")
        return self.base + off

    def valid(self, lo: int, hi_end: int) -> bool:
        """Is the byte range [lo, hi_end) (offsets) inside allocated memory? In a node, any GPU's region counts."""
        if self.views is None:
            return lo >= 256 and hi_end <= self.top
        return any(v.lo + 256 <= lo and hi_end <= v.top for v in self.views)

    def owner(self, byte_addr: int) -> int:
        if self.views is None:
            return self.gpu
        return min(len(self.views) - 1, (byte_addr - self.base) // (self.hi - self.lo))

    def alloc_remote(self, nbytes: int) -> int:
        """Memory on a peer GPU: same functional behaviour, but every access crosses NVLink."""
        addr = self.alloc(nbytes)
        self.remote.append((addr, addr + nbytes))
        return addr

    def is_remote(self, byte_addr: int) -> bool:
        if self.views is not None:
            return self.owner(byte_addr) != self.gpu
        for lo, hi in self.remote:
            if lo <= byte_addr < hi:
                return True
        return False

    def write(self, addr: int, arr: np.ndarray) -> None:
        raw = np.ascontiguousarray(arr).view(np.uint8).ravel()
        off = addr - self.base
        self.bytes[off:off + raw.size] = raw

    def read(self, addr: int, dtype, count: int) -> np.ndarray:
        n = np.dtype(dtype).itemsize * count
        off = addr - self.base
        return self.bytes[off:off + n].view(dtype).copy()


class SectorCache:
    """LRU cache of 32 B sectors; each entry remembers when its data lands. `data` optionally holds a
    snapshot of the sector's 8 words (the L1 is not coherent, so it can go stale)."""

    def __init__(self, capacity_sectors: int):
        self.cap = capacity_sectors
        self.d: OrderedDict[int, float] = OrderedDict()
        self.data: dict[int, np.ndarray] = {}

    def lookup(self, s: int):
        r = self.d.get(s)
        if r is not None:
            self.d.move_to_end(s)
        return r

    def fill(self, s: int, ready: float) -> None:
        self.d[s] = ready
        self.d.move_to_end(s)
        while len(self.d) > self.cap:
            old, _ = self.d.popitem(last=False)
            self.data.pop(old, None)

    def clear(self) -> None:
        self.d.clear()
        self.data.clear()


class MemorySystem:
    def __init__(self, cfg, mem=None, node=None, gpu_id=0):
        self.cfg = cfg
        self.node = node
        self.gpu_id = gpu_id
        self.mem = mem if mem is not None else GlobalMemory(cfg.mem_bytes, cfg.va_base)
        self.l2 = L2Cache(cfg.l2_bytes, cfg.l2_ways, cfg.l2_line)
        self.nvl_in_free = self.nvl_out_free = 0.0
        self.ib_in_free = self.ib_out_free = 0.0
        self.ecc_faults: dict[int, int] = {}       # sector -> number of flipped bits waiting in DRAM
        self.tlb_gpc = [TLBLevel(cfg.tlb_l2_entries, cfg.tlb_subs) for _ in cfg.gpc_sizes]
        self.tlb_l3 = TLBLevel(cfg.tlb_l3_entries, cfg.tlb_subs)
        self.walk_free = [0.0] * cfg.tlb_walkers
        self.last_read_t = 0                   # cycle at which the last load's data is read (see load())
        self.last_l1_hits: list = []           # sectors the last load hit in L1 / filled into L1
        self.last_l1_fills: list = []
        self.l2_free = [0.0, 0.0]              # per partition
        self.xbar_free = 0.0                   # partition-to-partition crossing
        self.hbm_free = [0.0] * cfg.hbm_channels
        self.reset_stats()

    def inject_ecc_error(self, addr: int, bits: int = 1) -> None:
        """Flip `bits` bits in the DRAM sector holding `addr`. With ECC on, one bit is corrected on the next
        DRAM read (costing time) and two or more abort the kernel; with ECC off the data is silently corrupted."""
        if not self.cfg.ecc_enabled:
            off = addr - self.mem.base
            for b in range(bits):
                self.mem.bytes[off + b // 8] ^= np.uint8(1 << (b % 8))
            return
        s = addr >> 5
        self.ecc_faults[s] = self.ecc_faults.get(s, 0) + bits

    def _ecc_check(self, s: int) -> int:
        """Extra cycles to deliver sector s from DRAM; raises on an uncorrectable error."""
        bits = self.ecc_faults.pop(s, 0)
        if not bits:
            return 0
        if bits >= 2:
            raise EccError(f"uncorrectable ECC error (double-bit) in DRAM sector {s * SECTOR:#x} "
                           f"(the driver reports XID 48 and the kernel is aborted)")
        self.ecc_corrected += 1
        return self.cfg.ecc_correct_cycles

    # ---- peers: NVLink (via a non-blocking NVSwitch) inside a host, InfiniBand between hosts ----
    def _link(self, owner):
        """(bytes/cycle, latency cycles, is_ib) of the path to GPU `owner`."""
        c = self.cfg
        if self.node is not None and self.node.hosts[owner] != self.node.hosts[self.gpu_id]:
            return (c.ib_gbs * 1e9 * c.ib_efficiency / (c.clock_ghz * 1e9), int(c.ib_latency_ns * c.clock_ghz), True)
        return self.nvlink_bpc, self.nvlink_lat, False

    def _remote_read(self, s: int, t0: float) -> float:
        """Data flows from the owner's egress to this GPU's ingress; both are shared by every flow on that port."""
        owner = self.mem.owner(s * SECTOR)
        peer = self.node.sims[owner].memsys if self.node is not None else self
        bpc, lat, ib = self._link(owner)
        a, b = ("ib_in_free", "ib_out_free") if ib else ("nvl_in_free", "nvl_out_free")
        t = max(getattr(self, a), getattr(peer, b), t0) + SECTOR / bpc
        setattr(self, a, t)
        setattr(peer, b, t)
        self.nvlink_bytes += SECTOR
        return self._fabric(s, t, ib) + lat

    def _remote_write(self, s: int, t0: float) -> float:
        owner = self.mem.owner(s * SECTOR)
        peer = self.node.sims[owner].memsys if self.node is not None else self
        bpc, lat, ib = self._link(owner)
        a, b = ("ib_out_free", "ib_in_free") if ib else ("nvl_out_free", "nvl_in_free")
        t = max(getattr(self, a), getattr(peer, b), t0) + SECTOR / bpc
        setattr(self, a, t)
        setattr(peer, b, t)
        self.nvlink_bytes += SECTOR
        return self._fabric(s, t, ib) + lat // 2

    def _fabric(self, s: int, t: float, ib: bool) -> float:
        """Queuing in the switches a flow crosses: one of the NVSwitch chips (flows are striped over them), or the
        oversubscribed InfiniBand first level. Without a Node there is no fabric to queue in."""
        node, c = self.node, self.cfg
        if node is None:
            return t
        if ib:
            bpc = node.n * c.ib_gbs * 1e9 * c.ib_efficiency / c.ib_oversubscription / (c.clock_ghz * 1e9)
            start = max(node.ib_fabric_free, t)
            node.ib_fabric_free = start + SECTOR / bpc
            return start                                  # delayed only if the shared level is already busy
        i = (s >> 3) % c.nvswitch_chips
        bpc = c.nvswitch_tbps * 1e12 / (c.clock_ghz * 1e9)
        start = max(node.switch_free[i], t)
        node.switch_free[i] = start + SECTOR / bpc
        return start

    @property
    def nvlink_bpc(self) -> float:
        c = self.cfg
        return c.nvlink_gbs_per_dir * 1e9 * c.nvlink_efficiency / (c.clock_ghz * 1e9)

    @property
    def nvlink_lat(self) -> int:
        return int(self.cfg.nvlink_latency_ns * self.cfg.clock_ghz)

    def reset_stats(self):
        self.hbm_bytes = self.hbm_write_bytes = 0
        self.l2_hit = self.l2_miss = self.l1_hit = self.st_bytes = 0
        self.l2_prefetch = 0
        self.tlb_l1_miss = self.tlb_walks = 0
        self.nvlink_bytes = 0
        self.ecc_corrected = 0

    def make_l1(self) -> SectorCache:
        return SectorCache(self.cfg.l1_max_bytes // SECTOR)

    def flush_l2(self) -> None:
        self.l2.clear()

    @staticmethod
    def _partition(s: int) -> int:
        """DRAM / L2 NUMA node (0 or 1) of sector s: a 64 KB pattern at 4 KB granularity (Liu et al.,
        H100): id = floor((x + 1 - H(x - 8)) / 2) mod 2 with x = (address mod 64 KB) / 4 KB."""
        x = ((s * SECTOR) & 0xFFFF) >> 12
        return ((x + 1 - (1 if x >= 8 else 0)) >> 1) & 1

    @classmethod
    def _sub(cls, s: int) -> int:
        """Sub-NUMA node (0-3): two per NUMA node, interleaved by address (2 KB granularity is assumed)."""
        return 2 * cls._partition(s) + ((s * SECTOR >> 11) & 1)

    def _l2_access(self, sm, s: int, t0: float) -> float:
        """Serve one sector request arriving at the L2 at t0; returns when the L2 has answered."""
        c = self.cfg
        part = self._partition(s)
        t = t0
        extra_hit = extra_miss = 0
        if part != sm.part:                     # remote NUMA node: cross the inter-partition link
            self.xbar_free = max(self.xbar_free, t) + SECTOR / c.l2_cross_bytes_per_cycle
            t = self.xbar_free
            extra_hit, extra_miss = c.lat_l2_far_extra, c.lat_hbm_far_extra
        elif self._sub(s) != sm.sub:            # same half, other sub-NUMA node
            extra_hit = extra_miss = c.lat_subnuma_extra
        self.l2_free[part] = max(self.l2_free[part], t) + SECTOR / (c.l2_bytes_per_cycle / 2)
        return self.l2_free[part], extra_hit, extra_miss

    def flush_tlb(self, sms=()) -> None:
        for t in self.tlb_gpc:
            t.clear()
        self.tlb_l3.clear()
        for sm in sms:
            sm.tlb.clear()

    def _tlb(self, sm, page: int, cycle: int) -> int:
        """Extra cycles to translate `page`: L1 hit 0 (or the rest of a translation another request already
        started); L2 (per GPC) hit; L3 (shared) hit; else a page walk on one of the walkers. Hits are reinserted
        into the levels above. An L1 entry remembers when its translation completes."""
        c = self.cfg
        ready = sm.tlb.get(page)
        if ready is not None:
            return max(0, int(ready - cycle))
        self.tlb_l1_miss += 1
        l2 = self.tlb_gpc[sm.gpc]
        if l2.hit(page):
            pen = c.tlb_l2_hit_cycles
        elif self.tlb_l3.hit(page):
            l2.insert(page)
            pen = c.tlb_l3_hit_cycles
        else:
            self.tlb_walks += 1
            self.tlb_l3.insert(page)
            l2.insert(page)
            w = min(range(len(self.walk_free)), key=self.walk_free.__getitem__)
            start = max(self.walk_free[w], cycle)
            self.walk_free[w] = start + c.tlb_walk_cycles
            pen = int(start - cycle) + c.tlb_walk_cycles
        sm.tlb.put(page, cycle + pen)
        return pen

    def _writeback(self, evicted, t: float) -> None:
        """Dirty sectors leaving the L2 occupy their HBM channel; nothing waits on them."""
        ch_bw = self.cfg.hbm_bytes_per_cycle / self.cfg.hbm_channels
        for s in evicted:
            ch = (((s >> 3) * 2654435761) >> 7) % self.cfg.hbm_channels
            self.hbm_free[ch] = max(self.hbm_free[ch], t) + SECTOR / ch_bw
            self.hbm_write_bytes += SECTOR

    def load(self, sm, sectors, cycle: int, use_l1: bool, width: int = 4, pf: int = 0) -> int:
        """Completion cycle of a warp load touching `sectors` (unique sector ids)."""
        c = self.cfg
        l1_lat = c.lat_l1 + dict(c.lat_load_width_extra).get(width, 0)
        pages: dict[int, int] = {}
        self.last_l1_hits, self.last_l1_fills = [], []
        read_t = cycle + 1                       # when the request is serviced by the cache that holds the data
        ch_bw = c.hbm_bytes_per_cycle / c.hbm_channels
        done = float(cycle)
        remote = bool(self.mem.remote) or self.mem.views is not None
        for s in sectors:
            s = int(s)
            if remote and self.mem.is_remote(s * SECTOR):    # peer GPU memory: NVLink / InfiniBand, uncached
                sm.port_free = max(sm.port_free, cycle) + SECTOR / c.sm_port_bytes_per_cycle
                ret = self._remote_read(s, sm.port_free)
                done = max(done, ret)
                read_t = max(read_t, ret - self._link(self.mem.owner(s * SECTOR))[1] // 2)
                continue
            if use_l1:
                r = sm.l1.lookup(s)
                if r is not None:
                    self.l1_hit += 1
                    self.last_l1_hits.append(s)
                    done = max(done, cycle + l1_lat, r)
                    read_t = max(read_t, cycle + l1_lat, r)
                    continue
            sm.port_free = max(sm.port_free, cycle) + SECTOR / c.sm_port_bytes_per_cycle
            pen = 0
            if c.model_tlb:                          # translation happens before the request reaches the L2,
                page = s >> (c.tlb_page_shift - 5)   # so a TLB miss delays everything downstream (and the fill)
                pen = pages.get(page)
                if pen is None:
                    pen = pages[page] = self._tlb(sm, page, cycle)
            t1, extra, extra_miss = self._l2_access(sm, s, sm.port_free)   # servers are reserved at the undelayed time
            r = self.l2.lookup(s)
            if r is not None:
                self.l2_hit += 1
                t = max(t1 + c.lat_l2 + extra + pen, r)
            else:
                self.l2_miss += 1
                self.hbm_bytes += SECTOR
                ch = (((s >> 3) * 2654435761) >> 7) % c.hbm_channels
                self.hbm_free[ch] = max(self.hbm_free[ch], t1) + SECTOR / ch_bw
                burst = c.dram_burst // SECTOR           # DRAM -> L2 moves 64 B: the neighbour sector rides along
                if burst > 1:
                    mate = s ^ 1
                    if self.l2.lookup(mate) is None:
                        self.hbm_bytes += SECTOR
                        self.hbm_free[ch] = max(self.hbm_free[ch], t1) + SECTOR / ch_bw
                        self.l2.fill(mate, self.hbm_free[ch] + c.lat_hbm + extra_miss + pen)
                t = self.hbm_free[ch] + c.lat_hbm + extra_miss + pen
                if self.ecc_faults:
                    t += self._ecc_check(s)
                    self._ecc_check(s ^ 1)
                self._writeback(self.l2.fill(s, t), t1)
                if pf:                                   # .L2::<pf>B: the rest of the aligned window rides along
                    n = pf // SECTOR
                    first = s // n * n
                    for q in range(first, first + n):
                        if q != s and self.l2.lookup(q) is None:
                            chq = (((q >> 3) * 2654435761) >> 7) % c.hbm_channels
                            self.hbm_free[chq] = max(self.hbm_free[chq], t1) + SECTOR / ch_bw
                            self.hbm_bytes += SECTOR
                            self.l2_prefetch += 1
                            self.l2.fill(q, self.hbm_free[chq] + c.lat_hbm + extra_miss + pen)
            if use_l1:
                sm.l1.fill(s, t)
                self.last_l1_fills.append(s)
            # the L2 services the read once the request arrives (one way: half an L2 hit, plus translation),
            # or once the line arrives from DRAM if it was not there yet
            arrive = t1 + pen + (c.lat_l2 + extra) // 2
            if r is None:
                arrive = max(arrive, t - c.lat_l2 // 2)
            read_t = max(read_t, arrive)
            done = max(done, t)
        self.last_read_t = int(min(read_t, done))
        return int(done) + 1

    def store(self, sm, sectors, cycle: int) -> int:
        """Write-through to L2; costs port and L2 bandwidth. Returns when the last sector lands."""
        c = self.cfg
        last = float(cycle)
        remote = bool(self.mem.remote) or self.mem.views is not None
        for s in sectors:
            s = int(s)
            if remote and self.mem.is_remote(s * SECTOR):    # posted write to a peer
                sm.port_free = max(sm.port_free, cycle) + SECTOR / c.sm_port_bytes_per_cycle
                last = max(last, self._remote_write(s, sm.port_free))
                continue
            sm.port_free = max(sm.port_free, cycle) + SECTOR / c.sm_port_bytes_per_cycle
            t, extra, _ = self._l2_access(sm, s, sm.port_free)
            self._writeback(self.l2.fill(s, t, dirty=True), t)
            self.st_bytes += SECTOR
            last = max(last, t + extra + c.lat_l2 // 2 + c.store_drain_cycles)   # one way to L2 plus the SM write path (assumed)
        return int(last) + 1
