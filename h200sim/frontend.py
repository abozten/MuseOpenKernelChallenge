"""Instruction fetch and constant caches.

Fetch: 16 B instructions, 128 B lines. Each sub-core has a small L0 I-cache; on a miss the
line comes from the SM's L1 I-cache, else from L2 (code is cached in L2 like data, so a
flushed L2 means a DRAM fetch), and a stream buffer prefetches the following lines.
Constants: LDC goes through an L0 constant cache per sub-core, then an SM-level cache, then L2.
"""

from collections import OrderedDict


class LRUSet:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

    def clear(self):
        self.d.clear()


class FrontEnd:
    def __init__(self, cfg):
        self.cfg = cfg
        self.l2_code = {}              # line -> cycle its data is in L2 (shared by all SMs)
        self.l2_const = {}
        self.fetch_miss_l0 = self.fetch_miss_l1 = self.fetch_miss_l2 = 0

    def flush_l2(self):
        self.l2_code.clear()
        self.l2_const.clear()

    def new_sm(self):
        c = self.cfg
        return LRUSet(c.l1i_bytes // c.icache_line), LRUSet(64)

    def new_smsp(self):
        c = self.cfg
        return LRUSet(c.l0i_bytes // c.icache_line), LRUSet(16)

    def reset_kernel(self, sms):
        for sm in sms:
            sm.l1i.clear()
            sm.l1c.clear()
            for sp in sm.smsp:
                sp.l0i.clear()
                sp.l0c.clear()

    def fetch(self, sp, w, pc, cycle):
        """Cycle at which the instruction at `pc` is available in sp's L0 I-cache."""
        c = self.cfg
        line = (w.cta.kernel.id << 32) | (pc * c.instr_bytes // c.icache_line)
        r = sp.l0i.get(line)
        if r is not None:
            if w.fline != line:                      # the stream buffer keeps running ahead of the fetch
                w.fline = line
                ahead = line + c.ifetch_stream_lines
                if sp.l0i.get(ahead) is None:
                    sp.l0i.put(ahead, self._line_ready(sp.sm, ahead, cycle) + c.ifetch_line_cycles)
            return r
        self.fetch_miss_l0 += 1
        sm = sp.sm
        t = self._line_ready(sm, line, cycle)
        sp.l0i.put(line, t)
        for k in range(1, c.ifetch_stream_lines + 1):        # stream buffer: next lines follow
            if sp.l0i.get(line + k) is None:
                sp.l0i.put(line + k, max(t, self._line_ready(sm, line + k, cycle)) + k * c.ifetch_line_cycles)
        return t

    def _line_ready(self, sm, line, cycle):
        c = self.cfg
        r = sm.l1i.get(line)
        if r is not None:
            return max(r, cycle + c.lat_l1i)
        self.fetch_miss_l1 += 1
        r = self.l2_code.get(line)
        if r is not None:
            t = max(r, cycle + c.lat_l2)
        else:
            self.fetch_miss_l2 += 1
            t = cycle + c.lat_hbm
            self.l2_code[line] = t
        sm.l1i.put(line, t)
        return t

    def warm_code(self, nlines):
        """Pretend a previous launch left the kernel's code in L2."""
        for i in range(nlines):
            self.l2_code[i] = 0

    def ldc(self, sp, key, cycle):
        """Result latency of an LDC of constant `key` (a 64 B line). Kernel parameters were just
        written by the launch, so their lines start out in L2."""
        c = self.cfg
        if not c.model_ccache or sp.l0c.get(key) is not None:
            return c.lat_ldc
        sm = sp.sm
        if sm.l1c.get(key) is not None:
            sp.l0c.put(key, 1)
            return c.lat_ldc + c.lat_ldc_miss
        r = self.l2_const.get(key)
        lat = c.lat_ldc + c.lat_ldc_miss + (c.lat_l2 if r is not None else c.lat_hbm)
        self.l2_const[key] = 1
        sm.l1c.put(key, 1)
        sp.l0c.put(key, 1)
        return lat
