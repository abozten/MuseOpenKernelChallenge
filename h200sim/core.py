"""Cycle-driven GPU core model: warp state machine, 4 sub-partition schedulers per SM,
per-register scoreboards, SIMT divergence stack, async engines (cp.async, TMA, wgmma,
mbarrier), thread-block clusters, and the CTA / cluster block scheduler.

Time skips ahead when nothing can issue. Asynchronous work (cp.async / TMA data landing,
mbarrier tx completion, wgmma results) is applied by timed events, so a program that reads
a destination before waiting on it sees the old contents, as on hardware.
"""

import enum
import heapq
import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import H200
from .isa import Pred, Program, Reg, UReg
from .regs import LANES, _f, _gaddr, _s32, _swz, _u, _wr, _wr64, pbool
from . import control, ops
from . import tensor as T
from .frontend import FrontEnd, LRUSet
from .memory import MemorySystem

INF = 1 << 60


class WS(enum.Enum):
    READY = "ready"              # next instruction can issue
    STALL_DEP = "stall_dep"      # waiting on an ALU/shuffle/tensor result
    STALL_MEM = "stall_mem"      # waiting on a load result
    STALL_PIPE = "stall_pipe"    # functional unit / LSU busy
    STALL_ASYNC = "stall_async"  # mbarrier / cp.async group / wgmma group wait
    STALL_FETCH = "stall_fetch"  # instruction not yet in the L0 I-cache
    BARRIER = "barrier"          # CTA or cluster barrier
    EXITED = "exited"


STALLS = (WS.STALL_DEP, WS.STALL_MEM, WS.STALL_PIPE, WS.STALL_ASYNC, WS.STALL_FETCH)


class MBar:
    """mbarrier: completes a phase when all arrivals are in and the expected tx bytes have landed."""

    def __init__(self, count):
        self.count, self.pending, self.tx, self.phase = count, count, 0, 0
        self.waiters = set()

    def arrive(self, at, n=1, tx=0):
        self.pending -= n
        self.tx += tx
        self._check(at)

    def complete_tx(self, at, nbytes):
        self.tx -= nbytes
        self._check(at)

    def _check(self, at):
        if self.pending == 0 and self.tx == 0:
            self.phase += 1
            self.pending = self.count
            for w in self.waiters:
                w.wake_at = max(at, w.since)
                w.smsp.wake = 0
            self.waiters.clear()


class Warp:
    def __init__(self, cta, wid, smsp, nregs, cycle, reg_limit):
        self.cta, self.wid, self.smsp = cta, wid, smsp
        self.regs = np.zeros((nregs, 32), np.uint32)
        self.ready_at = [0] * 330                       # R0..R255, uniform regs from 256, predicates from 320
        self.mem_dep = [False] * 330
        self.preds = np.zeros((8, 32), bool)
        self.uregs = np.zeros(64, np.uint32)
        self.reg_limit = reg_limit
        self.pc = 0
        self.state = WS.READY
        self.since = cycle
        self.wake_at = cycle
        self.hist = {s: 0 for s in WS}
        self.issued = 0
        # SIMT: active lanes and the reconvergence stack of (pc, mask, rpc)
        self.mask = np.ones(32, bool)
        self.full = True
        self.stack = []
        self.rpc = -1
        # async groups: completion cycles of committed groups, and the open group's max
        self.groups, self.cur_group = [], 0
        self.wg_groups, self.wg_cur = [], 0
        self.bar_kind = ""
        self.stall_until = 0             # control-bit mode: the stall count set by the last instruction
        self.sb = [[] for _ in range(6)] # control-bit mode: pending completion times per dependence counter
        self.fline = -1                  # instruction line last checked by the fetch stream
        self.st_visible = 0              # cycle by which this warp's global stores have all landed in L2
        self.acq_until = 0               # an acquire load is outstanding until this cycle
        self.st_floor = 0                # stores issued from now on become visible no earlier than this (release / fence order)

    def go(self, st, at, sim):
        self.hist[self.state] += at - self.since
        if sim.trace_sms and self.cta.sm.id in sim.trace_sms and len(sim.events) < 200000:
            sim.events.append((at, self.cta.sm.id, self.smsp.idx, self.cta.id, self.wid,
                               self.state.value, st.value, self.pc))
        self.state, self.since = st, at

    def set_mask(self, mask):
        self.mask = mask
        self.full = bool(mask.all())

    def loads_done(self):
        """Cycle by which every global load this warp issued has returned."""
        return max([self.ready_at[r] for r in range(len(self.mem_dep)) if self.mem_dep[r]], default=0)


class Cluster:
    def __init__(self, ctas=None):
        self.ctas = ctas or []
        self.at_bar = 0
        self.waiters = []

    def live(self):
        return sum(c.live for c in self.ctas)


class CTA:
    def __init__(self, cid, rank, sm, nwarps, smem_bytes, cluster):
        self.id, self.rank, self.sm, self.cluster = cid, rank, sm, cluster
        self.warps: list[Warp] = []
        self.smem = np.zeros((max(smem_bytes, 4) + 3) // 4 * 4, np.uint8)
        self.live = nwarps
        self.at_bar = 0
        self.mbar: dict[int, MBar] = {}
        self.kernel = None
        self.triggered = False          # executed griddepcontrol.launch_dependents (or exited)
        self.generic_dirty = False      # shared memory written by the generic proxy since the last fence.proxy
        self.bars = {}                  # named barriers: id -> [arrived threads, waiting warps]
        self.reg_pool = 0               # registers released by setmaxnreg.dec, in per-warp units (x32 threads)
        self.reg_waiters = set()


class SMSP:
    def __init__(self, sm, idx):
        self.sm, self.idx = sm, idx
        self.ctl = sm.sim.dep_mode == "control"      # compiler control bits instead of a scoreboard
        self.last_leave = 0
        self.warps: list[Warp] = []
        self.pipe_free = {"alu": 0, "int": 0, "fp32": 0, "xu": 0, "tensor": 0, "tma": 0, "fp64": 0, "uni": 0}
        self.last = None
        self.wake = 0
        self.issue_slots = 0
        self.mem_q = []            # cycles at which queued memory instructions leave the sub-core queue
        self.rf_free = 0           # register-file read-port bubbles block issue until here
        self.rfc = {}              # (bank, operand slot) -> (warp, register): register file cache
        self.l0i, self.l0c = sm.sim.frontend.new_smsp()

    def check(self, w, ins, cycle):
        t, mem = 0, False
        if self.ctl:
            if w.stall_until > cycle:
                return WS.STALL_DEP, w.stall_until
            for c in ins.wait:
                pend = w.sb[c]
                if pend:
                    tc = max(pend)
                    if tc > cycle and tc > t:
                        t, mem = tc, True
        else:
            ra = w.ready_at
            for r in ins.deps:
                if ra[r] > t:
                    t, mem = ra[r], w.mem_dep[r]
        if t > cycle:
            return WS.STALL_MEM if mem else WS.STALL_DEP, t
        op = ins.op
        if ins.maxreg > w.reg_limit:
            raise RuntimeError(f"pc={w.pc}: {ins} uses R{ins.maxreg - 1}, beyond the {w.reg_limit} registers "
                               f"this warp holds (setmaxnreg)")
        if self.rf_free > cycle and ins.srcs:
            return WS.STALL_PIPE, self.rf_free
        if op == "SETMAXNREG" and ins.extra:
            need = (ins.imm - w.reg_limit) * 32
            if need > w.cta.reg_pool:
                w.cta.reg_waiters.add(w)
                return WS.STALL_ASYNC, INF
        if op == "MBAR_WAIT":
            mb = w.cta.mbar[_s32(w, ins.a)]
            if (mb.phase & 1) == (_s32(w, ins.b) & 1):     # phase with that parity not complete yet
                mb.waiters.add(w)
                return WS.STALL_ASYNC, INF
        elif op == "EXIT":                       # registers are not released while loads are still returning
            t = max([w.ready_at[r] for r in range(len(w.mem_dep)) if w.mem_dep[r]], default=0)
            if t > cycle:
                return WS.STALL_MEM, t
        elif op == "MEMBAR":
            if ins.opt != "cta" and ins.sem != "acq_rel":    # fence.sc: prior stores landed, prior loads returned
                t = max(w.st_visible, w.loads_done())
                if t > cycle:
                    return WS.STALL_ASYNC, t
        elif op == "GRIDDEP_WAIT":
            pred = w.cta.kernel.after
            if pred is not None and not pred.complete:
                pred.waiters.add(w)
                return WS.STALL_ASYNC, INF
        elif op in ("CP_WAIT", "WGMMA_WAIT"):
            groups = w.groups if op == "CP_WAIT" else w.wg_groups
            need = len(groups) - ins.imm
            if need > 0:
                t = max(groups[:need])
                if t > cycle:
                    return WS.STALL_ASYNC, t
        if ins.pipe == "lsu":
            if w.acq_until > cycle:                     # nothing passes an outstanding acquire
                return WS.STALL_MEM, w.acq_until
            depth = self.sm.sim.cfg.mem_queue_depth
            live = [x for x in self.mem_q if x > cycle]
            if len(live) >= depth:                       # queue full: wait for the oldest of the last `depth`
                return WS.STALL_PIPE, live[len(live) - depth]
            pf = 0
        else:
            pf = self.pipe_free.get(ins.pipe, 0)
        if pf > cycle:
            return WS.STALL_PIPE, math.ceil(pf)
        return None, 0

    def _regfile_read(self, w, ins, cycle, cfg):
        """Two banks, one read port each: operands in the same bank read one per cycle. The
        compiler-managed register file cache (one entry per bank and operand slot) lets a
        reused operand skip the port. Same-bank operands cost bubbles for the next issue."""
        need = [0] * cfg.rf_banks
        for slot, reg in ins.srcs:
            key = (reg % cfg.rf_banks, slot)
            if self.rfc.get(key) != (w, reg):
                need[key[0]] += 1
            self.rfc.pop(key, None)                    # a read request to this slot invalidates it
            if slot in ins.reuse:
                self.rfc[key] = (w, reg)
        worst = max(need)
        if worst > 1:
            self.rf_free = cycle + worst

    def step(self, cycle, sim):
        """Try to issue one instruction. Warps with wake_at > cycle are known-stalled and skipped."""
        # compiler-guided greedy-then-youngest: stay on the last warp, else the youngest ready one
        order = reversed(self.warps) if self.last is None else \
            [self.last] + [w for w in reversed(self.warps) if w is not self.last]
        issued = False
        for w in order:
            if w.wake_at > cycle or w.state is WS.EXITED or w.state is WS.BARRIER:
                continue
            if w.state in STALLS and w.since < w.wake_at:  # stall ended unobserved: ready since wake_at
                w.go(WS.READY, w.wake_at, sim)
            while w.stack and w.pc == w.rpc:               # reached the reconvergence point
                w.pc, mask, w.rpc = w.stack.pop()
                w.set_mask(mask)
            ins = w.cta.kernel.instrs[w.pc]
            if sim.cfg.model_icache:
                t = sim.frontend.fetch(self, w, w.pc, cycle)
                if t > cycle:
                    if w.state is not WS.STALL_FETCH:
                        w.go(WS.STALL_FETCH, cycle, sim)
                    w.wake_at = t
                    continue
            st, t = self.check(w, ins, cycle)
            if st is not None:
                if w.state is not st:
                    w.go(st, cycle, sim)
                w.wake_at = t
                continue
            if w.state is not WS.READY:
                w.go(WS.READY, cycle, sim)
            if ins.srcs and sim.cfg.model_regfile:
                self._regfile_read(w, ins, cycle, sim.cfg)
            self.sm.execute(w, ins, cycle, sim)
            w.issued += 1
            self.last = w
            self.issue_slots += 1
            issued = True
            break
        nxt = INF
        for w in self.warps:
            if w.state is not WS.EXITED and w.state is not WS.BARRIER and w.wake_at < nxt:
                nxt = w.wake_at
        self.wake = max(cycle + 1, nxt) if nxt < INF else INF
        return issued


class SM:
    def __init__(self, sid, sim, gpc):
        self.id, self.sim, self.gpc = sid, sim, gpc
        cfg = sim.cfg
        self.part = gpc // 4          # NUMA node: GPCs 0-3 / 4-7
        self.sub = gpc // 2           # sub-NUMA node: GPC pairs
        self.smsp = [SMSP(self, i) for i in range(cfg.smsp_per_sm)]
        self.l1 = sim.memsys.make_l1()
        self.l1i, self.l1c = sim.frontend.new_sm()
        self.tlb = sim.tpc_tlbs[sid // 2]      # the L1 TLB is shared by the two SMs of a TPC
        self.pending = []                 # stores issued by this SM that have not landed in L2 yet
        self.port_free = 0.0
        self.lsu_free = 0
        self.dsmem_free = 0.0
        self.smem_free = 0.0
        self.mem_accept = 0
        self.ctas: list[CTA] = []
        self.regs_used = self.warps_used = self.smem_used = 0
        self.issued = 0

    # ---- occupancy ----
    def fits(self, nwarps, regs, smem):
        c = self.sim.cfg
        return (len(self.ctas) < c.max_ctas_per_sm and self.warps_used + nwarps <= c.max_warps_per_sm
                and self.regs_used + regs <= c.regs_per_sm and self.smem_used + smem <= c.smem_per_sm)

    def launch_cta(self, kernel, cid, rank, cluster, cycle):
        nw = kernel.warps_per_cta
        cta = CTA(cid, rank, self, nw, kernel.smem_bytes, cluster)
        cta.kernel = kernel
        cluster.ctas.append(cta)
        for i in range(nw):
            sp = self.smsp[i % len(self.smsp)]
            w = Warp(cta, i, sp, kernel.prog.nregs, cycle, kernel.prog.launch_regs)
            cta.warps.append(w)
            sp.warps.append(w)
            sp.wake = 0
        self.ctas.append(cta)
        self.warps_used += nw
        self.regs_used += kernel.regs_per_cta
        self.smem_used += kernel.smem_bytes
        kernel.resident += 1

    def retire_cta(self, cta):
        k = cta.kernel
        for w in cta.warps:
            w.smsp.warps.remove(w)
        self.ctas.remove(cta)
        self.warps_used -= len(cta.warps)
        self.regs_used -= k.regs_per_cta
        self.smem_used -= k.smem_bytes
        self.sim.retired.extend(cta.warps)
        k.resident -= 1
        k.ctas_done += 1

    def wake_all(self):
        for sp in self.smsp:
            sp.wake = 0

    # ---- execution ----
    def execute(self, w, ins, cycle, sim):
        op = ins.op
        w.wake_at = cycle + 1
        sp = w.smsp
        if ins.pipe in sp.pipe_free:
            sp.pipe_free[ins.pipe] = max(sp.pipe_free[ins.pipe], cycle) + ins.occ
            if ins.pipe == "tensor":
                w.cta.kernel.tc_busy += ins.occ
        self.issued += 1
        w.cta.kernel.op_counts[op] += 1
        w.cta.kernel.issued += 1
        sim.now = cycle
        if sim.log_issue:
            sim.issue_log.append((cycle, self.id, sp.idx, w.cta.id, w.wid, w.pc, op))
        w.pc += 1
        done = cycle + math.ceil(ins.lat)
        mem = False
        if ins.pipe == "lsu":
            self._enqueue_mem(sp, ins, cycle, sim.cfg)

        run, restore = True, None
        if ins.guard is not None:                       # @P / @!P: run on the lanes where the predicate holds
            g = pbool(w, Pred(ins.guard[0]))
            eff = w.mask & (~g if ins.guard[1] else g)
            if op == "BRA":
                run = False
                self._branch_lanes(w, ins, eff, w.mask & ~eff)
                w.wake_at = cycle + 2
            elif op == "EXIT":
                run = False
                self._exit_lanes(w, eff, cycle, sim)
            elif not eff.any():
                run = False                              # predicated off: slot and pipe are still spent
            else:
                restore = (w.mask, w.full)
                w.set_mask(eff)
        mem = False
        if run:
            done, mem = self._body(w, ins, cycle, sim, sp, done)
        if restore is not None:
            w.mask, w.full = restore
        if isinstance(ins.a, UReg):                      # uniform-register addresses resolve sooner
            if op == "LDG":
                done -= 3
            elif op == "LDS":
                done -= 1
        for r in ins.writes:
            w.ready_at[r] = done
            w.mem_dep[r] = mem
        if sp.ctl:                                         # set this instruction's control bits in motion
            w.stall_until = cycle + (max(ins.stall, 2) if ins.yield_ else ins.stall)
            if ins.inc_raw is not None:
                w.sb[ins.inc_raw] = [x for x in w.sb[ins.inc_raw] if x > cycle] + [done]
            if ins.inc_war is not None:
                war = sp.last_leave if ins.pipe == "lsu" else cycle + 4
                w.sb[ins.inc_war] = [x for x in w.sb[ins.inc_war] if x > cycle] + [war]

    def _body(self, w, ins, cycle, sim, sp, done):
        op = ins.op
        mem = False

        if op == "FFMA":
            _wr(w, ins.d, (_f(w, ins.a) * _f(w, ins.b) + _f(w, ins.c)).view(np.uint32))
        elif op == "MOV":
            _wr(w, ins.d, _u(w, ins.a))
        elif op == "S2R":
            _wr(w, ins.d, self._special(w, ins.name))
        elif op == "LDC":
            k = w.cta.kernel
            _wr(w, ins.d, np.full(32, k.params[ins.name] & 0xFFFFFFFF, np.uint32))
            done = cycle + sim.frontend.ldc(sp, ("p", k.id, k.param_line(ins.name)), cycle)
            mem = True
        elif op == "IADD":
            _wr(w, ins.d, _u(w, ins.a) + _u(w, ins.b))
        elif op == "IMAD":
            _wr(w, ins.d, _u(w, ins.a) * _u(w, ins.b) + _u(w, ins.c))
        elif op == "SHL":
            _wr(w, ins.d, _u(w, ins.a) << _u(w, ins.b))
        elif op == "SHR":
            _wr(w, ins.d, _u(w, ins.a) >> _u(w, ins.b))
        elif op == "AND":
            _wr(w, ins.d, _u(w, ins.a) & _u(w, ins.b))
        elif op == "OR":
            _wr(w, ins.d, _u(w, ins.a) | _u(w, ins.b))
        elif op == "XOR":
            _wr(w, ins.d, _u(w, ins.a) ^ _u(w, ins.b))
        elif op in ("SETP_LT", "SETP_EQ", "SETP_GE"):
            a, b = _u(w, ins.a).view(np.int32), _u(w, ins.b).view(np.int32)
            r = a < b if op == "SETP_LT" else (a == b if op == "SETP_EQ" else a >= b)
            _wr(w, ins.d, r.astype(np.uint32))
        elif op == "FADD":
            _wr(w, ins.d, (_f(w, ins.a) + _f(w, ins.b)).view(np.uint32))
        elif op == "FMUL":
            _wr(w, ins.d, (_f(w, ins.a) * _f(w, ins.b)).view(np.uint32))
        elif op == "SHFL_XOR":
            _wr(w, ins.d, _u(w, ins.a)[LANES ^ int(ins.b)])
        elif op == "HMMA":
            self._hmma(w, ins)
        elif op in ("LDG", "STG", "RED_ADD_F32"):
            done, mem = self._global(w, ins, cycle, sim)
        elif op in ("LDS", "STS"):
            done, mem = self._shared(w, ins, cycle)
        elif op == "BRA":
            w.pc = ins.target
            w.wake_at = cycle + 2
        elif op == "SETMAXNREG":
            cta = w.cta
            if ins.extra:
                cta.reg_pool -= (ins.imm - w.reg_limit) * 32
            else:
                cta.reg_pool += (w.reg_limit - ins.imm) * 32
                for x in cta.reg_waiters:
                    x.wake_at, x.smsp.wake = cycle, 0
                cta.reg_waiters.clear()
            w.reg_limit = ins.imm
        elif op == "BAR_ARRIVE":
            self._named_bar(w, ins, cycle, sim, wait=False)
        elif op == "GRIDDEP_LAUNCH":
            sim.trigger_cta(w.cta, cycle)
        elif op == "GRIDDEP_WAIT":
            pass
        elif op == "BRNZ":
            self._branch(w, ins)
            w.wake_at = cycle + 2
        elif op == "BAR":
            if ins.imm or ins.extra:
                self._named_bar(w, ins, cycle, sim, wait=True)
            else:
                cta = w.cta
                cta.at_bar += 1
                w.bar_kind = "cta"
                w.go(WS.BARRIER, cycle + 1, sim)
                w.wake_at = INF
                self._try_release(cta, cycle, sim)
        elif op == "LDC64":
            k = w.cta.kernel
            done = cycle + sim.frontend.ldc(sp, ("p", k.id, k.param_line(ins.name)), cycle)
            mem = True
            v = k.params[ins.name]
            _wr(w, ins.d, np.full(32, v & 0xFFFFFFFF, np.uint32))
            _wr(w, ins.d + 1, np.full(32, (v >> 32) & 0xFFFFFFFF, np.uint32))
        elif op == "IADD64":
            a64 = w.regs[ins.a].astype(np.uint64) | (w.regs[ins.a + 1].astype(np.uint64) << np.uint64(32))
            _wr64(w, ins.d, a64 + _u(w, ins.b).astype(np.uint64))
        elif op == "IMAD_WIDE":
            r = _u(w, ins.a).astype(np.uint64) * _u(w, ins.b).astype(np.uint64)
            if ins.c is not None:
                r = r + (w.regs[ins.c].astype(np.uint64) | (w.regs[ins.c + 1].astype(np.uint64) << np.uint64(32)))
            _wr64(w, ins.d, r)
        elif op == "EXIT":
            self._exit_lanes(w, w.mask.copy(), cycle, sim)
        elif op == "LDGSTS":
            self._ldgsts(w, ins, cycle, sim)
        elif op == "CP_COMMIT":
            w.groups.append(w.cur_group)
            w.cur_group = 0
        elif op == "CP_WAIT":
            del w.groups[:max(0, len(w.groups) - ins.imm)]
        elif op == "MBAR_INIT":
            w.cta.mbar[_s32(w, ins.a)] = MBar(ins.imm)
        elif op == "MBAR_ARRIVE":
            w.cta.mbar[_s32(w, ins.a)].arrive(cycle)
        elif op == "MBAR_ARRIVE_TX":
            w.cta.mbar[_s32(w, ins.a)].arrive(cycle, 1, ins.imm)
        elif op == "MBAR_ARRIVE_CL":
            tgt = w.cta.cluster.ctas[_s32(w, ins.b)]
            addr = _s32(w, ins.a)
            sim.at(cycle + sim.cfg.lat_dsmem // 2, lambda t, tgt=tgt, addr=addr: tgt.mbar[addr].arrive(t))
        elif op == "FENCE":
            w.cta.generic_dirty = False
        elif op == "MEMBAR":
            if ins.opt != "cta":
                if ins.sem == "acq_rel":                     # ordering without waiting
                    w.st_floor = max(w.st_floor, w.st_visible, w.loads_done())
                    w.acq_until = max(w.acq_until, w.loads_done())
                self.l1.data.clear()
        elif op == "MBAR_WAIT" or op == "WGMMA_FENCE":
            pass
        elif op in ("TMA_LOAD_1D", "TMA_LOAD_2D", "TMA_LOAD_ND", "TMA_LOAD_IM2COL"):
            self._tma_load(w, ins, cycle, sim)
        elif op == "TMA_STORE_1D":
            self._tma_store(w, ins, cycle, sim)
        elif op in ("TMA_STORE_2D", "TMA_STORE_ND"):
            self._tma_store_2d(w, ins, cycle, sim)
        elif op == "TMA_PREFETCH_2D":
            tm = sim.tmaps[ins.imm]
            sectors = tm.sectors(_s32(w, ins.b), _s32(w, ins.c))
            if len(sectors):
                sim.memsys.load(self, sectors, cycle + sim.cfg.tma_issue_lat, False)
        elif op == "WGMMA":
            self._wgmma(w, ins, cycle, sim)
        elif op == "WGMMA_COMMIT":
            w.wg_groups.append(w.wg_cur)
            w.wg_cur = 0
        elif op == "WGMMA_WAIT":
            del w.wg_groups[:max(0, len(w.wg_groups) - ins.imm)]
        elif op in ("LDS_CL", "STS_CL"):
            done, mem = self._shared_cluster(w, ins, cycle, sim)
        elif op == "CLUSTER_SYNC":
            cl = w.cta.cluster
            cl.at_bar += 1
            cl.waiters.append(w)
            w.bar_kind = "cluster"
            w.go(WS.BARRIER, cycle + 1, sim)
            w.wake_at = INF
            self._try_release_cluster(cl, cycle, sim)
        else:
            fn = ops.EXTRA.get(op)
            if fn is None:
                raise NotImplementedError(op)
            res = fn(self, w, ins, cycle, sim)
            if res is not None:
                done, mem = res
        return done, mem

    def _exit_lanes(self, w, gone, cycle, sim):
        """Lanes in `gone` retire. Surviving active lanes keep going; otherwise the next path parked
        on the reconvergence stack runs; with nothing left the warp exits."""
        w.stack = [(pc, m & ~gone, rpc) for pc, m, rpc in w.stack]
        rest = w.mask & ~gone
        if rest.any():
            w.set_mask(rest)
            return
        nxt = None
        while w.stack:
            pc, m, rpc = w.stack.pop()
            if m.any():
                nxt = (pc, m, rpc)
                break
        if nxt is not None:
            w.pc, mask, w.rpc = nxt
            w.set_mask(mask)
            w.wake_at = cycle + 2
            return
        w.go(WS.EXITED, cycle + 1, sim)
        w.wake_at = INF
        cta = w.cta
        cta.live -= 1
        if cta.live == 0:
            sim.cta_done.append(cta)
        else:
            self._try_release(cta, cycle, sim)

    def _named_bar(self, w, ins, cycle, sim, wait):
        """bar.sync / bar.arrive on barrier ins.imm, released when ins.extra threads (default: the
        whole CTA) have arrived."""
        cta = w.cta
        b = cta.bars.setdefault(ins.imm, [0, []])
        b[0] += 32
        if wait:
            w.bar_kind = "named"
            w.go(WS.BARRIER, cycle + 1, sim)
            w.wake_at = INF
            b[1].append(w)
        if b[0] >= (ins.extra or cta.live * 32):
            for x in b[1]:
                x.go(WS.READY, cycle + 6, sim)
                x.wake_at = cycle + 6
            cta.bars[ins.imm] = [0, []]
            self.wake_all()

    def _enqueue_mem(self, sp, ins, cycle, cfg):
        """Sub-core memory queue: 5 deep, address calc one instruction per 4 cycles, shared
        structures accept one per 2 cycles. An entry leaves once its source registers are read
        (the WAR latency)."""
        op, wd = ins.op, ins.width
        wd = wd if wd in (4, 8, 16) else 4
        uni = isinstance(ins.a, UReg)
        if op == "LDG":
            war = cfg.war_ldg_uniform if uni else cfg.war_ldg
        elif op in ("LDS", "LDS_CL", "LDMATRIX"):
            war = cfg.war_ldg_uniform if uni and op == "LDS" else cfg.war_lds
        elif op in ("STS", "STS_CL", "ATOMS"):
            war = dict(cfg.war_sts_uniform if uni else cfg.war_sts)[wd]
        elif op == "STMATRIX":
            war = dict(cfg.war_sts)[16]
        elif op == "LDGSTS":
            war = cfg.war_ldgsts
        else:                                            # STG, RED, ATOM
            war = dict(cfg.war_stg_uniform if uni else cfg.war_stg)[wd]
        leave = cycle + war
        if sp.mem_q:
            leave = max(leave, sp.mem_q[-1] + cfg.smsp_mem_addr_cycles)
        leave = max(leave, self.mem_accept)
        self.mem_accept = leave + cfg.sm_mem_accept_cycles
        sp.mem_q = [x for x in sp.mem_q if x > cycle]
        sp.mem_q.append(leave)
        sp.last_leave = leave

    def _occupy(self, w, cycles):
        """The shared L1/shared-memory stage is busy for `cycles` with this instruction's data
        (128 B/cycle), delaying whatever the sub-cores send next."""
        self.mem_accept = max(self.mem_accept, w.smsp.mem_q[-1] + cycles)

    # ---- barriers ----
    def _try_release(self, cta, cycle, sim):
        if cta.at_bar and cta.at_bar >= cta.live:
            for x in cta.warps:
                if x.state is WS.BARRIER and x.bar_kind == "cta":
                    x.go(WS.READY, cycle + 6, sim)
                    x.wake_at = cycle + 6
            cta.at_bar = 0
            self.wake_all()
        if cta.cluster.at_bar:
            self._try_release_cluster(cta.cluster, cycle, sim)

    def _try_release_cluster(self, cl, cycle, sim):
        if cl.at_bar and cl.at_bar >= cl.live():
            for x in cl.waiters:
                x.go(WS.READY, cycle + 10, sim)
                x.wake_at = cycle + 10
            for c in cl.ctas:
                c.sm.wake_all()
            cl.waiters.clear()
            cl.at_bar = 0

    def _special(self, w, name):
        if name == "laneid":
            return LANES.astype(np.uint32)
        if name == "tid":
            return (w.wid * 32 + LANES).astype(np.uint32)
        sim = self.sim
        if name == "clock":
            return np.full(32, self.sim.now & 0xFFFFFFFF, np.uint32)
        k = w.cta.kernel
        vals = {"warpid": w.wid, "ctaid": w.cta.id, "ntid": k.block, "smid": self.id,
                "nctaid": k.grid, "clusterrank": w.cta.rank, "clustersize": k.cluster_size,
                "clusterid": w.cta.id // k.cluster_size}
        return np.full(32, vals[name], np.uint32)

    # ---- control flow ----
    def _branch(self, w, ins):
        p = _u(w, ins.a) != 0
        self._branch_lanes(w, ins, p & w.mask, ~p & w.mask)

    def _branch_lanes(self, w, ins, taken, fall):
        if not fall.any():
            w.pc = ins.target
        elif taken.any():
            if ins.reconv is None:
                raise RuntimeError(f"divergent branch at pc={w.pc - 1} needs a reconv label")
            w.stack.append((ins.reconv, w.mask.copy(), w.rpc))   # join point restores the full mask
            w.stack.append((ins.target, taken, ins.reconv))      # taken path runs after the fall-through
            w.rpc = ins.reconv
            w.set_mask(fall)
        # else: nobody takes it, fall through

    # ---- memory ----
    def _addr(self, w, ins):
        return _u(w, ins.a).astype(np.int64) + ins.imm

    def _gwarp(self, w, ins):
        return _gaddr(w, ins.a, ins.wide) + ins.imm

    def _global(self, w, ins, cycle, sim):
        ms, mem, cfg = sim.memsys, sim.memsys.mem, sim.cfg
        act = w.mask
        addr = self._gwarp(w, ins)
        a = addr if w.full else addr[act]
        nw = ins.width // 4
        if a.size == 0:
            return cycle + 1, False
        off = a - mem.base
        if (a % ins.width).any() or not mem.valid(int(off.min()), int(off.max()) + ins.width):
            raise RuntimeError(f"illegal/misaligned global access at pc={w.pc - 1} ({ins}): "
                               f"addr range [{a.min():#x}, {a.max():#x}]")
        idx = (off >> 2)[:, None] + np.arange(nw)
        sectors = np.unique(a >> 5)
        self._occupy(w, max(1, math.ceil(len(sectors) / cfg.lsu_sectors_per_cycle)))
        if ins.op == "LDG":
            pf = int(ins.opt[2:]) if ins.opt.startswith("pf") else 0
            use_l1 = ins.cache == "ca" and ins.sem in ("", "weak")      # cg / cs / cv / relaxed / acquire go to L2
            t = ms.load(self, sectors, cycle, use_l1, ins.width, pf)
            hits, fills = list(ms.last_l1_hits), list(ms.last_l1_fills)
            full, d = w.full, ins.d

            def read(_):
                """The load reads memory where its request lands: later than issue, earlier than the data returns."""
                data = mem.u32[idx]
                if cfg.model_visibility and self.pending:    # this SM's own stores still in flight
                    data = self._forward_stores(data, idx)
                if use_l1 and cfg.model_l1_staleness:        # hits return what the L1 captured earlier
                    data = self._l1_snapshots(data, idx, hits, fills, mem)
                if full:
                    w.regs[d:d + nw] = data.T
                else:
                    w.regs[d:d + nw, act] = data.T
            if cfg.model_load_read_time:
                sim.at(max(cycle + 1, min(ms.last_read_t, t - 1)), read)
            else:
                read(None)
            if ins.sem == "acquire":
                self.l1.data.clear()                         # acquire invalidates this SM's L1
                w.acq_until = t
            return t, True
        if ins.op == "STG":
            vals = (w.regs[ins.b:ins.b + nw] if w.full else w.regs[ins.b:ins.b + nw, act]).T.copy()
            if ins.sem == "release":                          # st.release: ordered after earlier stores and loads
                w.st_floor = max(w.st_floor, w.st_visible, w.loads_done())
            t = ms.store(self, sectors, cycle)
            if cfg.model_l1_staleness:
                self._l1_write(idx, vals, mem)               # write-through updates this SM's L1 lines
            if cfg.model_visibility:
                self._defer_store(w, idx, vals, t, sim)      # other SMs see it when it lands in L2
            else:
                mem.u32[idx] = vals
            return cycle + 1, False
        vals = w.regs[ins.b] if w.full else w.regs[ins.b][act]      # RED.ADD.F32: executes in L2 at once
        np.add.at(mem.u32.view(np.float32), off >> 2, vals.view(np.float32))
        ms.store(self, sectors, cycle)
        return cycle + 1, False

    # ---- memory ordering ----
    def _defer_store(self, w, idx, vals, t, sim):
        mem = sim.memsys.mem
        t = max(t, w.st_floor)                                # ordered after what a release / fence covers
        words, v = idx.ravel().copy(), vals.ravel().copy()
        entry = (words, v, {int(a): int(b) for a, b in zip(words, v)})
        self.pending.append(entry)
        w.st_visible = max(w.st_visible, t)
        w.cta.kernel.last_store_t = max(w.cta.kernel.last_store_t, t)

        def land(_):
            mem.u32[words] = v
            self.pending[:] = [e for e in self.pending if e is not entry]
        sim.at(t, land)

    def _forward_stores(self, data, idx):
        flat, out = idx.ravel(), data.ravel().copy()
        for words, _, lookup in self.pending:
            for j in np.nonzero(np.isin(flat, words))[0]:
                out[j] = lookup[int(flat[j])]
        return out.reshape(data.shape)

    @staticmethod
    def _sector_of(word_idx, mem):
        return (mem.base + word_idx * 4) >> 5

    def _l1_snapshots(self, data, idx, hits, fills, mem):
        """Hits in the (non-coherent) L1 return the sector's contents as captured when it was filled;
        newly filled sectors capture the current contents."""
        l1 = self.l1
        flat = idx.ravel()
        sec = self._sector_of(flat, mem)
        out = None
        for s in hits:
            snap = l1.data.get(s)
            if snap is None:
                continue
            sel = np.nonzero(sec == s)[0]
            if sel.size:
                if out is None:
                    out = data.ravel().copy()
                out[sel] = snap[flat[sel] & 7]
        for s in fills:
            o = (s * 32 - mem.base) >> 2
            if 0 <= o and o + 8 <= mem.u32.size:
                l1.data[s] = mem.u32[o:o + 8].copy()
        return data if out is None else out.reshape(data.shape)

    def _l1_write(self, idx, vals, mem):
        if not self.l1.data:
            return
        flat, v = idx.ravel(), vals.ravel()
        sec = self._sector_of(flat, mem)
        for s in np.unique(sec):
            snap = self.l1.data.get(int(s))
            if snap is not None:
                sel = sec == s
                snap[flat[sel] & 7] = v[sel]

    def _bank_waves(self, addr, act, nw):
        group = 32 if nw == 1 else (16 if nw == 2 else 8)
        waves = 0
        for g in range(0, 32, group):
            sel = addr[g:g + group][act[g:g + group]]
            if sel.size == 0:
                continue
            words = np.unique((sel >> 2)[:, None] + np.arange(nw))
            waves += np.bincount(words % self.sim.cfg.smem_banks, minlength=self.sim.cfg.smem_banks).max()
        return max(1, int(waves))

    def _shared(self, w, ins, cycle):
        cfg = self.sim.cfg
        smem = w.cta.smem
        s32 = smem.view(np.uint32)
        act = w.mask
        addr = self._addr(w, ins)
        nw = ins.width // 4
        a = addr[act]
        if a.size == 0:
            return cycle + 1, False
        if (a % ins.width).any() or a.min() < 0 or a.max() + ins.width > smem.size:
            raise RuntimeError(f"illegal shared access at pc={w.pc - 1} ({ins})")
        idx = (a >> 2)[:, None] + np.arange(nw)
        if ins.op == "LDS":
            w.regs[ins.d:ins.d + nw, act] = s32[idx].T
        else:
            s32[idx] = w.regs[ins.b:ins.b + nw, act].T
            w.cta.generic_dirty = True
        waves = self._bank_waves(addr, act, nw)
        self._occupy(w, waves)
        return cycle + cfg.lat_smem + (2 if nw == 4 else 0) + waves - 1, True

    def _shared_cluster(self, w, ins, cycle, sim):
        """Distributed shared memory: another CTA of the cluster, over the SM-to-SM network."""
        cfg = sim.cfg
        tgt = w.cta.cluster.ctas[_s32(w, ins.c)]
        s32 = tgt.smem.view(np.uint32)
        act = w.mask
        addr = self._addr(w, ins)
        nw = ins.width // 4
        a = addr[act]
        if a.size == 0:
            return cycle + 1, False
        if (a % ins.width).any() or a.min() < 0 or a.max() + ins.width > tgt.smem.size:
            raise RuntimeError(f"illegal cluster shared access at pc={w.pc - 1} ({ins})")
        idx = (a >> 2)[:, None] + np.arange(nw)
        nbytes = a.size * ins.width
        self._occupy(w, max(1, nbytes // 128))
        if tgt is w.cta:                                 # resolves to this CTA's own shared memory
            if ins.op == "LDS_CL":
                w.regs[ins.d:ins.d + nw, act] = s32[idx].T
                return cycle + cfg.lat_dsmem_local, True
            s32[idx] = w.regs[ins.b:ins.b + nw, act].T
            return cycle + 1, False
        csz = len(w.cta.cluster.ctas)
        bw = dict(cfg.dsmem_bytes_per_cycle).get(csz) or \
            min(cfg.dsmem_bytes_per_cycle, key=lambda kv: abs(kv[0] - csz))[1]
        self.dsmem_free = max(self.dsmem_free, cycle) + nbytes / bw
        if ins.op == "LDS_CL":
            w.regs[ins.d:ins.d + nw, act] = s32[idx].T
            return int(self.dsmem_free) + cfg.lat_dsmem, True
        vals = w.regs[ins.b:ins.b + nw, act].T.copy()

        def land(t, s32=s32, idx=idx, vals=vals):
            s32[idx] = vals
        sim.at(int(self.dsmem_free) + cfg.lat_dsmem // 2, land)
        return cycle + 1, False

    # ---- async copy engines ----
    def _ldgsts(self, w, ins, cycle, sim):
        ms, mem, cfg = sim.memsys, sim.memsys.mem, sim.cfg
        act = w.mask
        saddr = _u(w, ins.a).astype(np.int64)[act]
        gaddr = _gaddr(w, ins.b, ins.wide)[act]
        size, nw = ins.width, ins.width // 4
        if gaddr.size == 0:
            return
        smem = w.cta.smem
        goff = gaddr - mem.base
        if ((gaddr % size) | (saddr % size)).any() or not mem.valid(int(goff.min()), int(goff.max()) + size) \
                or saddr.min() < 0 or saddr.max() + size > smem.size:
            raise RuntimeError(f"illegal cp.async at pc={w.pc - 1}")
        data = mem.u32[(goff >> 2)[:, None] + np.arange(nw)].copy()
        sectors = np.unique(gaddr >> 5)
        self._occupy(w, max(1, math.ceil(len(sectors) / cfg.lsu_sectors_per_cycle)))
        t = ms.load(self, sectors, cycle, ins.cache == "ca", size)
        s32 = smem.view(np.uint32)
        sidx = (saddr >> 2)[:, None] + np.arange(nw)

        def land(_, s32=s32, sidx=sidx, data=data):
            s32[sidx] = data
        sim.at(t, land)
        w.cur_group = max(w.cur_group, t)

    def _tma_deliver(self, w, cycle, sim, smem_addr, mbar_addr, payload, t, mcast, swizzle=0):
        """Land `payload` bytes at smem_addr (and complete the tx on mbar_addr) in every target CTA."""
        cl = w.cta.cluster
        targets = [cl.ctas[r] for r in range(len(cl.ctas)) if (mcast >> r) & 1] if mcast else [w.cta]
        if mcast:
            t += sim.cfg.lat_dsmem // 2

        dst = _swz(smem_addr + np.arange(payload.size), swizzle) if swizzle else None

        def land(at, targets=targets):
            for c in targets:
                if dst is None:
                    c.smem[smem_addr:smem_addr + payload.size] = payload
                else:
                    c.smem[dst] = payload
                c.sm.smem_free = max(c.sm.smem_free, at) + payload.size / sim.cfg.smem_bytes_per_cycle
                c.mbar[mbar_addr].complete_tx(at, payload.size)
        sim.at(t, land)

    def _tma_load(self, w, ins, cycle, sim):
        ms, mem = sim.memsys, sim.memsys.mem
        smem_addr = _s32(w, ins.a)
        swizzle = 0
        if ins.op == "TMA_LOAD_1D":
            g, n = int(_gaddr(w, ins.b, ins.wide)[0]), ins.imm
            off = g - mem.base
            if not mem.valid(off, off + n) or smem_addr < 0 or smem_addr + n > w.cta.smem.size:
                raise RuntimeError(f"illegal TMA 1D load at pc={w.pc - 1}")
            payload = mem.bytes[off:off + n].copy()
            sectors = np.arange(g >> 5, ((g + n - 1) >> 5) + 1)
        else:
            tm = sim.tmaps[ins.imm]
            coords = [_s32(w, c) for c in ins.coords] if ins.coords else [_s32(w, ins.b), _s32(w, ins.c)]
            if ins.op == "TMA_LOAD_IM2COL" and ins.opt:
                payload, sectors = tm.gather_w(mem.bytes, mem.base, coords, ins.opt, ins.lda, ins.ldb)
            elif ins.op == "TMA_LOAD_IM2COL":
                payload, sectors = tm.gather(mem.bytes, mem.base, coords, [_s32(w, o) for o in ins.offs])
            else:
                payload, sectors = tm.gather(mem.bytes, mem.base, *coords)
            swizzle = tm.swizzle
            if smem_addr < 0 or smem_addr + payload.size > w.cta.smem.size:
                raise RuntimeError(f"TMA tile does not fit shared memory at pc={w.pc - 1}")
            if swizzle and smem_addr % 1024:
                raise RuntimeError(f"swizzled TMA destination must be 1024 B aligned (pc={w.pc - 1})")
        t = ms.load(self, sectors, cycle + sim.cfg.tma_issue_lat, False) if len(sectors) else \
            cycle + sim.cfg.tma_issue_lat
        self._tma_deliver(w, cycle, sim, smem_addr, _s32(w, ins.d), payload, t, ins.extra, swizzle)

    def _tma_store(self, w, ins, cycle, sim):
        sim.async_read_smem(w, ins)
        ms, mem = sim.memsys, sim.memsys.mem
        g, s, n = int(_gaddr(w, ins.a, ins.wide)[0]), _s32(w, ins.b), ins.imm
        off = g - mem.base
        if not mem.valid(off, off + n) or s < 0 or s + n > w.cta.smem.size:
            raise RuntimeError(f"illegal TMA store at pc={w.pc - 1}")
        payload = w.cta.smem[s:s + n].copy()
        sectors = np.arange(g >> 5, ((g + n - 1) >> 5) + 1)
        t = ms.store(self, sectors, cycle + sim.cfg.tma_issue_lat)

        def land(_, off=off, payload=payload):
            mem.bytes[off:off + payload.size] = payload
        sim.at(t, land)
        w.cur_group = max(w.cur_group, t)

    def _tma_store_2d(self, w, ins, cycle, sim):
        sim.async_read_smem(w, ins)
        ms, mem = sim.memsys, sim.memsys.mem
        tm = sim.tmaps[ins.imm]
        s = _s32(w, ins.a)
        coords = [_s32(w, c) for c in ins.coords] if ins.coords else [_s32(w, ins.b), _s32(w, ins.c)]
        n = tm.nbytes
        if s < 0 or s + n > w.cta.smem.size or (tm.swizzle and s % 1024):
            raise RuntimeError(f"illegal TMA store source at pc={w.pc - 1}")
        payload = w.cta.smem[_swz(s + np.arange(n), tm.swizzle)].copy()
        sectors = tm.sectors(*coords)
        t = ms.store(self, sectors, cycle + sim.cfg.tma_issue_lat) if len(sectors) else cycle + 1
        if ins.extra and tm.eb != 4:
            raise RuntimeError("TMA reduce-add is modeled for fp32 tensors")
        sim.at(t, lambda _: tm.scatter(mem, payload, coords, bool(ins.extra)))
        w.cur_group = max(w.cur_group, t)

    # ---- tensor cores ----
    def _read_tile(self, cta, base, ld, rows, cols, eb, swz, dt, mn=False):
        """Decode a rows x cols tile of elements from shared memory. K-major: row stride ld bytes,
        elements of a row contiguous. MN-major (`mn`): the tile is stored transposed, each of the
        `cols` K positions a stride-ld line holding `rows` contiguous elements."""
        if mn:
            addr = base + np.arange(cols)[None, :] * ld + np.arange(rows)[:, None] * eb
        else:
            addr = base + np.arange(rows)[:, None] * ld + np.arange(cols)[None, :] * eb
        addr = _swz(addr, swz)
        view = {1: cta.smem, 2: cta.smem.view(np.uint16), 4: cta.smem.view(np.uint32)}[eb]
        return T.decode(dt, view[addr // eb])

    def _wgmma(self, w, ins, cycle, sim):
        cfg = sim.cfg
        sim.async_read_smem(w, ins)
        n, dt, accn, sparse, rs = ins.imm, ins.dtype, ins.acc, ins.sparse, ins.a_regs
        spec = T.DTYPES[dt]
        eb = spec.bits // 8
        kc = spec.k_wg                               # K of A as stored (compressed K when sparse)
        ke = 2 * kc if sparse else kc                # effective K that B spans
        r0 = 16 * (w.wid % 4)
        b_addr = _s32(w, ins.b)
        # --- timing: tensor pipe busy for the measured number of cycles; random data derates it ---
        occ = T.wgmma_cycles(n, rs, sparse)
        if cfg.tensor_data_random:
            occ /= dict(((a, b), r) for a, b, r in cfg.tensor_derate).get((dt, accn), 1.0)
        smem_bytes = T.wgmma_smem_wavefronts(n, rs, dt, sparse) * 128 // 4      # this warp's quarter of the instruction's reads
        sp = w.smsp
        start = max(cycle, sp.pipe_free["tensor"])
        sp.pipe_free["tensor"] = start + occ
        w.cta.kernel.tc_busy += occ
        self.smem_free = max(self.smem_free, start) + smem_bytes / cfg.smem_bytes_per_cycle
        t = int(math.ceil(max(start + occ, self.smem_free))) + cfg.wgmma_lat
        w.wg_cur = max(w.wg_cur, t)
        cta, d, lda, ldb, acc, swz = w.cta, ins.d, ins.lda, ins.ldb, bool(ins.extra), ins.swz
        a_regs = T.a_fragment(dt, w.regs[ins.a:ins.a + 4], kc) if rs else None
        a_addr = 0 if rs else _s32(w, ins.a)
        meta_regs = w.regs[ins.c].copy() if sparse else None      # per-lane metadata words, read at issue
        a_mn, b_mn = "a" in ins.opt, "b" in ins.opt
        a_end = a_addr + ((kc - 1) * lda + (r0 + 16) * eb if a_mn else (r0 + 15) * lda + kc * eb)
        b_end = b_addr + ((ke - 1) * ldb + n * eb if b_mn else (n - 1) * ldb + ke * eb)
        if not rs and (a_end > cta.smem.size or (a_addr | lda) % eb):
            raise RuntimeError(f"wgmma A outside shared memory / misaligned at pc={w.pc - 1}")
        if b_end > cta.smem.size or (b_addr | ldb) % eb:
            raise RuntimeError(f"wgmma B outside shared memory / misaligned at pc={w.pc - 1}")
        nreg = 2 if accn == "f16" else 4              # accumulator registers per 8-column block

        def land(_):
            Ac = a_regs if rs else (self._read_tile(cta, a_addr + r0 * eb, lda, 16, kc, eb, swz, dt, True) if a_mn
                                    else self._read_tile(cta, a_addr + r0 * lda, lda, 16, kc, eb, swz, dt))
            if sparse:
                Ac = T.decode_sparse_a(dt, Ac, meta_regs, ke, ins.sel)
            Bm = self._read_tile(cta, b_addr, ldb, n, ke, eb, swz, dt, b_mn)  # n x ke
            old = np.zeros((16, n), np.float32 if accn != "s32" else np.int32)
            if acc:
                for j in range(n // 8):
                    old[:, 8 * j:8 * j + 8] = T.c_fragment(w.regs[d + nreg * j:d + nreg * j + nreg], accn)
            if dt in ("e4m3", "e5m2"):
                D = T.fp8_accumulate(Ac, Bm, old)
            elif accn == "s32":
                D = (Ac.astype(np.int64) @ Bm.astype(np.int64).T + old).astype(np.int32)
            else:
                D = Ac.astype(np.float32) @ Bm.astype(np.float32).T + old
            for j in range(n // 8):
                T.store_c_fragment(w.regs[d + nreg * j:d + nreg * j + nreg], D[:, 8 * j:8 * j + 8], accn)
        sim.at(t, land)

    def _hmma(self, w, ins):
        """mma.sync, PTX fragment layouts (see tensor.py)."""
        dt, k, accn = ins.dtype, ins.k, ins.acc
        epr = 32 // T.DTYPES[dt].bits
        ka = k // 2 if ins.sparse else k                 # K of A as stored
        na, nb, nc = ka // (2 * epr), k // (4 * epr), 2 if accn == "f16" else 4
        A = T.a_fragment(dt, w.regs[ins.a:ins.a + na], ka)
        if ins.sparse:
            A = T.decode_sparse_a(dt, A, _u(w, ins.meta), k, ins.extra)
        B = T.b_fragment(dt, w.regs[ins.b:ins.b + nb], k)
        C = T.c_fragment(w.regs[ins.c:ins.c + nc], accn)
        if dt in ("e4m3", "e5m2"):
            D = T.fp8_accumulate(A, B.T, C)
        elif accn == "s32":
            D = (A.astype(np.int64) @ B.astype(np.int64) + C).astype(np.int32)
        else:
            D = A.astype(np.float32) @ B.astype(np.float32) + C
        T.store_c_fragment(w.regs[ins.d:ins.d + nc], D, accn)


class TensorMap:
    """CUtensorMap stand-in (tiled mode, rank 1-5): a row-major tensor in global memory and the box
    one TMA moves. dims / box are fastest-varying first; strides[i] is the byte stride of dim i+1.
    `swizzle` (0/32/64/128 B) is how the box is laid out in shared memory. The box lands dense,
    dim 0 fastest. Elements outside the tensor are zero-filled on loads and skipped on stores."""

    def __init__(self, base, dims, strides, box, elem_bytes, swizzle=0):
        assert swizzle in (0, 32, 64, 128) and 1 <= len(dims) <= 5 and len(strides) == len(dims) - 1
        assert all(1 <= b <= 256 for b in box), "TMA box dimensions are limited to 256"
        assert (box[0] * elem_bytes) % 16 == 0, "the box's inner extent must be a multiple of 16 B"
        assert not swizzle or box[0] * elem_bytes == swizzle, "swizzled boxes are one swizzle-width wide"
        self.base, self.dims, self.strides, self.box, self.eb, self.swizzle = base, tuple(dims), tuple(strides), \
            tuple(box), elem_bytes, swizzle
        self.nbytes = int(np.prod(box)) * elem_bytes
        self.nrows = int(np.prod(box[1:]))                 # contiguous runs per box

    def _spans(self, coords):
        """(dense run index, global byte start, byte end, box byte offset) for every in-bounds run."""
        eb, dims, box = self.eb, self.dims, self.box
        c0, c1 = max(coords[0], 0), min(coords[0] + box[0], dims[0])
        if c1 <= c0:
            return
        rng = [range(b) for b in reversed(box[1:])]        # slowest dimension first
        for run, idx in enumerate(itertools.product(*rng)):
            addr = self.base + c0 * eb
            ok = True
            for d, i in zip(range(len(dims) - 1, 0, -1), idx):
                g = coords[d] + i
                if not 0 <= g < dims[d]:
                    ok = False
                    break
                addr += g * self.strides[d - 1]
            if ok:
                yield run, addr, addr + (c1 - c0) * eb, (c0 - coords[0]) * eb

    def sectors(self, *coords):
        secs = [np.arange(a0 >> 5, ((a1 - 1) >> 5) + 1) for _, a0, a1, _ in self._spans(coords)]
        return np.unique(np.concatenate(secs)) if secs else np.empty(0, np.int64)

    def gather(self, mem, base_va, *coords):
        """The box at `coords` as dense bytes, plus the 32 B sectors read. The whole box counts
        toward an mbarrier's tx."""
        tile = np.zeros((self.nrows, self.box[0] * self.eb), np.uint8)
        for run, a0, a1, b0 in self._spans(coords):
            tile[run, b0:b0 + (a1 - a0)] = mem[a0 - base_va:a1 - base_va]
        return tile.ravel(), self.sectors(*coords)

    def scatter(self, memory, payload, coords, reduce=False):
        """Write the dense box back to global memory, clipped to the tensor."""
        tile = payload.reshape(self.nrows, self.box[0] * self.eb)
        for run, a0, a1, b0 in self._spans(coords):
            o0, o1 = a0 - memory.base, a1 - memory.base
            src = tile[run, b0:b0 + (a1 - a0)]
            if reduce:
                memory.bytes[o0:o1].view(np.float32)[:] += src.view(np.float32)
            else:
                memory.bytes[o0:o1] = src


class TensorMapIm2col:
    """Tensor map in im2col mode (convolution activations, NHWC-style: dims = (C, W, H[, D], N),
    fastest first; strides[i] is the byte stride of dims[i+1]).

    A load names a starting *filter base* position (coords: c, w, h[, d], n) inside the bounding box
    and the filter-tap `offsets` (ow, oh[, od]) added to it. The box then walks `pixels_per_column`
    base positions in raster order -- W fastest, from the lower corner to dim + upper - 1 in steps of
    the traversal stride, wrapping into the next row / image -- and for each reads `channels_per_pixel`
    channels at (base + offsets). Anything outside the tensor reads as zero. The box lands dense as
    pixels x channels."""

    def __init__(self, base, dims, strides, lower, upper, channels_per_pixel, pixels_per_column, elem_bytes,
                 traversal=None, swizzle=0):
        sp = len(dims) - 2
        assert 1 <= sp <= 3 and len(strides) == len(dims) - 1 and len(lower) == len(upper) == sp
        assert channels_per_pixel <= 256 and pixels_per_column <= 1024
        assert (channels_per_pixel * elem_bytes) % 16 == 0
        assert not swizzle or channels_per_pixel * elem_bytes == swizzle
        self.base, self.dims, self.strides, self.eb = base, tuple(dims), tuple(strides), elem_bytes
        self.lower, self.upper, self.sp = tuple(lower), tuple(upper), sp
        self.cpp, self.ppc = channels_per_pixel, pixels_per_column
        self.trav = tuple(traversal or (1,) * sp)
        self.swizzle = swizzle
        self.nbytes = pixels_per_column * channels_per_pixel * elem_bytes

    def _pixels(self, coords, offsets):
        """(dense row, global byte start, byte end) of every in-bounds pixel of the box."""
        c, sp = coords[0], self.sp
        a, n = list(coords[1:1 + sp]), coords[1 + sp]
        c0, c1 = max(c, 0), min(c + self.cpp, self.dims[0])
        for p in range(self.ppc):
            pos = [a[i] + offsets[i] for i in range(sp)]
            ok = c1 > c0 and 0 <= n < self.dims[-1] and all(0 <= pos[i] < self.dims[1 + i] for i in range(sp))
            if ok:
                addr = self.base + c0 * self.eb + n * self.strides[-1]
                addr += sum(pos[i] * self.strides[i] for i in range(sp))
                yield p, addr, addr + (c1 - c0) * self.eb, (c0 - c) * self.eb
            a[0] += self.trav[0]
            for i in range(sp):                                     # carry W -> H -> D -> N
                if a[i] > self.dims[1 + i] + self.upper[i] - 1:
                    a[i] = self.lower[i]
                    if i + 1 < sp:
                        a[i + 1] += self.trav[i + 1]
                    else:
                        n += 1
                else:
                    break

    def sectors(self, coords, offsets):
        secs = [np.arange(a0 >> 5, ((a1 - 1) >> 5) + 1) for _, a0, a1, _ in self._pixels(coords, offsets)]
        return np.unique(np.concatenate(secs)) if secs else np.empty(0, np.int64)

    def gather(self, mem, base_va, coords, offsets):
        tile = np.zeros((self.ppc, self.cpp * self.eb), np.uint8)
        for p, a0, a1, b0 in self._pixels(coords, offsets):
            tile[p, b0:b0 + (a1 - a0)] = mem[a0 - base_va:a1 - base_va]
        return tile.ravel(), self.sectors(coords, offsets)

    # ---- im2col::w, im2col::w::128, im2col_no_offs::w (PTX ISA 5.5.5) ----
    def w_rows(self, mode, whalo):
        """Pixels (rows of the shared-memory box) a ::w load brings in."""
        if mode == "w128":
            return 4 * (32 + whalo)                 # 128 pixels, wHalo more after every 32
        return self.ppc + whalo                      # Pixels-per-Column, then wHalo halo pixels at the end

    def _w_pixels(self, coords, mode, whalo, woffset):
        """(box row, global byte start, byte end, box byte offset) for the pixels of a ::w load. Only W varies:
        H and D (and N) stay at the coordinates given; positions outside the tensor read as zero. The first pixel
        is at W + wOffset and pixels step by the W traversal stride; the halo pixels continue the sequence."""
        c, sp = coords[0], self.sp
        w0 = coords[1] + woffset
        fixed = list(coords[2:2 + sp - 1])            # H (and D) positions
        n = coords[1 + sp]
        c0, c1 = max(c, 0), min(c + self.cpp, self.dims[0])
        stride = self.trav[0]
        ok_rest = c1 > c0 and 0 <= n < self.dims[-1] and all(0 <= fixed[i] < self.dims[2 + i] for i in range(sp - 1))
        if not ok_rest:
            return
        base = self.base + c0 * self.eb + n * self.strides[-1] + sum(fixed[i] * self.strides[1 + i] for i in range(sp - 1))
        if mode == "w128":
            positions = [32 * g + i for g in range(4) for i in range(32 + whalo)]       # main then halo, per group
        else:
            positions = list(range(self.ppc + whalo))
        for row, k in enumerate(positions):
            w = w0 + k * stride
            if 0 <= w < self.dims[1]:
                a0 = base + w * self.strides[0]
                yield row, a0, a0 + (c1 - c0) * self.eb, (c0 - c) * self.eb

    def gather_w(self, mem, base_va, coords, mode="w", whalo=0, woffset=0):
        if mode == "w_no_offs":
            if whalo or woffset:
                raise RuntimeError("im2col_no_offs::w takes no wHalo / wOffset")
            if min(coords) < 0:
                raise RuntimeError("im2col_no_offs::w: tensor coordinates must not be negative")
        rows = self.w_rows("w128" if mode == "w128" else "w", whalo)
        tile = np.zeros((rows, self.cpp * self.eb), np.uint8)
        secs = []
        for r, a0, a1, b0 in self._w_pixels(coords, "w128" if mode == "w128" else "w", whalo, woffset):
            tile[r, b0:b0 + (a1 - a0)] = mem[a0 - base_va:a1 - base_va]
            secs.append(np.arange(a0 >> 5, ((a1 - 1) >> 5) + 1))
        return tile.ravel(), (np.unique(np.concatenate(secs)) if secs else np.empty(0, np.int64))


@dataclass
class Launch:
    """One kernel launch in a chain. pdl=True makes it a programmatic-dependent-launch secondary."""
    prog: Program
    grid: int
    block: int
    params: dict
    smem_bytes: int = 0
    cluster: int = 1
    pdl: bool = False
    stream: int = 0          # kernels in different streams run concurrently; one stream runs in order


class Kernel:
    """Everything that belongs to one launched grid."""

    def __init__(self, sim, kid, L, cfg):
        prog = L.prog
        if sim.dep_mode == "control":
            control.annotate(prog)
        assert L.block % 32 == 0 and prog.launch_regs <= cfg.max_regs_per_thread and L.grid % L.cluster == 0
        assert L.cluster <= 16 and L.smem_bytes <= cfg.smem_per_cta_max
        self.id, self.prog, self.instrs = kid, prog, prog.instrs
        self.grid, self.block, self.params = L.grid, L.block, L.params
        self.smem_bytes, self.cluster_size, self.pdl = L.smem_bytes, L.cluster, L.pdl
        self.warps_per_cta = L.block // 32
        per_warp = math.ceil(prog.launch_regs * 32 / cfg.reg_alloc_warp_unit) * cfg.reg_alloc_warp_unit
        self.regs_per_cta = per_warp * self.warps_per_cta
        self.param_idx = {n: i for i, n in enumerate(L.params)}
        self.nclusters = L.grid // L.cluster
        self.next_cl = self.resident = self.ctas_done = self.triggered = self.issued = 0
        self.op_counts = Counter()
        self.last_store_t = 0
        self.tc_busy = 0.0               # sub-core cycles the tensor pipes were busy
        self.started = self.complete = False
        self.start_at = None
        self.begin_cycle = self.end_cycle = 0
        self.after = self.next_kernel = None
        self.waiters = set()

    def param_line(self, name):
        return self.param_idx[name] // 8      # 8 parameters of 8 B per 64 B constant line


@dataclass
class ChainStats:
    kernels: list
    cycles: int
    time_us: float

    def report(self):
        lines = [f"chain of {len(self.kernels)} kernels: {self.cycles} cycles = {self.time_us:.2f} us"]
        for s in self.kernels:
            lines.append(f"  {s.kernel}: cycles {s.begin_cycle}..{s.end_cycle} ({s.time_us:.2f} us)")
        return "\n".join(lines)


@dataclass
class LaunchStats:
    kernel: str
    cycles: int
    time_us: float
    ctas: int
    warps_per_cta: int
    ctas_per_sm_max: int
    instrs: int
    hbm_bytes: int
    hbm_tbs: float
    hbm_write_bytes: int
    l1_hit: int
    l2_hit: int
    l2_miss: int
    issue_util: float
    state_cycles: dict = field(default_factory=dict)
    op_counts: dict = field(default_factory=dict)
    begin_cycle: int = 0
    end_cycle: int = 0
    power_w: float = 0.0              # estimated average board power (assumed coefficients, see config)
    clock_scale: float = 1.0          # DVFS: fraction of the nominal clock the power cap allows
    time_us_dvfs: float = 0.0         # time_us at the throttled clock
    ecc_corrected: int = 0
    nvlink_bytes: int = 0
    hazards: list = field(default_factory=list)

    def report(self, floor_bytes=None):
        c = self.state_cycles
        tot = sum(v for k, v in c.items() if k != "exited") or 1
        top = ", ".join(f"{k} {v}" for k, v in sorted(self.op_counts.items(), key=lambda kv: -kv[1])[:8])
        lines = [
            f"kernel {self.kernel}: {self.cycles} cycles = {self.time_us:.2f} us",
            f"  grid {self.ctas} CTAs x {self.warps_per_cta} warps, up to {self.ctas_per_sm_max} CTAs/SM resident",
            f"  warp instrs {self.instrs}, issue-slot utilisation {100 * self.issue_util:.1f}%",
            f"  top ops: {top}",
            f"  HBM read {self.hbm_bytes / 1e6:.2f} MB -> {self.hbm_tbs:.2f} TB/s average over the kernel"
            f" (+{self.hbm_write_bytes / 1e6:.2f} MB dirty write-back)",
            f"  L1 sector hits {self.l1_hit}, L2 hits {self.l2_hit}, L2 misses {self.l2_miss}",
            "  warp time by state: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in c.items() if k != "exited"),
        ]
        if floor_bytes:
            lines.append(f"  bandwidth floor ({floor_bytes / 1e6:.2f} MB at 4.8 TB/s): "
                         f"{floor_bytes / 4.8e12 * 1e6:.2f} us")
        return "\n".join(lines)


class Sim:
    def __init__(self, cfg: H200 | None = None, trace_sms=(), dep_mode: str = "scoreboard", mem=None, node=None,
                 gpu_id: int = 0):
        """dep_mode: 'scoreboard' (per-register hardware scoreboard) or 'control' (compiler stall counts and
        dependence counters, as the real ISA does; see control.py)."""
        assert dep_mode in ("scoreboard", "control")
        self.dep_mode = dep_mode
        self.cfg = cfg or H200()
        assert sum(self.cfg.gpc_sizes) == self.cfg.num_sms
        self.node, self.gpu_id = node, gpu_id
        self.tpc_tlbs = [LRUSet(self.cfg.tlb_l1_entries) for _ in range((self.cfg.num_sms + 1) // 2)]
        self.memsys = MemorySystem(self.cfg, mem, node, gpu_id)
        self.frontend = FrontEnd(self.cfg)
        self.mem = self.memsys.mem
        gpc_of = [g for g, n in enumerate(self.cfg.gpc_sizes) for _ in range(n)]
        self.sms = [SM(i, self, gpc_of[i]) for i in range(self.cfg.num_sms)]
        self.gpcs = [[sm for sm in self.sms if sm.gpc == g] for g in range(len(self.cfg.gpc_sizes))]
        self.trace_sms = set(trace_sms)
        self.events = []                 # warp state transitions (when tracing)
        self.retired: list[Warp] = []    # finished warps, kept so registers can be inspected
        self.cta_done: list[CTA] = []
        self.tmaps: list[TensorMap] = []
        self.op_counts = Counter()
        self.hazards = []                # ordering problems noticed during the run
        self._hazard_seen = set()
        self.constant = np.zeros(64 * 1024, np.uint8)       # the 64 KB constant bank (__constant__)
        self.now = 0
        self.log_issue = False          # set True to record (cycle, sm, smsp, cta, warp, pc, op) per issue
        self.issue_log = []
        self.heap = []                   # (cycle, seq, fn): async completions
        self.seq = itertools.count()
        self.cycle = 0

    def inject_ecc_error(self, addr: int, bits: int = 1) -> None:
        self.memsys.inject_ecc_error(addr, bits)

    def map_peer(self, nbytes: int) -> int:
        """Allocate memory that lives on a peer GPU: loads, stores, atomics and TMA reach it over NVLink."""
        return self.mem.alloc_remote(nbytes)

    def write_constant(self, offset: int, arr: np.ndarray) -> None:
        raw = np.ascontiguousarray(arr).view(np.uint8).ravel()
        self.constant[offset:offset + raw.size] = raw

    def tensor_map(self, base, rows, cols, elem_bytes, box_rows, box_cols, row_stride=None, swizzle=0) -> int:
        """A 2D row-major [rows, cols] tensor and a box_rows x box_cols box."""
        return self.tensor_map_nd(base, (cols, rows), ((row_stride or cols * elem_bytes),),
                                  (box_cols, box_rows), elem_bytes, swizzle)

    def tensor_map_nd(self, base, dims, strides, box, elem_bytes, swizzle=0) -> int:
        """Tiled tensor map of rank 1-5. dims and box are fastest-varying first; strides[i] is the
        byte stride of dims[i+1]."""
        self.tmaps.append(TensorMap(base, dims, strides, box, elem_bytes, swizzle))
        return len(self.tmaps) - 1

    def tensor_map_im2col(self, base, dims, strides, lower, upper, channels_per_pixel, pixels_per_column,
                          elem_bytes, traversal=None, swizzle=0) -> int:
        """cuTensorMapEncodeIm2col: see TensorMapIm2col."""
        self.tmaps.append(TensorMapIm2col(base, dims, strides, lower, upper, channels_per_pixel,
                                          pixels_per_column, elem_bytes, traversal, swizzle))
        return len(self.tmaps) - 1

    def at(self, t, fn):
        heapq.heappush(self.heap, (int(t), next(self.seq), fn))

    def _pump(self, cycle):
        while self.heap and self.heap[0][0] <= cycle:
            t, _, fn = heapq.heappop(self.heap)
            fn(t)

    # ---- block scheduler ----
    def _place(self, k, cluster_idx, cycle):
        cs = k.cluster_size
        need = (k.warps_per_cta, k.regs_per_cta, k.smem_bytes)
        if cs == 1:
            cands = [sm for sm in self.sms if sm.fits(*need)]
            if not cands:
                return False
            chosen = [min(cands, key=lambda sm: (len(sm.ctas), sm.id))]
        else:
            chosen = None
            for g in range(len(self.gpcs)):
                gpc = self.gpcs[(cluster_idx + g) % len(self.gpcs)]
                cands = sorted((sm for sm in gpc if sm.fits(*need)), key=lambda sm: (len(sm.ctas), sm.id))
                if len(cands) >= cs:
                    chosen = cands[:cs]
                    break
            if chosen is None:
                return False
        cl = Cluster()
        for r, sm in enumerate(chosen):
            sm.launch_cta(k, cluster_idx * cs + r, r, cl, cycle)
        return True

    def async_read_smem(self, w, ins):
        """The async proxy (TMA stores, wgmma) reads shared memory; generic-proxy writes need a
        fence.proxy.async first or the read may see stale data."""
        if w.cta.generic_dirty:
            key = (w.cta.kernel.id, w.pc - 1)
            if key not in self._hazard_seen:
                self._hazard_seen.add(key)
                msg = (f"kernel {w.cta.kernel.prog.name} pc={w.pc - 1}: {ins.op} reads shared memory written by "
                       f"the generic proxy with no fence.proxy.async in between")
                if self.cfg.strict_proxy_fence:
                    raise RuntimeError(msg)
                self.hazards.append(msg)

    def trigger_cta(self, cta, cycle):
        """griddepcontrol.launch_dependents (or the CTA ending): when every CTA of the grid has done
        so, a PDL successor may be scheduled."""
        if cta.triggered:
            return
        cta.triggered = True
        k = cta.kernel
        k.triggered += 1
        if k.triggered == k.grid and k.next_kernel is not None and k.next_kernel.pdl:
            k.next_kernel.start_at = cycle + int(self.cfg.pdl_gap_us * self.cfg.clock_ghz * 1e3)

    def _complete_kernel(self, k, cycle):
        k.complete, k.end_cycle = True, cycle
        for w in k.waiters:
            w.wake_at, w.smsp.wake = max(cycle, w.since), 0
        k.waiters.clear()
        n = k.next_kernel
        if n is not None and not n.pdl and n.start_at is None:
            gap = dict(self.cfg.launch_gap_us)[self.launch_mode]
            n.start_at = cycle + int(gap * self.cfg.clock_ghz * 1e3)

    def launch(self, prog: Program, grid: int, block: int, params: dict, smem_bytes: int = 0,
               cluster: int = 1) -> LaunchStats:
        """Run one kernel to completion."""
        return self.launch_chain([Launch(prog, grid, block, params, smem_bytes, cluster)]).kernels[0]

    def launch_chain(self, launches, mode: str = "graph") -> "ChainStats":
        """Run kernels to completion (see `_run_chain`)."""
        gen = self._run_chain(launches, mode)
        try:
            while True:
                next(gen)
        except StopIteration as done:
            return done.value

    def _run_chain(self, launches, mode: str = "graph"):
        """Generator form of the run loop: yields the cycle of the next iteration before executing it, so a
        multi-GPU Node can advance every GPU in global time order; returns the ChainStats. Runs kernels; those with the same `stream` run in order, different streams run concurrently
        and share the SMs. A launch with pdl=True is a programmatic-dependent-launch
        secondary: it is scheduled as soon as every CTA of its predecessor executes
        `griddep_launch` (or exits), and blocks at `griddep_wait` until the predecessor has
        completed. Other launches start a launch gap (graph / stream) after the predecessor ends."""
        cfg = self.cfg
        self.launch_mode = mode
        kernels, tail = [], {}
        for i, L in enumerate(launches):
            k = Kernel(self, i, L, cfg)
            if L.stream in tail:                                # stream order: wait for the stream's previous kernel
                k.after = tail[L.stream]
                tail[L.stream].next_kernel = k
            tail[L.stream] = k
            kernels.append(k)
        self.retired.clear()
        self.events.clear()
        self.heap.clear()
        self.issue_log.clear()
        self.hazards.clear()
        self._hazard_seen.clear()
        self.memsys.reset_stats()
        self.frontend.reset_kernel(self.sms)
        for sm in self.sms:
            sm.pending.clear()
            sm.issued = 0
            sm.l1.clear()
            sm.port_free = sm.dsmem_free = sm.smem_free = 0.0
            sm.lsu_free = 0
            sm.mem_accept = 0
            for sp in sm.smsp:
                sp.issue_slots = 0
                sp.mem_q = []
                sp.pipe_free = {key: 0 for key in sp.pipe_free}
        self.memsys.l2_free = [0.0, 0.0]
        self.memsys.xbar_free = 0.0
        self.memsys.nvl_in_free = self.memsys.nvl_out_free = 0.0
        self.memsys.hbm_free = [0.0] * cfg.hbm_channels
        start = cycle = self.cycle
        for k in kernels:
            if k.after is None:                                 # first kernel of each stream
                k.start_at = cycle
        max_res = 0
        replace = True
        while True:
            yield cycle
            for k in kernels:                                   # kernels whose launch time has come
                if not k.started and k.start_at is not None and k.start_at <= cycle:
                    k.started, k.begin_cycle = True, cycle
                    for line in {i // 8 for i in range(len(k.params))}:
                        self.frontend.l2_const[("p", k.id, line)] = 1
                    replace = True
            if replace:
                for k in kernels:
                    while k.started and k.next_cl < k.nclusters and self._place(k, k.next_cl, cycle):
                        k.next_cl += 1
                    if k.started and k.resident == 0 and k.next_cl == 0 and not k.complete:
                        raise RuntimeError("CTA / cluster does not fit (registers, shared memory, or cluster size vs GPC)")
                cur = max(len(sm.ctas) for sm in self.sms)
                if cur > max_res:
                    max_res = cur
                    self._set_l1_capacity(max(k.smem_bytes for k in kernels if k.started), max_res)
                replace = False
            self._pump(cycle)
            any_issue = False
            wake = INF
            for sm in self.sms:
                if not sm.ctas:
                    continue
                for sp in sm.smsp:
                    if sp.wake <= cycle:
                        any_issue |= sp.step(cycle, self)
                    wake = min(wake, sp.wake)
            if self.heap:
                wake = min(wake, self.heap[0][0])
            if self.cta_done:
                for cta in self.cta_done:
                    cta.sm.retire_cta(cta)
                    self.trigger_cta(cta, cycle)
                    k = cta.kernel
                    if k.ctas_done == k.grid:                  # complete once the kernel's writes are visible
                        self.at(max(cycle, k.last_store_t), lambda t, k=k: self._complete_kernel(k, t))
                self.cta_done.clear()
                replace, wake = True, 0
            pend = [k.start_at for k in kernels if not k.started and k.start_at is not None]
            if pend:
                wake = min(wake, min(pend))
            if all(k.complete for k in kernels):
                break
            if wake >= INF and not any_issue:
                raise RuntimeError("deadlock: no warp can ever issue (barrier / mbarrier / griddep_wait mismatch?)")
            cycle = cycle + 1 if any_issue or wake <= cycle else wake
        while self.heap:                 # outstanding stores / copies still have to land
            t, _, fn = heapq.heappop(self.heap)
            fn(t)
            cycle = max(cycle, t)
        self.cycle = cycle
        stats = [self._stats(k, max_res) for k in kernels]
        return ChainStats(stats, cycle - start, (cycle - start) / (cfg.clock_ghz * 1e3))

    def _set_l1_capacity(self, smem_bytes, resident_ctas):
        """L1 and shared memory share 256 KB: the shared-memory carve-out is the smallest
        configuration holding the resident CTAs, and L1 gets what is left (up to 208 KB)."""
        cfg = self.cfg
        need = resident_ctas * (smem_bytes + cfg.l1_reserved) if smem_bytes else 0
        carve = next((c for c in cfg.smem_carveouts if c >= need), cfg.smem_carveouts[-1])
        l1 = min(cfg.l1_max_bytes, cfg.l1_smem_unified - carve - cfg.l1_reserved)
        self.l1_bytes, self.smem_carveout = l1, carve
        for sm in self.sms:
            sm.l1.cap = l1 // 32

    def _stats(self, k, max_res):
        start = k.begin_cycle
        cyc = max(1, k.end_cycle - start)
        hist = {s.value: 0 for s in WS}
        for w in self.retired:
            if w.cta.kernel is k:
                for s, v in w.hist.items():
                    hist[s.value] += v
        ms = self.memsys
        slots = cyc * self.cfg.num_sms * self.cfg.smsp_per_sm
        return LaunchStats(
            kernel=k.prog.name, cycles=cyc, time_us=cyc / (self.cfg.clock_ghz * 1e3), ctas=k.grid,
            warps_per_cta=k.warps_per_cta, ctas_per_sm_max=max_res, instrs=k.issued,
            hbm_bytes=ms.hbm_bytes, hbm_tbs=ms.hbm_bytes / (cyc / (self.cfg.clock_ghz * 1e9)) / 1e12,
            hbm_write_bytes=ms.hbm_write_bytes,
            l1_hit=ms.l1_hit, l2_hit=ms.l2_hit, l2_miss=ms.l2_miss,
            issue_util=k.issued / slots, state_cycles=hist, op_counts=dict(k.op_counts),
            begin_cycle=start, end_cycle=k.end_cycle, ecc_corrected=ms.ecc_corrected,
            nvlink_bytes=ms.nvlink_bytes, hazards=list(self.hazards),
            **self._power(k, cyc, slots))

    def _power(self, k, cyc, slots):
        """Average board power from activity and the SM clock the cap allows.

        P = static + HBM (memory clock is fixed, so it does not scale) + dynamic * s^3, where s is the SM clock
        relative to boost (voltage follows frequency, so dynamic power ~ f^3) and `dynamic` collects L2/NoC
        movement, instruction issue and tensor activity at the boost clock. Under a cap,
        s = ((cap - static - mem) / dynamic)^(1/3), floored at the 345 MHz minimum clock."""
        c, ms = self.cfg, self.memsys
        tc_util = min(1.0, k.tc_busy / max(1, cyc * c.num_sms * c.smsp_per_sm))
        hbm_util = min(1.0, ms.hbm_bytes / (cyc * c.hbm_bytes_per_cycle))
        l2_util = min(1.0, (ms.l2_hit + ms.l2_miss) * 32 / (cyc * c.l2_bytes_per_cycle))
        mem = c.power_mem_w * hbm_util
        dyn = (c.power_l2_w * l2_util + c.power_issue_w * min(1.0, k.issued / slots)
               + tc_util * (c.power_tensor_zero_w + (c.power_tensor_random_w - c.power_tensor_zero_w) * c.tensor_data_toggle))
        scale = 1.0
        if c.power_static_w + mem + dyn > c.power_cap_w and dyn > 0:
            scale = max(c.clock_min_ghz / c.clock_ghz, ((c.power_cap_w - c.power_static_w - mem) / dyn) ** (1 / 3)
                        if c.power_cap_w > c.power_static_w + mem else 0.0)
        power = c.power_static_w + mem + dyn * scale ** 3
        t = cyc / (c.clock_ghz * 1e3)
        return dict(power_w=power, clock_scale=scale, time_us_dvfs=t / scale)

    def dump_regs(self, warp: Warp, lanes=(0, 1, 2, 3), upto=None):
        n = upto or warp.regs.shape[0]
        out = [f"SM{warp.cta.sm.id} SMSP{warp.smsp.idx} CTA{warp.cta.id} warp{warp.wid}  pc={warp.pc}"]
        out.append("reg   " + "  ".join(f"lane{l:<10}" for l in lanes))
        for r in range(n):
            row = warp.regs[r]
            out.append(f"R{r:<4} " + "  ".join(f"{row[l]:#010x}/{row[l:l+1].view(np.float32)[0]:<9.4g}" for l in lanes))
        return "\n".join(out)
