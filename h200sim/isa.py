"""A small SASS-like ISA and an assembler.

Registers are 32-bit, per thread. `Reg` is an int subclass, so plain ints/floats are
immediates and `Reg` instances are register operands. Control flow is warp-uniform
(a branch whose predicate differs between lanes raises); memory ops and arithmetic are
per-lane. Global addresses are 32-bit byte offsets.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from . import tensor as T


class Reg(int):
    """A register index; `r + 1` is the next register (for vector register groups)."""

    def __add__(self, other):
        return Reg(int(self) + int(other))

    def __repr__(self):
        return f"R{int(self)}"


class UReg(int):
    """A uniform register: one 32-bit value per warp (64 per warp), written by the uniform
    datapath. Usable as an address / scalar operand almost anywhere a register is."""

    def __repr__(self):
        return f"UR{int(self)}"


class Pred(int):
    """A per-lane predicate register P0..P6 (P7 is always true)."""

    def __repr__(self):
        return f"P{int(self)}"


UBASE, PBASE = 256, 320       # scoreboard ids of uniform registers and predicates (R0..R255 below)
PT = Pred(7)


# op -> (pipe, result latency, pipe occupancy in cycles)
TIMING = {
    "MOV": ("alu", 4, 1), "S2R": ("alu", 4, 1), "LDC": ("alu", 4, 1),
    "IADD": ("int", 4, 2), "IMAD": ("int", 5, 2), "SHL": ("int", 4, 2), "SHR": ("int", 4, 2),
    "AND": ("int", 4, 2), "OR": ("int", 4, 2), "XOR": ("int", 4, 2), "SETP_LT": ("int", 4, 2),
    "FADD": ("fp32", 4, 1), "FMUL": ("fp32", 4, 1), "FFMA": ("fp32", 4, 1),
    "SHFL_XOR": ("xu", 24, 2),
    "HMMA": ("tensor", 33, 6),
    "LDG": ("lsu", 0, 0), "STG": ("lsu", 0, 0), "RED_ADD_F32": ("lsu", 0, 0),
    "LDS": ("lsu", 0, 0), "STS": ("lsu", 0, 0),
    "SETP_EQ": ("int", 4, 2), "SETP_GE": ("int", 4, 2),
    "LDC64": ("alu", 4, 1), "IADD64": ("int", 6, 4), "IMAD_WIDE": ("int", 6, 4),
    "TMA_STORE_2D": ("tma", 0, 4), "TMA_PREFETCH_2D": ("tma", 0, 4),
    "TMA_LOAD_ND": ("tma", 0, 4), "TMA_STORE_ND": ("tma", 0, 4), "TMA_LOAD_IM2COL": ("tma", 0, 4),
    "DMMA": ("tensor", 32, 8),          # m8n8k4 fp64 tensor core (latency assumed, rate from the whitepaper)
    # async copy / barriers / bulk engines
    "LDGSTS": ("lsu", 0, 0), "CP_COMMIT": ("alu", 0, 1), "CP_WAIT": ("alu", 0, 1),
    "MBAR_INIT": ("alu", 0, 1), "MBAR_ARRIVE": ("alu", 0, 1), "MBAR_ARRIVE_TX": ("alu", 0, 1),
    "MBAR_ARRIVE_CL": ("alu", 0, 1), "MBAR_WAIT": ("alu", 0, 1), "FENCE": ("alu", 0, 1),
    "TMA_LOAD_1D": ("tma", 0, 4), "TMA_LOAD_2D": ("tma", 0, 4), "TMA_STORE_1D": ("tma", 0, 4),
    # warpgroup MMA
    "WGMMA": ("none", 0, 0), "WGMMA_FENCE": ("alu", 0, 1), "WGMMA_COMMIT": ("alu", 0, 1),
    "WGMMA_WAIT": ("alu", 0, 1),
    # clusters
    "LDS_CL": ("lsu", 0, 0), "STS_CL": ("lsu", 0, 0), "CLUSTER_SYNC": ("none", 0, 0),
    "BAR": ("none", 0, 0), "BRA": ("none", 0, 0), "BRNZ": ("none", 0, 0), "EXIT": ("none", 0, 0),
    # latencies below measured on Volta / Pascal (Jia et al., Table 4.1): FP64 8, HFMA2 6, MUFU/FLO/BREV 14,
    # POPC 10, conversions 14 (Pascal), IADD3/LOP3/SHF/ISETP 4, IMAD 5. FP64 issues one warp-instr / 2 cycles per sub-core on GH100
    "DADD": ("fp64", 8, 2), "DMUL": ("fp64", 8, 2), "DFMA": ("fp64", 8, 2),
    # packed 16-bit math
    "HADD2": ("fp32", 6, 1), "HMUL2": ("fp32", 6, 1), "HFMA2": ("fp32", 6, 1),
    "HADD2_BF16": ("fp32", 6, 1), "HMUL2_BF16": ("fp32", 6, 1), "HFMA2_BF16": ("fp32", 6, 1),
    # conversions and the special-function unit: 16 lanes/SM -> 8 cycles per warp (the issue interval is assumed)
    "I2F": ("xu", 14, 4), "F2I": ("xu", 14, 4), "F2FP_BF16": ("xu", 14, 4), "F2FP_F16": ("xu", 14, 4),
    "F2FP_E4M3": ("xu", 14, 4), "F2FP_E5M2": ("xu", 14, 4),
    "CVT_F16X2_E4M3": ("xu", 14, 4), "CVT_F16X2_E5M2": ("xu", 14, 4),
    "MUFU_RCP": ("xu", 14, 8), "MUFU_RSQ": ("xu", 14, 8), "MUFU_SQRT": ("xu", 14, 8),
    "MUFU_EX2": ("xu", 14, 8), "MUFU_LG2": ("xu", 14, 8), "MUFU_SIN": ("xu", 14, 8),
    "MUFU_COS": ("xu", 14, 8), "MUFU_TANH": ("xu", 14, 8),
    # min/max, DPX dynamic-programming ops, bit twiddling
    "IMNMX": ("int", 5, 2), "FMNMX": ("fp32", 5, 1), "VIMNMX3": ("int", 6, 2), "VIADDMNMX": ("int", 6, 2),
    "PRMT": ("int", 4, 2), "LOP3": ("int", 4, 2), "SHF": ("int", 4, 2), "POPC": ("xu", 10, 4), "BREV": ("xu", 14, 4),
    "FLO": ("xu", 14, 4), "IABS": ("int", 4, 2), "SEL": ("int", 4, 2), "IMUL_HI": ("int", 5, 2),
    "ISETP": ("int", 4, 2), "FSETP": ("fp32", 4, 1), "PSETP": ("int", 4, 2),
    # warp-level
    "SHFL_UP": ("xu", 24, 2), "SHFL_DOWN": ("xu", 24, 2), "SHFL_IDX": ("xu", 24, 2),
    "VOTE_BALLOT": ("xu", 24, 2), "VOTE_ANY": ("xu", 24, 2), "VOTE_ALL": ("xu", 24, 2),
    "REDUX": ("xu", 24, 2), "ELECT": ("alu", 4, 1), "ACTIVEMASK": ("alu", 4, 1), "NANOSLEEP": ("alu", 0, 1),
    # memory
    "ATOM": ("lsu", 0, 0), "ATOMS": ("lsu", 0, 0), "LDC_MEM": ("alu", 0, 1),
    "LDMATRIX": ("lsu", 0, 0), "STMATRIX": ("lsu", 0, 0),
    "SETMAXNREG": ("alu", 0, 1), "BAR_ARRIVE": ("none", 0, 0),
    "GRIDDEP_LAUNCH": ("alu", 0, 1), "GRIDDEP_WAIT": ("alu", 0, 1), "MEMBAR": ("alu", 0, 1),
    # uniform datapath (latency / occupancy assumed)
    "UMOV": ("uni", 5, 2), "UIADD": ("uni", 5, 2), "UIMAD": ("uni", 5, 2), "USHL": ("uni", 5, 2),
    "USHR": ("uni", 5, 2), "UAND": ("uni", 5, 2), "UOR": ("uni", 5, 2), "UISETP": ("uni", 5, 2),
    "S2UR": ("uni", 5, 2), "ULDC": ("uni", 5, 2), "R2UR": ("uni", 8, 2),
}


@dataclass
class Instr:
    op: str
    d: Any = None
    a: Any = None
    b: Any = None
    c: Any = None
    imm: int = 0
    width: int = 4
    cache: str = "ca"
    name: str = ""
    target: Any = None
    reconv: Any = None    # reconvergence point of a possibly-divergent branch
    extra: int = 0        # multicast mask / accumulate flag
    lda: int = 0
    ldb: int = 0
    coords: tuple = ()    # TMA tensor coordinates (fastest dimension first), for rank > 2
    offs: tuple = ()      # TMA im2col filter-tap offsets
    wide: bool = False    # global address operand is a 64-bit register pair
    swz: int = 0          # wgmma shared-memory swizzle: 0, 32, 64 or 128 bytes
    a_regs: bool = False  # wgmma A operand comes from registers
    yield_: bool = False  # control bit Yield: do not issue this warp again next cycle (switch warps if possible)
    stall: int = 1        # control bits (see control.py): cycles to wait after issuing this instruction
    wait: tuple = ()      # ... dependence counters that must be zero before this issues
    inc_raw: Any = None   # ... counter incremented at issue, decremented at write-back
    inc_war: Any = None   # ... counter incremented at issue, decremented once the sources are read
    sel: int = 0          # sparsity selector (mma.sp / sparse wgmma): which lanes of a quad supply the metadata
    sem: str = ""         # memory semantics of a global access: weak / relaxed / acquire / release
    guard: Any = None     # (predicate index, negate): the instruction runs on lanes where it holds
    maxreg: int = 0       # highest R register index + 1 the instruction touches
    opt: str = ""         # sub-operation: cmp / min / max / rounding / format
    dtype: str = ""       # tensor ops: input element type
    acc: str = ""         # tensor ops: accumulator type (f32, f16, s32)
    k: int = 0            # mma.sync K
    sparse: bool = False  # 2:4 structured sparsity (wgmma / mma.sp)
    meta: Any = None      # mma.sp metadata register
    srcs: tuple = ()      # ((operand slot, register), ...) of the regular-register sources
    reuse: tuple = ()     # operand slots whose register-file-cache reuse bit is set
    deps: tuple = ()      # registers read or written (scoreboard checks)
    writes: tuple = ()
    pipe: str = "alu"
    lat: int = 0
    occ: int = 0

    def __str__(self):
        parts = [self.op] + [str(x) for x in (self.d, self.a, self.b, self.c) if x is not None]
        if self.imm:
            parts.append(f"+{self.imm}")
        if self.name:
            parts.append(self.name)
        if self.target is not None:
            parts.append(f"->{self.target}")
        return " ".join(parts)


@dataclass
class Program:
    instrs: list
    nregs: int             # highest register used + 1 (storage)
    name: str = "kernel"
    launch_regs: int = 0   # registers per thread the kernel launches with (setmaxnreg can move it)


def _did(x):
    """Scoreboard id of an operand: R0..R255, uniform registers, predicates; None for immediates."""
    if isinstance(x, Reg):
        return int(x)
    if isinstance(x, UReg):
        return UBASE + int(x)
    if isinstance(x, Pred) and int(x) != 7:
        return PBASE + int(x)
    return None


def _regs(*xs):
    return tuple(i for i in map(_did, xs) if i is not None)


def _span(base, n):
    return tuple(range(int(base), int(base) + n))


def _wdeps(addr, wide):
    """Registers an address operand reads: one, or the pair of a 64-bit pointer."""
    return _span(addr, 2) if wide else _regs(addr)


class Asm:
    def __init__(self, name="kernel"):
        self.name = name
        self.code: list[Instr] = []
        self.labels: dict[str, int] = {}
        self.nxt = 0
        self.nu = 0
        self.np = 0
        self._guard = None

    # ---- registers / labels ----
    def reg(self, n: int = 1) -> Reg:
        """Allocate n consecutive registers, aligned to n (as vector accesses require)."""
        align = n if n in (2, 4) else 1
        self.nxt = (self.nxt + align - 1) // align * align
        r = Reg(self.nxt)
        self.nxt += n
        return r

    def label(self, name: str):
        self.labels[name] = len(self.code)

    def _emit(self, op, deps=(), writes=(), **kw) -> Instr:
        pipe, lat, occ = TIMING[op]
        ins = Instr(op=op, deps=tuple(dict.fromkeys(deps + writes)), writes=tuple(writes),
                    pipe=pipe, lat=lat, occ=occ, guard=self._guard, **kw)
        if self._guard is not None and self._guard[0] != 7:
            ins.deps = tuple(dict.fromkeys(ins.deps + (PBASE + self._guard[0],)))
        self.code.append(ins)
        return ins

    # ---- predicates, uniform registers, guards ----
    def preg(self) -> Pred:
        assert self.np < 7, "only P0..P6 are allocatable"
        self.np += 1
        return Pred(self.np - 1)

    def ureg(self, n: int = 1) -> UReg:
        assert self.nu + n <= 64, "64 uniform registers per warp"
        self.nu += n
        return UReg(self.nu - n)

    @contextmanager
    def guard(self, p, negate=False):
        """`with a.guard(p):` runs every instruction emitted inside only on lanes where p holds
        (a SASS @P0 / @!P0 prefix). Guarded `bra` is a possibly-divergent branch."""
        assert self._guard is None, "guards do not nest"
        self._guard = (int(p), bool(negate))
        try:
            yield
        finally:
            self._guard = None

    def _gen(self, op, d=None, a=None, b=None, c=None, dw=1, aw=1, bw=1, cw=1, **kw):
        """Emit an op whose operands may be registers (with a width), uniform registers,
        predicates or immediates. Sources become scoreboard dependences, `d` is written."""
        def span(x, w):
            return _span(x, w) if isinstance(x, Reg) else _regs(x)
        deps = span(a, aw) + span(b, bw) + span(c, cw)
        writes = span(d, dw) if d is not None else ()
        return self._emit(op, deps=deps, writes=writes, d=d, a=a, b=b, c=c, **kw)

    # ---- ALU ----
    def _alu(self, op, d, a, b=None, c=None):
        srcs = tuple((i, int(x)) for i, x in enumerate((a, b, c)) if isinstance(x, Reg))
        self._emit(op, deps=_regs(a, b, c), writes=_regs(d), d=d, a=a, b=b, c=c, srcs=srcs)

    def yield_hint(self):
        """Set the Yield control bit on the instruction just emitted (control-bit mode): the warp is not
        eligible next cycle, so the scheduler moves to another warp; with no other warp ready a bubble results."""
        self.code[-1].yield_ = True

    def reuse(self, *slots):
        """Set the register-file-cache reuse bit on source operand slots (0 = first source) of the
        instruction just emitted, as the compiler does for operands the next instruction re-reads."""
        self.code[-1].reuse = tuple(slots)

    def mov(self, d, a): self._alu("MOV", d, a)
    def iadd(self, d, a, b): self._alu("IADD", d, a, b)
    def imad(self, d, a, b, c): self._alu("IMAD", d, a, b, c)
    def shl(self, d, a, b): self._alu("SHL", d, a, b)
    def shr(self, d, a, b): self._alu("SHR", d, a, b)
    def and_(self, d, a, b): self._alu("AND", d, a, b)
    def or_(self, d, a, b): self._alu("OR", d, a, b)
    def xor(self, d, a, b): self._alu("XOR", d, a, b)
    def setp_lt(self, d, a, b): self._alu("SETP_LT", d, a, b)
    def setp_eq(self, d, a, b): self._alu("SETP_EQ", d, a, b)
    def setp_ge(self, d, a, b): self._alu("SETP_GE", d, a, b)
    def fadd(self, d, a, b): self._alu("FADD", d, a, b)
    def fmul(self, d, a, b): self._alu("FMUL", d, a, b)
    def ffma(self, d, a, b, c): self._alu("FFMA", d, a, b, c)
    def shfl_xor(self, d, a, mask): self._alu("SHFL_XOR", d, a, mask)

    def ldc64(self, d, name):
        """64-bit kernel parameter into the register pair d, d+1."""
        assert d % 2 == 0
        self._emit("LDC64", writes=(int(d), int(d) + 1), d=d, name=name)

    def iadd64(self, d, a, b):
        """d(pair) = a(pair) + b, b an unsigned 32-bit register or immediate."""
        assert d % 2 == 0 and a % 2 == 0
        self._emit("IADD64", deps=(int(a), int(a) + 1) + _regs(b), writes=(int(d), int(d) + 1), d=d, a=a, b=b)

    def imad_wide(self, d, a, b, c=None):
        """d(pair) = a * b (unsigned 32 x 32 -> 64) + c(pair) -- SASS IMAD.WIDE.U32."""
        assert d % 2 == 0 and (c is None or c % 2 == 0)
        cd = () if c is None else (int(c), int(c) + 1)
        self._emit("IMAD_WIDE", deps=_regs(a, b) + cd, writes=(int(d), int(d) + 1), d=d, a=a, b=b, c=c)

    def s2r(self, d, name):
        """name: tid, laneid, warpid, ctaid, nctaid, ntid, smid, clusterid, clusterrank, clustersize, clock"""
        self._emit("S2R", writes=_regs(d), d=d, name=name)

    def ldc(self, d, name):
        """Kernel parameter by name."""
        self._emit("LDC", writes=_regs(d), d=d, name=name)

    # ---- memory ----
    def ldg(self, d, addr, imm=0, width=16, cache="ca", wide=False, l2pf=0, sem="weak"):
        """d <- global[addr + imm], width bytes per lane into width/4 consecutive regs.
        cache='cs' streams past L1 (ld.global.nc.L1::no_allocate). l2pf=64/128/256 is the
        `.L2::256B` prefetch qualifier: an L2 miss also pulls in the rest of that aligned window."""
        assert l2pf in (0, 64, 128, 256) and sem in ("weak", "relaxed", "acquire")
        # sem: "relaxed" / "acquire" (gpu scope) read L2 (bypass L1); acquire also invalidates this SM's L1
        # and holds back this warp's later memory operations until it returns.
        n = width // 4
        assert d % (n if n in (2, 4) else 1) == 0, "vector destination must be aligned"
        self._emit("LDG", deps=_wdeps(addr, wide), writes=_span(d, n), d=d, a=addr, imm=imm, width=width,
                   cache=cache, wide=wide, opt=f"pf{l2pf}" if l2pf else "", sem=sem)

    def stg(self, addr, src, imm=0, width=4, wide=False, sem="weak"):
        """sem='release' (st.release.gpu): waits until this warp's earlier stores are visible and its
        earlier loads have returned before the store goes out."""
        assert sem in ("weak", "relaxed", "release")
        n = width // 4
        self._emit("STG", deps=_wdeps(addr, wide) + _span(src, n), a=addr, b=src, imm=imm, width=width, wide=wide,
                   sem=sem)

    def red_add(self, addr, src, imm=0, wide=False):
        self._emit("RED_ADD_F32", deps=_wdeps(addr, wide) + _regs(src), a=addr, b=src, imm=imm, wide=wide)

    def lds(self, d, addr, imm=0, width=4):
        n = width // 4
        self._emit("LDS", deps=_regs(addr), writes=_span(d, n), d=d, a=addr, imm=imm, width=width)

    def sts(self, addr, src, imm=0, width=4):
        n = width // 4
        self._emit("STS", deps=_regs(addr) + _span(src, n), a=addr, b=src, imm=imm, width=width)

    def bar(self, id=0, count=0):
        """bar.sync id, count: wait until `count` threads (default: the whole CTA) have arrived."""
        self._emit("BAR", imm=id, extra=count)

    def bar_arrive(self, id, count):
        """bar.arrive: signal arrival on barrier `id` without waiting."""
        self._emit("BAR_ARRIVE", imm=id, extra=count)

    # ---- async copy, mbarrier, TMA ----
    def ldgsts(self, saddr, gaddr, size=16, cache="cg", wide=False):
        """cp.async: per-lane global -> shared copy; lands later, tracked by commit/wait groups.
        cache='cg' bypasses L1 (16 B only), 'ca' allocates in L1."""
        self._emit("LDGSTS", deps=_regs(saddr) + _wdeps(gaddr, wide), a=saddr, b=gaddr, width=size, cache=cache,
                   wide=wide)

    def cp_commit(self): self._emit("CP_COMMIT")
    def cp_wait(self, n=0): self._emit("CP_WAIT", imm=n)

    def mbar_init(self, addr, count): self._emit("MBAR_INIT", deps=_regs(addr), a=addr, imm=count)
    def mbar_arrive(self, addr): self._emit("MBAR_ARRIVE", deps=_regs(addr), a=addr)

    def mbar_arrive_tx(self, addr, nbytes):
        """mbarrier.arrive.expect_tx: one arrival plus `nbytes` of expected async traffic."""
        self._emit("MBAR_ARRIVE_TX", deps=_regs(addr), a=addr, imm=nbytes)

    def mbar_arrive_cl(self, addr, rank):
        """Arrive on the mbarrier at the same smem offset in cluster CTA `rank`."""
        self._emit("MBAR_ARRIVE_CL", deps=_regs(addr, rank), a=addr, b=rank)

    def mbar_wait(self, addr, parity):
        """try_wait.parity loop: blocks until the phase with this parity has completed."""
        self._emit("MBAR_WAIT", deps=_regs(addr, parity), a=addr, b=parity)

    def fence(self):
        """fence.proxy.async: make generic-proxy shared-memory writes visible to TMA stores / wgmma."""
        self._emit("FENCE")

    def membar(self, scope="gpu", sem="sc"):
        """membar / fence.{sc,acq_rel} at 'cta', 'gpu' or 'sys' scope (membar.gl is fence.sc.gpu).
        sc: the warp stalls until its earlier global stores have landed in L2 and its earlier loads have
        returned, so a later load cannot be satisfied before the store is visible (store buffering is forbidden).
        acq_rel: no stall -- later stores are only ordered after earlier ones (they become visible no earlier)
        and later accesses wait for earlier loads; a later load may still read before an earlier store is
        visible, as PTX allows. Both invalidate this SM's L1 (the acquire half)."""
        assert scope in ("cta", "gpu", "sys") and sem in ("acq_rel", "sc")
        self._emit("MEMBAR", opt=scope, sem=sem)

    def tma_load_1d(self, smem, gaddr, nbytes, mbar, mcast=0, wide=False):
        """cp.async.bulk global -> shared, completing `nbytes` on `mbar`. mcast: CTA-rank bit mask."""
        self._emit("TMA_LOAD_1D", deps=_regs(smem, mbar) + _wdeps(gaddr, wide), a=smem, b=gaddr, d=mbar,
                   imm=nbytes, extra=mcast, wide=wide)

    def tma_load_2d(self, smem, tmap, col, row, mbar, mcast=0):
        """cp.async.bulk.tensor.2d: copies the tensor map's box at (row, col); out-of-range is zero."""
        self._emit("TMA_LOAD_2D", deps=_regs(smem, col, row, mbar), a=smem, b=col, c=row, d=mbar, imm=tmap, extra=mcast)

    def tma_store_1d(self, gaddr, smem, nbytes, wide=False):
        """cp.async.bulk shared -> global; completion tracked by cp_commit / cp_wait."""
        self._emit("TMA_STORE_1D", deps=_regs(smem) + _wdeps(gaddr, wide), a=gaddr, b=smem, imm=nbytes, wide=wide)

    def tma_store_2d(self, smem, tmap, col, row, reduce=False):
        """cp.async.bulk.tensor.2d shared -> global (clipped to the tensor). reduce=True is
        cp.reduce.async.bulk with .add on fp32 tensors (split-K accumulation in L2)."""
        self._emit("TMA_STORE_2D", deps=_regs(smem, col, row), a=smem, b=col, c=row, imm=tmap, extra=int(reduce))

    def tma_load_nd(self, smem, tmap, coords, mbar, mcast=0):
        """Tensor load of any rank 1-5; coords are registers / immediates, fastest dimension first."""
        self._emit("TMA_LOAD_ND", deps=_regs(smem, mbar, *coords), a=smem, d=mbar, imm=tmap, extra=mcast,
                   coords=tuple(coords))

    def tma_load_im2col(self, smem, tmap, coords, offsets, mbar, mcast=0):
        """cp.async.bulk.tensor im2col load: coords = (c, w, h[, d], n) of the first filter base position,
        offsets = filter-tap offsets (ow, oh[, od]) added to every base position."""
        self._emit("TMA_LOAD_IM2COL", deps=_regs(smem, mbar, *coords, *offsets), a=smem, d=mbar, imm=tmap,
                   extra=mcast, coords=tuple(coords), offs=tuple(offsets))

    def tma_load_im2col_w(self, smem, tmap, coords, mbar, mode="w", whalo=0, woffset=0, mcast=0):
        """im2col::w (Pixels-per-Column pixels along W, then `whalo` halo pixels), im2col::w::128 (128 pixels with
        `whalo` halo pixels after every 32) or im2col_no_offs::w (no halo / offset, coordinates must be
        non-negative). H and D stay fixed at the coordinates; `woffset` shifts the first pixel along W.
        Returns the number of rows (pixels) the box holds, which is what the mbarrier must expect."""
        assert mode in ("w", "w128", "w_no_offs")
        assert mode != "w_no_offs" or (whalo == 0 and woffset == 0), "im2col_no_offs::w has no wHalo / wOffset"
        self._emit("TMA_LOAD_IM2COL", deps=_regs(smem, mbar, *coords), a=smem, d=mbar, imm=tmap, extra=mcast,
                   coords=tuple(coords), opt=mode, lda=whalo, ldb=woffset)

    def tma_store_nd(self, smem, tmap, coords, reduce=False):
        self._emit("TMA_STORE_ND", deps=_regs(smem, *coords), a=smem, imm=tmap, extra=int(reduce),
                   coords=tuple(coords))

    def tma_prefetch_2d(self, tmap, col, row):
        """cp.async.bulk.prefetch.tensor: pull the box into L2, nothing lands in shared memory."""
        self._emit("TMA_PREFETCH_2D", deps=_regs(col, row), b=col, c=row, imm=tmap)

    # ---- warpgroup MMA ----
    def wgmma(self, d, a, b, n, lda, ldb, accumulate=True, swizzle=0, a_in_regs=False,
              dtype="bf16", acc="f32", sparse=False, meta=None, sel=0, a_mn=False, b_mn=False):
        """wgmma.mma_async m64nNkK. A: 64 x K at smem[a], row stride lda bytes. B: N x K (K-major)
        at smem[b], row stride ldb bytes. D: N/2 registers per thread (N/4 when accumulating in
        f16), in the mma C-fragment layout, one 8-column block at a time. Issued by all 4 warps of
        the warpgroup; warp i computes rows 16i..16i+15. Results land asynchronously: wgmma_commit
        then wgmma_wait before reading d.
        swizzle: 0/32/64/128, must match the TMA tensor map that filled the tile.
        a_in_regs: A is the 16 x K slice in 4 registers a..a+3 (mma.sync A-fragment layout) and
        lda is ignored; the registers are read at issue.
        dtype/acc: see hmma. K per instruction is 16 (f16/bf16), 8 (tf32), 32 (fp8/int8).
        sparse: A is structured-sparse (2:4, or 1:2 for tf32), K/2 values stored, the effective K doubles and
        B spans the full 2K. `meta` is the register holding each lane's metadata word and `sel` the sparsity
        selector, exactly as for mma.sp (each warp supplies the metadata of its own 16 rows): 16-bit and tf32
        types use a thread pair (sel 0/1), 8-bit types all four threads (sel 0).
        a_mn / b_mn: the operand is stored MN-major (transposed): K lines of stride lda / ldb, each
        holding the 64 (A) or N (B) contiguous elements. Only 16-bit types support this."""
        assert not (a_mn or b_mn) or T.DTYPES[dtype].bits == 16, "transposed operands need 16-bit types"
        assert acc in T.ACC_OK[dtype], f"{dtype} cannot accumulate in {acc}"
        assert n % 8 == 0 and 8 <= n <= 256
        assert (meta is not None) == sparse, "sparse wgmma needs a metadata address"
        if a_in_regs:
            assert a % 4 == 0
        self._emit("WGMMA", deps=(_span(a, 4) if a_in_regs else _regs(a)) + _regs(b, meta), d=d, a=a, b=b,
                   c=meta, imm=n, lda=lda, ldb=ldb, extra=int(accumulate), swz=swizzle, a_regs=a_in_regs,
                   dtype=dtype, acc=acc, sparse=sparse, sel=sel, opt=("a" if a_mn else "") + ("b" if b_mn else ""))

    def wgmma_fence(self): self._emit("WGMMA_FENCE")
    def wgmma_commit(self): self._emit("WGMMA_COMMIT")
    def wgmma_wait(self, n=0): self._emit("WGMMA_WAIT", imm=n)

    # ---- clusters / distributed shared memory ----
    def lds_cl(self, d, addr, rank, imm=0, width=4):
        n = width // 4
        self._emit("LDS_CL", deps=_regs(addr, rank), writes=_span(d, n), d=d, a=addr, c=rank, imm=imm, width=width)

    def sts_cl(self, addr, src, rank, imm=0, width=4):
        n = width // 4
        self._emit("STS_CL", deps=_regs(addr, rank) + _span(src, n), a=addr, b=src, c=rank, imm=imm, width=width)

    def cluster_sync(self): self._emit("CLUSTER_SYNC")

    def hmma_sp(self, d, a, b, c, meta, sel=0, dtype="bf16", acc="f32", k=None):
        """mma.sp: A is 2:4 compressed (K/2 stored), B is dense K x 8. `k` is the effective (dense)
        K: 16 or 32 for f16 / bf16, 8 or 16 for tf32, 32 or 64 for 8-bit types. `meta` is a register
        holding each lane's metadata word; `sel` is the sparsity selector (which lane of a quad
        supplies it). The metadata bit layout is reconstructed, see tensor.py."""
        dt = T.DTYPES[dtype]
        k = k or 2 * dt.k_mma
        assert acc in T.ACC_OK[dtype]
        epr = 32 // dt.bits
        na, nb, nc = (k // 2) // (2 * epr), k // (4 * epr), 2 if acc == "f16" else 4
        assert na >= 2 and d % 2 == 0 and a % min(na, 4) == 0 and b % min(nb, 2) == 0 and c % 2 == 0
        ins = self._emit("HMMA", deps=_span(a, na) + _span(b, nb) + _span(c, nc) + _regs(meta), writes=_span(d, nc),
                         d=d, a=a, b=b, c=c, dtype=dtype, acc=acc, k=k, sparse=True, meta=meta, extra=sel)
        ins.lat, ins.occ = T.mma_sp_timing(dtype, k, acc)

    def dmma(self, d, a, b, c):
        """mma.sync.m8n8k4 fp64: D(8x8) = A(8x4) x B(4x8) + C. Each lane holds one double of A
        (row l/4, col l%4) in a, one of B (row l%4, col l/4) in b, and two of C / D (row l/4,
        cols 2(l%4), 2(l%4)+1) in c / d. All are register pairs."""
        assert d % 2 == 0 and a % 2 == 0 and b % 2 == 0 and c % 2 == 0
        self._emit("DMMA", deps=_span(a, 2) + _span(b, 2) + _span(c, 4), writes=_span(d, 4), d=d, a=a, b=b, c=c)

    # ---- tensor core ----
    def hmma(self, d, a, b, c, dtype="bf16", acc="f32", k=None):
        """mma.sync m16n8kK: D = A x B + C. dtype: f16 bf16 (k 8/16), tf32 (k 4/8), e4m3 e5m2 s8 u8
        (k 16/32); acc: f32, f16 (f16/fp8 inputs only) or s32 (int8). Register counts follow the
        PTX fragments: A = K/(2*epr) regs, B = K/(4*epr), C/D = 4 (2 packed-f16 when acc is f16)."""
        dt = T.DTYPES[dtype]
        k = k or dt.k_mma
        assert acc in T.ACC_OK[dtype], f"{dtype} cannot accumulate in {acc}"
        epr = 32 // dt.bits
        na, nb, nc = k // (2 * epr), k // (4 * epr), 2 if acc == "f16" else 4
        assert na >= 2 and d % 2 == 0 and a % min(na, 4) == 0 and b % min(nb, 2) == 0 and c % 2 == 0
        ins = self._emit("HMMA", deps=_span(a, na) + _span(b, nb) + _span(c, nc), writes=_span(d, nc),
                         d=d, a=a, b=b, c=c, dtype=dtype, acc=acc, k=k)
        ins.lat, ins.occ = T.mma_timing(dtype, k, acc)

    # ---- control ----
    def bra(self, label, reconv=None): self._emit("BRA", target=label, reconv=reconv)
    def brnz(self, pred, label, reconv=None):
        """Branch if pred != 0. If lanes disagree, `reconv` (a label) is where they rejoin."""
        self._emit("BRNZ", deps=_regs(pred), a=pred, target=label, reconv=reconv)
    def exit(self): self._emit("EXIT")

    # ---- double precision and packed 16-bit ----
    def dadd(self, d, a, b): self._gen("DADD", d, a, b, dw=2, aw=2, bw=2)
    def dmul(self, d, a, b): self._gen("DMUL", d, a, b, dw=2, aw=2, bw=2)
    def dfma(self, d, a, b, c): self._gen("DFMA", d, a, b, c, dw=2, aw=2, bw=2, cw=2)
    def hadd2(self, d, a, b, bf16=False): self._gen("HADD2_BF16" if bf16 else "HADD2", d, a, b)
    def hmul2(self, d, a, b, bf16=False): self._gen("HMUL2_BF16" if bf16 else "HMUL2", d, a, b)
    def hfma2(self, d, a, b, c, bf16=False): self._gen("HFMA2_BF16" if bf16 else "HFMA2", d, a, b, c)

    # ---- conversions and special functions ----
    def i2f(self, d, a, unsigned=False): self._gen("I2F", d, a, opt="u" if unsigned else "s")
    def f2i(self, d, a, round="rz"): self._gen("F2I", d, a, opt=round)

    def f2fp(self, d, a, b, fmt="bf16"):
        """cvt.rn.<fmt>x2.f32: pack fp32 a (high half) and b (low half) into two 16-bit (bf16/f16)
        or 8-bit (e4m3/e5m2, satfinite, in the low 16 bits) values."""
        self._gen("F2FP_" + fmt.upper(), d, a, b)

    def cvt_f16x2_fp8(self, d, a, fmt="e4m3"):
        """cvt.rn.f16x2.e4m3x2 / e5m2x2: unpack two fp8 values from the low 16 bits to f16x2."""
        self._gen("CVT_F16X2_" + fmt.upper(), d, a)

    def mufu(self, fn, d, a):
        """fn: rcp rsq sqrt ex2 lg2 sin cos tanh."""
        self._gen("MUFU_" + fn.upper(), d, a)

    # ---- integer / float misc, DPX ----
    def imnmx(self, d, a, b, op="min", unsigned=False): self._gen("IMNMX", d, a, b, opt=op + ("u" if unsigned else "s"))
    def fmnmx(self, d, a, b, op="min"): self._gen("FMNMX", d, a, b, opt=op)

    def vimnmx3(self, d, a, b, c, op="min", fmt="s32"):
        """DPX: three-input min/max. fmt: s32 u32 s16x2 u16x2."""
        self._gen("VIMNMX3", d, a, b, c, opt=op + "." + fmt)

    def viaddmnmx(self, d, a, b, c, op="max", fmt="s32", relu=False):
        """DPX: min/max(a + b, c), optionally clamped at zero (the dynamic-programming recurrence)."""
        self._gen("VIADDMNMX", d, a, b, c, opt=op + "." + fmt + (".relu" if relu else ""))

    def prmt(self, d, a, b, sel): self._gen("PRMT", d, a, b, sel)
    def lop3(self, d, a, b, c, lut): self._gen("LOP3", d, a, b, c, imm=lut)
    def shf(self, d, a, b, n, left=True):
        """Funnel shift {b:a} (b the high word) by the immediate n: left gives the high word, right the low."""
        self._gen("SHF", d, a, b, imm=n, opt="l" if left else "r")
    def popc(self, d, a): self._gen("POPC", d, a)
    def brev(self, d, a): self._gen("BREV", d, a)
    def flo(self, d, a): self._gen("FLO", d, a)
    def iabs(self, d, a): self._gen("IABS", d, a)
    def imul_hi(self, d, a, b, unsigned=True): self._gen("IMUL_HI", d, a, b, opt="u" if unsigned else "s")
    def sel(self, d, a, b, p): self._gen("SEL", d, a, b, p)

    def isetp(self, cmp, p, a, b, unsigned=False):
        """p = (a cmp b) per lane; cmp: lt le gt ge eq ne."""
        self._gen("ISETP", p, a, b, opt=cmp + ("u" if unsigned else "s"))

    def fsetp(self, cmp, p, a, b): self._gen("FSETP", p, a, b, opt=cmp)
    def psetp(self, op, p, q, r=PT): self._gen("PSETP", p, q, r, opt=op)

    # ---- warp-level ----
    def shfl_up(self, d, a, delta, width=32): self._gen("SHFL_UP", d, a, delta, imm=width)
    def shfl_down(self, d, a, delta, width=32): self._gen("SHFL_DOWN", d, a, delta, imm=width)
    def shfl_idx(self, d, a, lane, width=32): self._gen("SHFL_IDX", d, a, lane, imm=width)
    def vote_ballot(self, d, p): self._gen("VOTE_BALLOT", d, p)
    def vote_any(self, d, p): self._gen("VOTE_ANY", d, p)
    def vote_all(self, d, p): self._gen("VOTE_ALL", d, p)
    def redux(self, d, a, op="add"): self._gen("REDUX", d, a, opt=op)
    def elect(self, p): self._gen("ELECT", p)
    def activemask(self, d): self._gen("ACTIVEMASK", d)
    def nanosleep(self, cycles):
        """Suspend the warp for about `cycles` (an immediate or a register whose first lane is used)."""
        if isinstance(cycles, (Reg, UReg)):
            self._emit("NANOSLEEP", deps=_regs(cycles), a=cycles)
        else:
            self._emit("NANOSLEEP", imm=cycles)

    # ---- atomics, constant memory, matrix loads ----
    def atom(self, d, addr, src, op="add", fmt="u32", cmp=None, imm=0, wide=False):
        """Global atomic on a 32-bit word, returns the old value in d (d may be None to discard).
        op: add min max and or xor exch cas (cas compares with `cmp`); fmt: u32 s32 f32."""
        deps = _wdeps(addr, wide) + _regs(src, cmp)
        self._emit("ATOM", deps=deps, writes=_regs(d) if d is not None else (), d=d, a=addr, b=src, c=cmp,
                   imm=imm, wide=wide, opt=op + "." + fmt)

    def atoms(self, d, addr, src, op="add", fmt="u32", cmp=None, imm=0):
        """Shared-memory atomic (same ops as atom)."""
        self._emit("ATOMS", deps=_regs(addr, src, cmp), writes=_regs(d) if d is not None else (), d=d, a=addr,
                   b=src, c=cmp, imm=imm, opt=op + "." + fmt)

    def ldc_mem(self, d, addr, imm=0):
        """Load a word from the 64 KB constant bank (Sim.write_constant). Lanes reading different
        addresses are serialized."""
        self._gen("LDC_MEM", d, addr, imm=imm)

    def ldmatrix(self, d, addr, num=4, trans=False):
        """ldmatrix.x{num}[.trans]: lanes 8i..8i+7 give the 16 B row addresses of 8x8 b16 matrix i;
        thread l gets row l/4, columns 2(l%4), 2(l%4)+1 of each matrix into d..d+num-1."""
        assert num in (1, 2, 4)
        self._emit("LDMATRIX", deps=_regs(addr), writes=_span(d, num), d=d, a=addr, imm=num, opt="t" if trans else "")

    def stmatrix(self, addr, src, num=4, trans=False):
        assert num in (1, 2, 4)
        self._emit("STMATRIX", deps=_regs(addr) + _span(src, num), a=addr, b=src, imm=num, opt="t" if trans else "")

    def griddep_launch(self):
        """griddepcontrol.launch_dependents: this CTA allows the next (PDL) kernel to be scheduled."""
        self._emit("GRIDDEP_LAUNCH")

    def griddep_wait(self):
        """griddepcontrol.wait: block until the previous kernel in the chain has completed and its
        writes are visible."""
        self._emit("GRIDDEP_WAIT")

    def setmaxnreg(self, n, inc):
        """setmaxnreg.inc / .dec: move this warpgroup's per-thread register budget to n (multiple of 8,
        24..256). dec returns registers to the CTA's pool, inc waits until the pool holds them."""
        assert n % 8 == 0 and 24 <= n <= 256
        self._emit("SETMAXNREG", imm=n, extra=int(inc))

    # ---- uniform datapath ----
    def umov(self, d, a): self._gen("UMOV", d, a)
    def uiadd(self, d, a, b): self._gen("UIADD", d, a, b)
    def uimad(self, d, a, b, c): self._gen("UIMAD", d, a, b, c)
    def ushl(self, d, a, b): self._gen("USHL", d, a, b)
    def ushr(self, d, a, b): self._gen("USHR", d, a, b)
    def uand(self, d, a, b): self._gen("UAND", d, a, b)
    def uor(self, d, a, b): self._gen("UOR", d, a, b)
    def uisetp(self, cmp, d, a, b): self._gen("UISETP", d, a, b, opt=cmp)
    def s2ur(self, d, name): self._gen("S2UR", d, name=name)
    def uldc(self, d, name): self._gen("ULDC", d, name=name)
    def r2ur(self, d, a): self._gen("R2UR", d, a)
    def ubrnz(self, pred, label): self._emit("BRNZ", deps=_regs(pred), a=pred, target=label)

    def build(self, launch_regs=None) -> Program:
        for ins in self.code:
            if isinstance(ins.target, str):
                ins.target = self.labels[ins.target]
            if isinstance(ins.reconv, str):
                ins.reconv = self.labels[ins.reconv]
        if not self.code or self.code[-1].op != "EXIT":
            self.exit()
        n = max(8, (self.nxt + 7) // 8 * 8)
        for ins in self.code:                       # highest register each instruction touches
            ins.maxreg = max([x + 1 for x in ins.deps if x < UBASE] or [0])
        return Program(self.code, n, self.name, launch_regs or n)
