"""Semantics of the extended op set: FP64, packed 16-bit math, conversions, MUFU, min/max and
DPX, bit ops, predicates, warp votes / shuffles / reductions, atomics, constant memory,
ldmatrix / stmatrix and the uniform datapath.

Each function takes (sm, warp, instr, cycle, sim) and returns None (result latency from the
op table) or (done_cycle, is_memory_result).
"""

import math

import numpy as np

from . import tensor as T
from .isa import Pred, Reg, UReg
from .memory import to_bf16_bits
from .regs import LANES, _f, _gaddr, _s32, _swz, _u, _wr, _wr64, pbool

EXTRA = {}


def op(*names):
    def deco(fn):
        for n in names:
            EXTRA[n] = fn
        return fn
    return deco


# ---------------------------------------------------------------- FP64

def _d(w, x):
    if isinstance(x, float):
        return np.full(32, x, np.float64)
    return (w.regs[x].astype(np.uint64) | (w.regs[x + 1].astype(np.uint64) << np.uint64(32))).view(np.float64)


def _wd(w, d, v):
    _wr64(w, d, np.ascontiguousarray(v, np.float64).view(np.uint64))


@op("DADD")
def _dadd(sm, w, ins, cycle, sim): _wd(w, ins.d, _d(w, ins.a) + _d(w, ins.b))
@op("DMUL")
def _dmul(sm, w, ins, cycle, sim): _wd(w, ins.d, _d(w, ins.a) * _d(w, ins.b))
@op("DFMA")
def _dfma(sm, w, ins, cycle, sim): _wd(w, ins.d, _d(w, ins.a) * _d(w, ins.b) + _d(w, ins.c))


# ---------------------------------------------------------------- packed 16-bit

def _h2(w, x, bf):
    u = _u(w, x)
    if bf:
        return (u << 16).view(np.float32), (u & 0xFFFF0000).view(np.float32)
    return ((u & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32),
            (u >> 16).astype(np.uint16).view(np.float16).astype(np.float32))


def _pack_h2(lo, hi, bf):
    if bf:
        return to_bf16_bits(lo).astype(np.uint32) | (to_bf16_bits(hi).astype(np.uint32) << 16)
    return (lo.astype(np.float16).view(np.uint16).astype(np.uint32)
            | (hi.astype(np.float16).view(np.uint16).astype(np.uint32) << 16))


def _h2op(fn):
    def run(sm, w, ins, cycle, sim):
        bf = ins.op.endswith("BF16")
        ops_ = [_h2(w, x, bf) for x in (ins.a, ins.b, ins.c) if x is not None]
        with np.errstate(all="ignore"):
            lo = fn(*[o[0].astype(np.float64) for o in ops_])      # one rounding at the end, like an FMA
            hi = fn(*[o[1].astype(np.float64) for o in ops_])
        _wr(w, ins.d, _pack_h2(lo.astype(np.float32), hi.astype(np.float32), bf))
    return run


for _n, _f_ in (("HADD2", lambda a, b: a + b), ("HMUL2", lambda a, b: a * b), ("HFMA2", lambda a, b, c: a * b + c)):
    EXTRA[_n] = EXTRA[_n + "_BF16"] = _h2op(_f_)


# ---------------------------------------------------------------- conversions, MUFU

@op("I2F")
def _i2f(sm, w, ins, cycle, sim):
    u = _u(w, ins.a)
    _wr(w, ins.d, (u if ins.opt == "u" else u.view(np.int32)).astype(np.float32).view(np.uint32))


@op("F2I")
def _f2i(sm, w, ins, cycle, sim):
    x = _f(w, ins.a).astype(np.float64)
    x = np.rint(x) if ins.opt == "rn" else np.trunc(x)
    x = np.clip(np.nan_to_num(x, nan=0.0), -2.0 ** 31, 2.0 ** 31 - 1)
    _wr(w, ins.d, x.astype(np.int32).view(np.uint32))


def _f2fp(fmt):
    def run(sm, w, ins, cycle, sim):
        a, b = _f(w, ins.a), _f(w, ins.b)
        if fmt == "bf16":
            r = (to_bf16_bits(a).astype(np.uint32) << 16) | to_bf16_bits(b).astype(np.uint32)
        elif fmt == "f16":
            r = _pack_h2(b, a, False)
        else:
            lut = T.E4M3 if fmt == "e4m3" else T.E5M2
            r = (T.encode_fp8(a, lut).astype(np.uint32) << 8) | T.encode_fp8(b, lut).astype(np.uint32)
        _wr(w, ins.d, r)
    return run


for _fmt in ("bf16", "f16", "e4m3", "e5m2"):
    EXTRA["F2FP_" + _fmt.upper()] = _f2fp(_fmt)


def _fp8x2_to_f16x2(lut):
    def run(sm, w, ins, cycle, sim):
        u = _u(w, ins.a)
        lo, hi = lut[(u & 0xFF).astype(np.uint8)], lut[((u >> 8) & 0xFF).astype(np.uint8)]
        _wr(w, ins.d, _pack_h2(lo, hi, False))
    return run


EXTRA["CVT_F16X2_E4M3"] = _fp8x2_to_f16x2(T.E4M3)
EXTRA["CVT_F16X2_E5M2"] = _fp8x2_to_f16x2(T.E5M2)

_MUFU = {"RCP": lambda x: 1.0 / x, "RSQ": lambda x: 1.0 / np.sqrt(x), "SQRT": np.sqrt, "EX2": np.exp2,
         "LG2": np.log2, "SIN": np.sin, "COS": np.cos, "TANH": np.tanh}


def _mufu(fn):
    def run(sm, w, ins, cycle, sim):
        with np.errstate(all="ignore"):
            _wr(w, ins.d, fn(_f(w, ins.a)).astype(np.float32).view(np.uint32))
    return run


for _n, _fn in _MUFU.items():
    EXTRA["MUFU_" + _n] = _mufu(_fn)


# ---------------------------------------------------------------- min/max, DPX, bit ops

@op("IMNMX")
def _imnmx(sm, w, ins, cycle, sim):
    uns = ins.opt.endswith("u")
    a, b = (_u(w, ins.a), _u(w, ins.b)) if uns else (_u(w, ins.a).view(np.int32), _u(w, ins.b).view(np.int32))
    r = np.minimum(a, b) if ins.opt.startswith("min") else np.maximum(a, b)
    _wr(w, ins.d, r.view(np.uint32) if not uns else r)


@op("FMNMX")
def _fmnmx(sm, w, ins, cycle, sim):
    f = np.fmin if ins.opt == "min" else np.fmax
    _wr(w, ins.d, f(_f(w, ins.a), _f(w, ins.b)).view(np.uint32))


def _split(u, fmt):
    """Operand lanes as arrays of int64 'elements' (1 for 32-bit formats, 2 for x2 formats)."""
    if fmt == "s32":
        return [u.view(np.int32).astype(np.int64)]
    if fmt == "u32":
        return [u.astype(np.int64)]
    if fmt == "s16x2":
        return [(u & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64),
                (u >> 16).astype(np.uint16).view(np.int16).astype(np.int64)]
    return [(u & 0xFFFF).astype(np.int64), (u >> 16).astype(np.int64)]


def _join(parts, fmt):
    if fmt in ("s32", "u32"):
        return (parts[0] & 0xFFFFFFFF).astype(np.uint32)
    return ((parts[0] & 0xFFFF).astype(np.uint32)) | ((parts[1] & 0xFFFF).astype(np.uint32) << 16)


@op("VIMNMX3")
def _vimnmx3(sm, w, ins, cycle, sim):
    mm, fmt = ins.opt.split(".")
    f = np.minimum if mm == "min" else np.maximum
    xs = [_split(_u(w, x), fmt) for x in (ins.a, ins.b, ins.c)]
    _wr(w, ins.d, _join([f(f(xs[0][i], xs[1][i]), xs[2][i]) for i in range(len(xs[0]))], fmt))


@op("VIADDMNMX")
def _viaddmnmx(sm, w, ins, cycle, sim):
    parts = ins.opt.split(".")
    mm, fmt, relu = parts[0], parts[1], len(parts) > 2
    f = np.minimum if mm == "min" else np.maximum
    xa, xb, xc = (_split(_u(w, x), fmt) for x in (ins.a, ins.b, ins.c))
    out = []
    for i in range(len(xa)):
        r = f(xa[i] + xb[i], xc[i])
        out.append(np.maximum(r, 0) if relu else r)
    _wr(w, ins.d, _join(out, fmt))


@op("PRMT")
def _prmt(sm, w, ins, cycle, sim):
    a, b, sel = _u(w, ins.a), _u(w, ins.b), _u(w, ins.c)
    pool = np.stack([(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)])
    out = np.zeros(32, np.uint32)
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        byte = pool[(s & 7).astype(int), LANES]
        byte = np.where(s & 8, np.where(byte & 0x80, 0xFF, 0), byte)        # sign-replicate mode
        out |= byte.astype(np.uint32) << (8 * i)
    _wr(w, ins.d, out)


@op("LOP3")
def _lop3(sm, w, ins, cycle, sim):
    a, b, c = _u(w, ins.a), _u(w, ins.b), _u(w, ins.c)
    out = np.zeros(32, np.uint32)
    for i in range(8):
        if (ins.imm >> i) & 1:
            out |= (a if i & 4 else ~a) & (b if i & 2 else ~b) & (c if i & 1 else ~c)
    _wr(w, ins.d, out)


@op("SHF")
def _shf(sm, w, ins, cycle, sim):
    """SHF.L.W / SHF.R.W: funnel shift of the 64-bit pair {b:a} (b high), shift count wrapped to 5 bits."""
    a, b = _u(w, ins.a).astype(np.uint64), _u(w, ins.b).astype(np.uint64)
    n = np.uint64(int(ins.imm) & 31)
    v = (b << np.uint64(32)) | a
    r = (v << n) >> np.uint64(32) if ins.opt == "l" else v >> n
    _wr(w, ins.d, (r & np.uint64(0xFFFFFFFF)).astype(np.uint32))


@op("POPC")
def _popc(sm, w, ins, cycle, sim):
    u = _u(w, ins.a)
    _wr(w, ins.d, np.array([bin(int(x)).count("1") for x in u], np.uint32))


@op("BREV")
def _brev(sm, w, ins, cycle, sim):
    u = _u(w, ins.a)
    _wr(w, ins.d, np.array([int(f"{int(x):032b}"[::-1], 2) for x in u], np.uint32))


@op("FLO")
def _flo(sm, w, ins, cycle, sim):
    u = _u(w, ins.a)
    _wr(w, ins.d, np.array([int(x).bit_length() - 1 & 0xFFFFFFFF for x in u], np.uint32))


@op("IABS")
def _iabs(sm, w, ins, cycle, sim):
    _wr(w, ins.d, np.abs(_u(w, ins.a).view(np.int32).astype(np.int64)).astype(np.uint32))


@op("SEL")
def _sel(sm, w, ins, cycle, sim):
    _wr(w, ins.d, np.where(pbool(w, ins.c), _u(w, ins.a), _u(w, ins.b)))


@op("IMUL_HI")
def _imul_hi(sm, w, ins, cycle, sim):
    if ins.opt == "u":
        r = (_u(w, ins.a).astype(np.uint64) * _u(w, ins.b).astype(np.uint64)) >> np.uint64(32)
    else:
        r = ((_u(w, ins.a).view(np.int32).astype(np.int64) * _u(w, ins.b).view(np.int32).astype(np.int64)) >> 32)
    _wr(w, ins.d, (r & 0xFFFFFFFF).astype(np.uint32))


_CMP = {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
        "eq": np.equal, "ne": np.not_equal}


@op("ISETP")
def _isetp(sm, w, ins, cycle, sim):
    cmp, uns = ins.opt[:-1], ins.opt[-1] == "u"
    a, b = (_u(w, ins.a), _u(w, ins.b)) if uns else (_u(w, ins.a).view(np.int32), _u(w, ins.b).view(np.int32))
    _wr(w, ins.d, _CMP[cmp](a, b).astype(np.uint32))


@op("FSETP")
def _fsetp(sm, w, ins, cycle, sim):
    _wr(w, ins.d, _CMP[ins.opt](_f(w, ins.a), _f(w, ins.b)).astype(np.uint32))


@op("PSETP")
def _psetp(sm, w, ins, cycle, sim):
    q, r = pbool(w, ins.a), pbool(w, ins.b)
    v = {"and": q & r, "or": q | r, "xor": q ^ r, "andn": q & ~r, "orn": q | ~r}[ins.opt]
    _wr(w, ins.d, v.astype(np.uint32))


# ---------------------------------------------------------------- warp-level

def _shfl(kind):
    def run(sm, w, ins, cycle, sim):
        src, width = _u(w, ins.a), ins.imm
        arg = _u(w, ins.b).astype(np.int64)
        base = LANES - (LANES % width)
        if kind == "up":
            s = LANES - arg
            ok = (LANES % width) >= arg
        elif kind == "down":
            s = LANES + arg
            ok = (LANES % width) + arg < width
        else:
            s = base + (arg & (width - 1))
            ok = np.ones(32, bool)
        _wr(w, ins.d, np.where(ok, src[np.clip(s, 0, 31)], src))
    return run


EXTRA["SHFL_UP"], EXTRA["SHFL_DOWN"], EXTRA["SHFL_IDX"] = _shfl("up"), _shfl("down"), _shfl("idx")


def _ballot(w, p):
    bits = pbool(w, p) & w.mask
    return int(sum(1 << i for i in np.nonzero(bits)[0]))


@op("VOTE_BALLOT")
def _vote_ballot(sm, w, ins, cycle, sim):
    _wr(w, ins.d, np.full(32, _ballot(w, ins.a), np.uint32))


@op("VOTE_ANY")
def _vote_any(sm, w, ins, cycle, sim):
    _wr(w, ins.d, np.full(32, int(_ballot(w, ins.a) != 0), np.uint32))


@op("VOTE_ALL")
def _vote_all(sm, w, ins, cycle, sim):
    active = int(sum(1 << i for i in np.nonzero(w.mask)[0]))
    _wr(w, ins.d, np.full(32, int(_ballot(w, ins.a) == active), np.uint32))


@op("REDUX")
def _redux(sm, w, ins, cycle, sim):
    v = _u(w, ins.a)[w.mask]
    r = {"add": lambda x: int(x.astype(np.uint64).sum()), "min": lambda x: int(x.min()),
         "max": lambda x: int(x.max()), "mins": lambda x: int(x.view(np.int32).min()),
         "maxs": lambda x: int(x.view(np.int32).max()), "and": lambda x: int(np.bitwise_and.reduce(x)),
         "or": lambda x: int(np.bitwise_or.reduce(x)), "xor": lambda x: int(np.bitwise_xor.reduce(x))}[ins.opt](v)
    _wr(w, ins.d, np.full(32, r & 0xFFFFFFFF, np.uint32))


@op("ELECT")
def _elect(sm, w, ins, cycle, sim):
    first = np.zeros(32, np.uint32)
    first[int(np.nonzero(w.mask)[0][0])] = 1
    _wr(w, ins.d, first)


@op("ACTIVEMASK")
def _activemask(sm, w, ins, cycle, sim):
    _wr(w, ins.d, np.full(32, int(sum(1 << i for i in np.nonzero(w.mask)[0])), np.uint32))


@op("NANOSLEEP")
def _nanosleep(sm, w, ins, cycle, sim):
    n = int(_u(w, ins.a)[0]) if ins.a is not None else ins.imm
    w.wake_at = cycle + max(1, n)


# ---------------------------------------------------------------- uniform datapath

def _uni(fn):
    def run(sm, w, ins, cycle, sim):
        vals = [np.uint32(_u(w, x)[0]) if x is not None else None for x in (ins.a, ins.b, ins.c)]
        with np.errstate(all="ignore"):
            w.uregs[ins.d] = np.uint32(fn(*[int(v) if v is not None else None for v in vals]) & 0xFFFFFFFF)
    return run


EXTRA["UMOV"] = _uni(lambda a, b, c: a)
EXTRA["UIADD"] = _uni(lambda a, b, c: a + b)
EXTRA["UIMAD"] = _uni(lambda a, b, c: a * b + c)
EXTRA["USHL"] = _uni(lambda a, b, c: a << b)
EXTRA["USHR"] = _uni(lambda a, b, c: a >> b)
EXTRA["UAND"] = _uni(lambda a, b, c: a & b)
EXTRA["UOR"] = _uni(lambda a, b, c: a | b)


@op("UISETP")
def _uisetp(sm, w, ins, cycle, sim):
    a, b = (np.int32(np.uint32(_u(w, x)[0]).view(np.int32)) for x in (ins.a, ins.b))
    w.uregs[ins.d] = np.uint32(bool(_CMP[ins.opt](a, b)))


@op("S2UR")
def _s2ur(sm, w, ins, cycle, sim):
    w.uregs[ins.d] = np.uint32(sm._special(w, ins.name)[0])


@op("ULDC")
def _uldc(sm, w, ins, cycle, sim):
    k = w.cta.kernel
    w.uregs[ins.d] = np.uint32(k.params[ins.name] & 0xFFFFFFFF)
    return cycle + sim.frontend.ldc(w.smsp, ("p", k.id, k.param_line(ins.name)), cycle), False


@op("R2UR")
def _r2ur(sm, w, ins, cycle, sim):
    lane = int(np.nonzero(w.mask)[0][0])
    w.uregs[ins.d] = np.uint32(_u(w, ins.a)[lane])


# ---------------------------------------------------------------- atomics, constant memory

def _atomic(old, src, cmp, op_, fmt):
    if fmt == "f32":
        o, s = np.uint32(old).view(np.float32), np.uint32(src).view(np.float32)
        r = {"add": o + s, "min": min(o, s), "max": max(o, s), "exch": s}[op_]
        return int(np.float32(r).view(np.uint32))
    if fmt == "s32":
        o, s = int(np.uint32(old).view(np.int32)), int(np.uint32(src).view(np.int32))
    else:
        o, s = int(old), int(src)
    r = {"add": o + s, "min": min(o, s), "max": max(o, s), "and": o & s, "or": o | s, "xor": o ^ s,
         "exch": s, "cas": s if o == int(cmp if fmt == "u32" else np.uint32(cmp).view(np.int32)) else o}[op_]
    return r & 0xFFFFFFFF


@op("ATOM")
def _atom(sm, w, ins, cycle, sim):
    ms, mem, cfg = sim.memsys, sim.memsys.mem, sim.cfg
    op_, fmt = ins.opt.split(".")
    act = np.nonzero(w.mask)[0]
    addr = _gaddr(w, ins.a, ins.wide)[act] + ins.imm
    off = addr - mem.base
    if (addr % 4).any() or not mem.valid(int(off.min()), int(off.max()) + 4):
        raise RuntimeError(f"illegal atomic at pc={w.pc - 1}")
    src, cmp = _u(w, ins.b), (_u(w, ins.c) if ins.c is not None else None)
    old = np.zeros(32, np.uint32)
    for lane, o in zip(act, off):          # same-address lanes serialize in lane order
        i = int(o) >> 2
        old[lane] = mem.u32[i]
        mem.u32[i] = _atomic(mem.u32[i], src[lane], cmp[lane] if cmp is not None else 0, op_, fmt)
    sectors = np.unique(addr >> 5)
    collisions = int(max(np.unique(off, return_counts=True)[1]))
    xs, ys = zip(*sim.cfg.atomic_global_curve)                     # Volta: latency vs same-address lanes
    t = ms.load(sm, sectors, cycle, False, 4) + int(np.interp(collisions, xs, ys))
    ms.store(sm, sectors, cycle)
    if ins.d is not None:
        _wr(w, ins.d, old)
    return t, True


@op("ATOMS")
def _atoms(sm, w, ins, cycle, sim):
    op_, fmt = ins.opt.split(".")
    s32 = w.cta.smem.view(np.uint32)
    act = np.nonzero(w.mask)[0]
    addr = _u(w, ins.a).astype(np.int64)[act] + ins.imm
    if (addr % 4).any() or addr.min() < 0 or addr.max() + 4 > w.cta.smem.size:
        raise RuntimeError(f"illegal shared atomic at pc={w.pc - 1}")
    src, cmp = _u(w, ins.b), (_u(w, ins.c) if ins.c is not None else None)
    old = np.zeros(32, np.uint32)
    for lane, a in zip(act, addr):
        i = int(a) >> 2
        old[lane] = s32[i]
        s32[i] = _atomic(s32[i], src[lane], cmp[lane] if cmp is not None else 0, op_, fmt)
    collisions = int(max(np.unique(addr, return_counts=True)[1]))
    w.cta.generic_dirty = True
    if ins.d is not None:
        _wr(w, ins.d, old)
    xs, ys = zip(*sim.cfg.atomic_shared_curve)
    return cycle + sim.cfg.lat_smem + int(np.interp(collisions, xs, ys)), True


@op("LDC_MEM")
def _ldc_mem(sm, w, ins, cycle, sim):
    act = w.mask
    addr = _u(w, ins.a).astype(np.int64)[act] + ins.imm
    if (addr % 4).any() or addr.min() < 0 or addr.max() + 4 > sim.constant.size:
        raise RuntimeError(f"illegal constant access at pc={w.pc - 1}")
    vals = sim.constant.view(np.uint32)[addr >> 2]
    out = w.regs[ins.d]
    out[act] = vals
    distinct = np.unique(addr)
    lat = max(sim.frontend.ldc(w.smsp, ("c", int(a) >> 6), cycle) for a in distinct)
    return cycle + lat + 2 * (len(distinct) - 1), True


# ---------------------------------------------------------------- ldmatrix / stmatrix

def _matrix_waves(addr_rows, cfg):
    waves = 0
    for m in range(0, len(addr_rows), 8):
        words = np.unique((addr_rows[m:m + 8] >> 2)[:, None] + np.arange(4))
        waves += int(np.bincount(words % cfg.smem_banks, minlength=cfg.smem_banks).max())
    return max(1, waves)


@op("LDMATRIX")
def _ldmatrix(sm, w, ins, cycle, sim):
    num, trans = ins.imm, ins.opt == "t"
    s16 = w.cta.smem.view(np.uint16)
    rows = _u(w, ins.a).astype(np.int64)[:8 * num]
    if (rows % 16).any() or rows.max() + 16 > w.cta.smem.size:
        raise RuntimeError(f"illegal ldmatrix row address at pc={w.pc - 1}")
    g, t = LANES >> 2, LANES & 3
    for m in range(num):
        M = s16[(rows[8 * m:8 * m + 8, None] >> 1) + np.arange(8)]            # 8 x 8 b16
        lo, hi = (M[t * 2, g], M[t * 2 + 1, g]) if trans else (M[g, t * 2], M[g, t * 2 + 1])
        _wr(w, ins.d + m, lo.astype(np.uint32) | (hi.astype(np.uint32) << 16))
    waves = _matrix_waves(rows, sim.cfg)
    sm._occupy(w, waves)
    return cycle + sim.cfg.lat_smem + waves, True


@op("STMATRIX")
def _stmatrix(sm, w, ins, cycle, sim):
    num, trans = ins.imm, ins.opt == "t"
    s16 = w.cta.smem.view(np.uint16)
    rows = _u(w, ins.a).astype(np.int64)[:8 * num]
    if (rows % 16).any() or rows.max() + 16 > w.cta.smem.size:
        raise RuntimeError(f"illegal stmatrix row address at pc={w.pc - 1}")
    g, t = LANES >> 2, LANES & 3
    for m in range(num):
        reg = w.regs[ins.b + m]
        lo, hi = (reg & 0xFFFF).astype(np.uint16), (reg >> 16).astype(np.uint16)
        M = np.zeros((8, 8), np.uint16)
        if trans:
            M[t * 2, g], M[t * 2 + 1, g] = lo, hi
        else:
            M[g, t * 2], M[g, t * 2 + 1] = lo, hi
        s16[(rows[8 * m:8 * m + 8, None] >> 1) + np.arange(8)] = M
    w.cta.generic_dirty = True
    sm._occupy(w, _matrix_waves(rows, sim.cfg))
    return cycle + 1, False


# ---------------------------------------------------------------- FP64 tensor core

@op("DMMA")
def _dmma(sm, w, ins, cycle, sim):
    g, t = LANES >> 2, LANES & 3
    A, B, C = np.zeros((8, 4)), np.zeros((4, 8)), np.zeros((8, 8))
    A[g, t] = _d(w, ins.a)
    B[t, g] = _d(w, ins.b)
    C[g, 2 * t], C[g, 2 * t + 1] = _d(w, ins.c), _d(w, ins.c + 2)
    D = A @ B + C
    _wd(w, ins.d, D[g, 2 * t])
    _wd(w, ins.d + 2, D[g, 2 * t + 1])
