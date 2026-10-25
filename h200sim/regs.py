"""Operand access helpers shared by the core and the extended op set."""

import numpy as np

from .isa import Pred, Reg, UReg

LANES = np.arange(32)


def _u(w, x):
    """uint32[32] value of an operand: register, uniform register, predicate, float or int immediate."""
    if isinstance(x, Reg):
        return w.regs[x]
    if isinstance(x, UReg):
        return np.full(32, w.uregs[x], np.uint32)
    if isinstance(x, Pred):
        return pbool(w, x).astype(np.uint32)
    if isinstance(x, float):
        return np.full(32, np.float32(x).view(np.uint32), np.uint32)
    return np.full(32, int(x) & 0xFFFFFFFF, np.uint32)


def _f(w, x):
    return _u(w, x).view(np.float32)


def _s32(w, x):
    """Lane 0's value as a signed int (uniform operands: addresses, barrier ids, coordinates)."""
    return int(_u(w, x).view(np.int32)[0])


def pbool(w, p):
    """bool[32] of a predicate operand, or of a register's non-zero lanes."""
    if isinstance(p, Pred):
        return np.ones(32, bool) if int(p) == 7 else w.preds[int(p)]
    return _u(w, p) != 0


def _gaddr(w, x, wide):
    """Per-lane global byte addresses: a 32-bit register, or the lo/hi pair of a 64-bit pointer."""
    if wide:
        if isinstance(x, UReg):
            return np.full(32, int(w.uregs[x]) | (int(w.uregs[x + 1]) << 32), np.int64)
        return w.regs[x].astype(np.int64) | (w.regs[x + 1].astype(np.int64) << 32)
    return _u(w, x).astype(np.int64)


def _swz(addr, mode):
    """Shared-memory swizzle: 16 B chunk index (addr bits 4..) XOR row-in-atom (addr bits 7..)."""
    if not mode:
        return addr
    return addr ^ (((addr >> 7) & {32: 1, 64: 3, 128: 7}[mode]) << 4)


def _wr(w, d, v):
    """Write a uint32[32] result to register d on the active lanes (or a uniform / predicate dest)."""
    if isinstance(d, UReg):
        w.uregs[d] = np.uint32(v[0]) if hasattr(v, "__len__") else np.uint32(v)
    elif isinstance(d, Pred):
        if int(d) != 7:
            w.preds[int(d)][w.mask] = (v != 0)[w.mask]
    elif w.full:
        w.regs[d][:] = v
    else:
        w.regs[d][w.mask] = v[w.mask]


def _wr64(w, d, v):
    _wr(w, d, (v & np.uint64(0xFFFFFFFF)).astype(np.uint32))
    _wr(w, d + 1, (v >> np.uint64(32)).astype(np.uint32))
