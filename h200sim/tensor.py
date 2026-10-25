"""Tensor-core data types, fragment layouts, timing and numerics.

Layouts are the PTX mma.sync / wgmma register fragments. Timing comes from the Hopper
microbenchmark papers (H800 PCIe, 114 SMs at 1755 MHz; see README sources): mma.sync reaches
~65 % of the wgmma peak, wgmma runs a m64nNk(*) in max(N/2, floor) cycles per sub-core whatever
the data type (a 2x/0.5x data type changes FLOPs, not cycles), sparse wgmma runs 2:4 matrices at
the same cycles with twice the effective K.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DT:
    name: str
    bits: int          # element size in a register / shared memory
    kind: str          # "f" float, "i" signed int, "u" unsigned int
    k_mma: int         # K of the dense mma.sync shape (m16n8kK)
    k_wg: int          # K of one dense wgmma (m64nNkK)
    peak: int          # dense FLOP (or op) per cycle per sub-core, bf16/f16 = 1024


DTYPES = {d.name: d for d in (
    DT("f16", 16, "f", 16, 16, 1024), DT("bf16", 16, "f", 16, 16, 1024),
    DT("tf32", 32, "f", 8, 8, 512),
    DT("e4m3", 8, "f", 32, 32, 2048), DT("e5m2", 8, "f", 32, 32, 2048),
    DT("s8", 8, "i", 32, 32, 2048), DT("u8", 8, "u", 32, 32, 2048),
)}
ACC_OK = {"f16": ("f32", "f16"), "bf16": ("f32",), "tf32": ("f32",), "e4m3": ("f32", "f16"),
          "e5m2": ("f32", "f16"), "s8": ("s32",), "u8": ("s32",)}


def _fp8_lut(exp_bits, man_bits, bias, has_inf):
    out = np.zeros(256, np.float32)
    for code in range(256):
        s = -1.0 if code >> 7 else 1.0
        e = (code >> man_bits) & ((1 << exp_bits) - 1)
        m = code & ((1 << man_bits) - 1)
        top = (1 << exp_bits) - 1
        if has_inf and e == top:
            v = np.inf if m == 0 else np.nan
        elif not has_inf and e == top and m == (1 << man_bits) - 1:
            v = np.nan                                  # e4m3fn: only S.1111.111 is NaN
        elif e == 0:
            v = m / (1 << man_bits) * 2.0 ** (1 - bias)
        else:
            v = (1 + m / (1 << man_bits)) * 2.0 ** (e - bias)
        out[code] = s * v
    return out


E4M3 = _fp8_lut(4, 3, 7, False)       # max 448
E5M2 = _fp8_lut(5, 2, 15, True)       # max 57344


def decode(dt: str, raw: np.ndarray) -> np.ndarray:
    """Element container values (uint8 / uint16 / uint32) -> float32 (int32 for integer types)."""
    if dt == "bf16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if dt == "f16":
        return raw.astype(np.uint16).view(np.float16).astype(np.float32)
    if dt == "tf32":                                    # the low 13 mantissa bits are ignored
        return (raw.astype(np.uint32) & 0xFFFFE000).view(np.float32)
    if dt == "e4m3":
        return E4M3[raw.astype(np.uint8)]
    if dt == "e5m2":
        return E5M2[raw.astype(np.uint8)]
    if dt == "s8":
        return raw.astype(np.uint8).view(np.int8).astype(np.int32)
    if dt == "u8":
        return raw.astype(np.uint8).astype(np.int32)
    raise ValueError(dt)


def encode_fp8(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """float32 -> fp8 codes, round to nearest even, saturating to the largest finite value."""
    finite = np.isfinite(lut)
    codes = np.nonzero(finite)[0]
    vals = lut[codes]
    order = np.argsort(vals, kind="stable")
    codes, vals = codes[order], vals[order]
    x = np.nan_to_num(np.clip(x, vals[0], vals[-1]), nan=0.0)
    hi = np.clip(np.searchsorted(vals, x), 1, len(vals) - 1)
    lo = hi - 1
    dl, dh = x - vals[lo], vals[hi] - x
    pick_hi = (dh < dl) | ((dh == dl) & ((codes[hi] & 1) == 0))
    return np.where(pick_hi, codes[hi], codes[lo]).astype(np.uint8)


# ---------------------------------------------------------------- fragments

def _geom(bits, k):
    epr = 32 // bits
    return epr, k // (4 * epr)          # elements per register, k-blocks


def unpack_regs(dt: str, regs: np.ndarray) -> np.ndarray:
    """(n, 32 lanes) uint32 registers -> (n, 32, elements per register) decoded values."""
    bits = DTYPES[dt].bits
    epr = 32 // bits
    out = np.stack([(regs >> (bits * i)) & ((1 << bits) - 1) for i in range(epr)], axis=-1)
    return decode(dt, out)


def a_fragment(dt: str, regs: np.ndarray, k: int) -> np.ndarray:
    """(m16 x k) matrix from the A-fragment registers (regs is (n_regs, 32))."""
    epr, kb = _geom(DTYPES[dt].bits, k)
    vals = unpack_regs(dt, regs)
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    A = np.zeros((16, k), vals.dtype)
    r = 0
    for b in range(kb):
        for half in (0, 1):
            for i in range(epr):
                A[g + 8 * half, b * 4 * epr + epr * t + i] = vals[r, :, i]
            r += 1
    return A


def pack_a_fragment(dt: str, A: np.ndarray, k: int) -> np.ndarray:
    """Inverse of a_fragment for raw element containers: A is (16, k) of raw uint codes."""
    bits = DTYPES[dt].bits
    epr, kb = _geom(bits, k)
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    out = np.zeros((kb * 2, 32), np.uint32)
    r = 0
    for b in range(kb):
        for half in (0, 1):
            for i in range(epr):
                out[r] |= A[g + 8 * half, b * 4 * epr + epr * t + i].astype(np.uint32) << (bits * i)
            r += 1
    return out


def b_fragment(dt: str, regs: np.ndarray, k: int) -> np.ndarray:
    """(k x 8) matrix from the B-fragment registers."""
    epr, kb = _geom(DTYPES[dt].bits, k)
    vals = unpack_regs(dt, regs)
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    B = np.zeros((k, 8), vals.dtype)
    for b in range(kb):
        for i in range(epr):
            B[b * 4 * epr + epr * t + i, g] = vals[b, :, i]
    return B


def pack_b_fragment(dt: str, B: np.ndarray, k: int) -> np.ndarray:
    bits = DTYPES[dt].bits
    epr, kb = _geom(bits, k)
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    out = np.zeros((kb, 32), np.uint32)
    for b in range(kb):
        for i in range(epr):
            out[b] |= B[b * 4 * epr + epr * t + i, g].astype(np.uint32) << (bits * i)
    return out


def c_fragment(regs: np.ndarray, acc: str) -> np.ndarray:
    """(16 x 8) accumulator from 4 registers (f32/s32) or 2 packed-f16 registers."""
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    C = np.zeros((16, 8), np.float32 if acc != "s32" else np.int32)
    if acc == "f16":
        for half in (0, 1):
            h = regs[half].view(np.float16)             # (64,) = lane0 lo, lane0 hi, ...
            C[g + 8 * half, 2 * t] = h[0::2].astype(np.float32)
            C[g + 8 * half, 2 * t + 1] = h[1::2].astype(np.float32)
        return C
    for r in range(4):
        v = regs[r].view(np.float32 if acc == "f32" else np.int32)
        C[g + 8 * (r >> 1), 2 * t + (r & 1)] = v
    return C


def store_c_fragment(regs: np.ndarray, D: np.ndarray, acc: str) -> None:
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    if acc == "f16":
        for half in (0, 1):
            h = np.empty(64, np.float16)
            h[0::2] = D[g + 8 * half, 2 * t].astype(np.float16)
            h[1::2] = D[g + 8 * half, 2 * t + 1].astype(np.float16)
            regs[half][:] = h.view(np.uint32)
        return
    for r in range(4):
        v = D[g + 8 * (r >> 1), 2 * t + (r & 1)]
        regs[r][:] = v.astype(np.float32 if acc == "f32" else np.int32).view(np.uint32)


# ---------------------------------------------------------------- numerics

def _trunc_mantissa(x: np.ndarray, bits: int) -> np.ndarray:
    """Round toward zero to `bits` fraction bits (fp32 container)."""
    u = x.astype(np.float32).view(np.uint32)
    mask = np.uint32(0xFFFFFFFF << (23 - bits) & 0xFFFFFFFF)
    return (u & mask).view(np.float32)


def fp8_accumulate(A: np.ndarray, B: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Hopper FP8 tensor-core accumulation as DeepSeek measured it: the products of one
    instruction are aligned to the largest product's exponent and only ~13 fraction bits are
    kept (the rest is truncated), and the running accumulator is a 13-bit-mantissa (FP22)
    register. A (m x k), B (n x k) decoded, acc (m x n); returns the new accumulator."""
    P = A[:, None, :].astype(np.float64) * B[None, :, :].astype(np.float64)       # m x n x k
    mag = np.abs(P).max(axis=-1, keepdims=True)
    e = np.floor(np.log2(np.where(mag > 0, mag, 1.0)))
    quantum = 2.0 ** (e - 13)
    S = (np.trunc(P / quantum) * quantum).sum(axis=-1)
    return _trunc_mantissa((acc.astype(np.float64) + S).astype(np.float32), 13)


def dense_from_sparse(Acomp: np.ndarray, meta: np.ndarray, k_dense: int) -> np.ndarray:
    """Rebuild the dense (m x k) 2:4 matrix. Acomp is (m x k/2); meta is (m x k/4) bytes, each
    holding two 2-bit positions idx0 | idx1 << 2 of the kept values within a group of four."""
    m = Acomp.shape[0]
    A = np.zeros((m, k_dense), Acomp.dtype)
    rows = np.arange(m)[:, None]
    grp = np.arange(k_dense // 4)[None, :]
    i0, i1 = meta & 3, (meta >> 2) & 3
    A[rows, grp * 4 + i0] = Acomp[:, 0::2]
    A[rows, grp * 4 + i1] = Acomp[:, 1::2]
    return A


# ---------------------------------------------------------------- timing

# mma.sync throughput as a fraction of the dtype's wgmma peak, from the H800 measurements
# (Hopper papers, Table VII): mma reaches 62.9 % of peak on average.
_MMA_FRAC = {
    ("f16", 8, "f16"): 368.6 / 756.5, ("f16", 16, "f16"): 494.4 / 756.5,
    ("f16", 8, "f32"): 363.7 / 756.5, ("f16", 16, "f32"): 490.7 / 756.5,
    ("tf32", 4, "f32"): 180.6 / 378.0, ("tf32", 8, "f32"): 246.4 / 378.0,
    ("i8", 16, "s32"): 730.3 / 1513.0, ("i8", 32, "s32"): 977.9 / 1513.0,
}
_MMA_LAT = {("f16", 8): 16.0, ("f16", 16): 24.1, ("tf32", 4): 16.5, ("tf32", 8): 24.5,
            ("i8", 16): 16.1, ("i8", 32): 24.0}


def mma_timing(dt: str, k: int, acc: str):
    """(result latency, sub-core tensor-pipe occupancy) in cycles for one mma.sync m16n8kK."""
    cls = {"bf16": "f16", "e4m3": "i8", "e5m2": "i8", "s8": "i8", "u8": "i8"}.get(dt, dt)
    flops = 2 * 16 * 8 * k
    frac = _MMA_FRAC.get((cls, k, acc), _MMA_FRAC.get((cls, k, "f32" if acc == "f16" else acc), 0.65))
    return _MMA_LAT.get((cls, k), 24.0), flops / (DTYPES[dt].peak * frac)


# wgmma floor on cycles for small N (H800, m64nNk16 FP16, Table IX): the pipeline cannot go
# below this however little work there is. N/2 is the work-bound value.
_FLOOR_SS = {8: 18.0, 16: 20.0, 32: 24.0}
_FLOOR_RS = {8: 13.0, 16: 13.0, 32: 16.0}
_FLOOR_SP_RS = {8: 16.0, 16: 18.0, 32: 18.0}


def wgmma_smem_wavefronts(n: int, rs: bool, dtype: str = "bf16", sparse: bool = False) -> int:
    """128 B shared-memory wavefronts one whole-warpgroup wgmma m64nNkK reads: A (64 x K, unless it is in
    registers) once, B (N x K) once -- the hardware shares B between the four warps."""
    spec = DTYPES[dtype]
    eb = spec.bits // 8
    kc = spec.k_wg
    ke = 2 * kc if sparse else kc
    nbytes = (0 if rs else 64 * kc * eb) + n * ke * eb          # sparse metadata is in registers
    return -(-nbytes // 128)


def _interp(table, n):
    xs = sorted(table)
    return float(np.interp(n, xs, [table[x] for x in xs]))


def wgmma_cycles(n: int, rs: bool, sparse: bool) -> float:
    """Cycles one sub-core's tensor pipe is busy with its m16 slice of a m64nNkK wgmma, which is
    also its latency in a dependent chain."""
    work = n / 2.0
    if sparse:
        if rs:
            return max(work, _interp(_FLOOR_SP_RS, min(n, 32)))
        return work + 16.0                              # sparse SS reads the A tile + metadata: +16
    return max(work, _interp(_FLOOR_RS if rs else _FLOOR_SS, min(n, 32)))


# ---------------------------------------------------------------- mma.sp (sparse mma.sync)
#
# From the PTX ISA (9.7.14.6): A is structured-sparse -- 2:4 for 16-bit, 8-bit and fp8 types (two kept values in
# each chunk of four), 1:2 for tf32 (one kept value in each chunk of two). The metadata holds, per chunk, the
# positions of the kept values: a nibble idx0 | idx1 << 2 for 2:4 (valid values 0100 1000 1001 1100 1101 1110,
# i.e. idx0 < idx1), and for tf32 a nibble 0100 (keep the first element) or 1110 (keep the second).
# Who supplies the metadata (the "sparsity selector"):
#   m16n8k16 f16/bf16 and m16n8k8 tf32: one thread of each quad (selector 0..3);
#   m16n8k32 f16/bf16, m16n8k16 tf32, m16n8k32 8-bit: a pair of threads (selector 0 = T0,T1; 1 = T2,T3);
#   m16n8k64 8-bit and fp8: all four threads (selector 0).
# The A fragment holds the kept values: thread t of quad row g holds the kept values of dense chunk t (and chunk
# t+4 for the larger shapes) for rows g and g+8, which is the dense fragment of half the K for 2:4 and of a
# quarter... (see a_fragment use). The text gives how many nibbles a register holds; the exact bit position of
# each chunk inside a lane is drawn only in the figures, so the order below (row g first, chunks ascending) is
# the natural reading, not a verified one.

_VALID_24 = {(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)}
_TF32_NIBBLE = (0b0100, 0b1110)         # keep element 0 / keep element 1 of a 2-wide chunk


def _sparse_geometry(dt: str, k: int):
    """(chunk width, kept per chunk, chunks per row)."""
    gw = 2 if dt == "tf32" else 4
    return gw, (1 if dt == "tf32" else 2), k // gw


def _sparse_lanes(dt: str, k: int, sel: int):
    """[(lane offset in quad, row half, first chunk, chunks)] describing who holds which metadata."""
    bits = DTYPES[dt].bits
    gw, _, chunks = _sparse_geometry(dt, k)
    if bits == 8 or dt == "tf32" and chunks > 4:
        out = []                                               # one row per lane, up to 8 chunks (32 bits) each
        for p in range(max(1, chunks // 8)):
            base = 2 * sel + 2 * p
            out += [(base, 0, 8 * p, min(8, chunks)), (base + 1, 1, 8 * p, min(8, chunks))]
        return out
    if chunks <= 4:                                            # both rows of the thread's group in one lane
        return [(sel, 0, 0, chunks), (sel, 1, 0, chunks)]
    return [(2 * sel, 0, 0, 4), (2 * sel, 1, 0, 4), (2 * sel + 1, 0, 4, 4), (2 * sel + 1, 1, 4, 4)]


def encode_sparse_meta(dt: str, idx: np.ndarray, k: int, sel: int = 0) -> np.ndarray:
    """idx: (16 rows, chunks, kept) positions inside each chunk -> (32,) uint32 metadata register per lane.
    2:4 types: kept = 2 with idx0 < idx1; tf32 (1:2): kept = 1 with a position 0 or 1."""
    meta = np.zeros(32, np.uint32)
    for g in range(8):
        for quad_lane, half, c0, nchunks in _sparse_lanes(dt, k, sel):
            row = g + 8 * half
            word = 0
            for j in range(nchunks):
                if dt == "tf32":
                    nib = _TF32_NIBBLE[int(idx[row, c0 + j, 0])]
                else:
                    i0, i1 = (int(x) for x in idx[row, c0 + j])
                    assert (i0, i1) in _VALID_24, "invalid 2:4 index pair"
                    nib = i0 | (i1 << 2)
                word |= nib << (4 * j)
            if DTYPES[dt].bits == 8 or (dt == "tf32" and k // 2 > 4):
                meta[4 * g + quad_lane] = np.uint32(word)
            else:                                              # two rows share the lane: low / high 16 bits
                meta[4 * g + quad_lane] |= np.uint32(word << (16 * half))
    return meta


def decode_sparse_a(dt: str, comp: np.ndarray, meta: np.ndarray, k: int, sel: int = 0) -> np.ndarray:
    """Dense (16 x k) A from the compressed matrix (16 x k/2) and the per-lane metadata registers."""
    gw, kept, _ = _sparse_geometry(dt, k)
    A = np.zeros((16, k), comp.dtype)
    one_row_per_lane = DTYPES[dt].bits == 8 or (dt == "tf32" and k // 2 > 4)
    for g in range(8):
        for quad_lane, half, c0, nchunks in _sparse_lanes(dt, k, sel):
            row = g + 8 * half
            word = int(meta[4 * g + quad_lane])
            if not one_row_per_lane:
                word = (word >> (16 * half)) & 0xFFFF
            for j in range(nchunks):
                nib = (word >> (4 * j)) & 0xF
                c = c0 + j
                if dt == "tf32":
                    A[row, 2 * c + (1 if nib == _TF32_NIBBLE[1] else 0)] = comp[row, c]
                else:
                    A[row, 4 * c + (nib & 3)] = comp[row, 2 * c]
                    A[row, 4 * c + (nib >> 2)] = comp[row, 2 * c + 1]
    return A


# sparse mma.sync timing, H800 (Hopper papers Table VII): (class, actual K, acc) -> (latency, TFLOPS)
_MMA_SP = {("f16", 16, "f16"): (16.0, 493.8), ("f16", 32, "f16"): (24.0, 722.8),
           ("f16", 16, "f32"): (16.0, 488.7), ("f16", 32, "f32"): (24.0, 721.8),
           ("tf32", 8, "f32"): (16.4, 240.7), ("tf32", 16, "f32"): (24.4, 363.3),
           ("i8", 32, "s32"): (16.1, 970.0), ("i8", 64, "s32"): (24.2, 1435.0)}
_PEAK_TF = {"f16": 756.5, "tf32": 378.0, "i8": 1513.0}


def mma_sp_timing(dt: str, k: int, acc: str):
    cls = {"bf16": "f16", "e4m3": "i8", "e5m2": "i8", "s8": "i8", "u8": "i8"}.get(dt, dt)
    lat, tf = _MMA_SP.get((cls, k, acc), _MMA_SP.get((cls, k, "f32" if acc == "f16" else acc), (24.0, 0.65 * _PEAK_TF[cls])))
    flops = 2 * 16 * 8 * k                              # effective (dense-equivalent) FLOPs
    return lat, flops / (DTYPES[dt].peak * tf / _PEAK_TF[cls])
