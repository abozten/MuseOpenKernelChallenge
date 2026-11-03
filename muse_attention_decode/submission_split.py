# okc: problem=muse_attention_decode
"""Gated attention decode in three Triton launches, where SGLang runs seven:

  _prologue   per-head QK RMSNorm, interleaved RoPE, FP8 casts of q, k and v, and the
              scattered cache store (SGLang: FusedQKNorm, two casts, store_cache, a q cast)
  _attend     split-KV flash decode over the paged FP8 cache. With page size 1 every token is
              a random row, so a program reads both KV heads' whole 128-byte rows at once and
              serves both heads with one tl.dot against a block-diagonal Q (rows of head 0 are
              zero in head 1's columns and vice versa); the wasted MMA work is free next to
              the gather. Verify blocks (8 tokens x 8 heads per KV head) take one KV head per
              program instead. Q is the FP8-rounded query, as FA3 uses.
  _combine    merges the splits and applies RMSNorm(out) * sigmoid(gate) (SGLang: FA3's
              combine and norm_gate)."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


FP8 = torch.float8_e4m3fn
ROPE_POSITIONS = 1 << 21
LOG2E = 1.4426950408889634
_ROPE = {}
_SMS = {}


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, built as SGLang's RotaryEmbedding does."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


@triton.jit
def _prologue(qkv, pos, loc, cos_sin, k_cache, v_cache, q_out, k_scale, v_scale,
              HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
              ROPE: tl.constexpr, SCALED: tl.constexpr):
    # One program per (token, head); heads [0, HQ) are queries, [HQ, HQ + HKV) keys + values.
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    row = qkv + t * ((2 * HQ + 2 * HKV) * D) + h * D
    ev = tl.load(row + 2 * i).to(tl.float32)
    od = tl.load(row + 2 * i + 1).to(tl.float32)
    r = tl.rsqrt((tl.sum(ev * ev, 0) + tl.sum(od * od, 0)) / D + 1e-5)
    ev = ev * r
    od = od * r
    if ROPE:
        p = tl.load(pos + t)
        c = tl.load(cos_sin + p * D + i)
        s = tl.load(cos_sin + p * D + HALF + i)
        ev, od = ev * c - od * s, ev * s + od * c
    if h < HQ:
        dst = q_out + (t * HQ + h) * D
        tl.store(dst + 2 * i, ev.to(tl.float8e4nv))
        tl.store(dst + 2 * i + 1, od.to(tl.float8e4nv))
    else:
        kh = h - HQ
        slot = tl.load(loc + t)
        v = tl.load(row + HKV * D + 2 * i).to(tl.float32)
        vo = tl.load(row + HKV * D + 2 * i + 1).to(tl.float32)
        if SCALED:
            ks = 1.0 / tl.load(k_scale)
            vs = 1.0 / tl.load(v_scale)
            ev, od, v, vo = ev * ks, od * ks, v * vs, vo * vs
        kd = k_cache + (slot * HKV + kh) * D
        vd = v_cache + (slot * HKV + kh) * D
        tl.store(kd + 2 * i, ev.to(tl.float8e4nv))
        tl.store(kd + 2 * i + 1, od.to(tl.float8e4nv))
        tl.store(vd + 2 * i, v.to(tl.float8e4nv))
        tl.store(vd + 2 * i + 1, vo.to(tl.float8e4nv))


@triton.jit
def _attend(q8, k_cache, v_cache, page_table, kv_lens, k_scale, part_acc, part_ml, scale,
            pt_stride, CHUNK,
            L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
            PAGE: tl.constexpr, WINDOW: tl.constexpr, NS: tl.constexpr, KPP: tl.constexpr,
            M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Program (b, pair, s): request b, KV heads KPP*pair .. KPP*pair + KPP - 1, the s-th chunk of
    # its keys. Rows r of the block: token j = r // (KPP*G), local KV head kl = (r // G) % KPP,
    # group member g = r % G. KPP = 2 (decode) reads whole 128-byte rows with a block-diagonal Q;
    # KPP = 1 (verify: 8 tokens x 8 heads already fill the MMA) wastes no MMA work.
    b = tl.program_id(0)
    pair = tl.program_id(1)
    s = tl.program_id(2)
    D: tl.constexpr = 64
    W2: tl.constexpr = KPP * D
    ROW: tl.constexpr = HKV * D  # bytes per cache row (all KV heads of one token)

    r = tl.arange(0, M)
    j = r // (KPP * G)
    kl = (r // G) % KPP
    qh = (pair * KPP + kl) * G + r % G
    c = tl.arange(0, W2)
    rvalid = j < L
    qmask = rvalid[:, None] & ((c[None, :] // D) == kl[:, None])
    q = tl.load(q8 + ((b * L + j) * HQ + qh)[:, None] * D + (c % D)[None, :], mask=qmask, other=0.0)
    q = q.to(tl.bfloat16)

    kv_len = tl.load(kv_lens + b)
    seq = kv_len - L
    p_j = seq + j  # each row's query position
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(seq - WINDOW + 1, 0)
    start = lo + s * CHUNK
    end = tl.minimum(start + CHUNK, kv_len)

    m_i = tl.full([M], float("-inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, W2], tl.float32)
    n = tl.arange(0, BLOCK_N)
    kscale = tl.load(k_scale) * scale
    pt = page_table + b * pt_stride
    for t0 in range(start, end, BLOCK_N):
        t = t0 + n
        tin = t < end
        page = tl.load(pt + t // PAGE, mask=tin, other=0)
        rows = (page.to(tl.int64) * PAGE + t % PAGE) * ROW + pair * W2
        k = tl.load(k_cache + rows[:, None] + c[None, :], mask=tin[:, None], other=0.0)
        v = tl.load(v_cache + rows[:, None] + c[None, :], mask=tin[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * kscale
        vis = tin[None, :] & (t[None, :] <= p_j[:, None])
        if WINDOW > 0:
            vis = vis & (t[None, :] > p_j[:, None] - WINDOW)
        sc = tl.where(vis, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(sc - m_use[:, None])
        alpha = tl.exp2(m_i - m_use)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new

    # Partials per (token, query head, split): acc [64] from the row's own KV head's columns.
    base = ((b * L + j) * HQ + qh) * NS + s
    smask = rvalid[:, None] & ((c[None, :] // D) == kl[:, None])
    tl.store(part_acc + base[:, None] * D + (c % D)[None, :], acc, mask=smask)
    tl.store(part_ml + base * 2, m_i, mask=rvalid)
    tl.store(part_ml + base * 2 + 1, l_i, mask=rvalid)


@triton.jit
def _combine(part_acc, part_ml, qkv, v_scale, out,
             HQ: tl.constexpr, HKV: tl.constexpr, NS: tl.constexpr, SB: tl.constexpr):
    # One program per (token, query head): merge NS splits, then RMSNorm * sigmoid(gate).
    th = tl.program_id(0)
    t = th // HQ
    h = th % HQ
    D: tl.constexpr = 64
    d = tl.arange(0, D)
    sb = tl.arange(0, SB)
    m_all = tl.full([SB], float("-inf"), tl.float32)
    for s0 in range(0, NS, SB):
        m_all = tl.maximum(m_all, tl.load(part_ml + (th * NS + s0 + sb) * 2, mask=s0 + sb < NS,
                                          other=float("-inf")))
    m = tl.max(m_all, 0)
    l = tl.zeros([SB], tl.float32)
    o = tl.zeros([D], tl.float32)
    for s0 in range(0, NS, SB):
        sm = s0 + sb < NS
        ms = tl.load(part_ml + (th * NS + s0 + sb) * 2, mask=sm, other=float("-inf"))
        ls = tl.load(part_ml + (th * NS + s0 + sb) * 2 + 1, mask=sm, other=0.0)
        w = tl.exp2(ms - m)
        a = tl.load(part_acc + (th * NS + s0 + sb)[:, None] * D + d[None, :], mask=sm[:, None], other=0.0)
        l += w * ls
        o += tl.sum(a * w[:, None], 0)
    o = o / tl.sum(l, 0) * tl.load(v_scale)
    o = o.to(tl.bfloat16).to(tl.float32)  # FA3 hands norm_gate a BF16 output
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + 1e-5)
    g = tl.load(qkv + t * ((2 * HQ + 2 * HKV) * D) + (HQ + 2 * HKV + h) * D + d).to(tl.float32)
    o = o * tl.sigmoid(g)
    tl.store(out + t * (HQ * D) + h * D + d, o.to(tl.bfloat16))


def _sms(device):
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device]


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    assert d == 64
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    rope = bool(cfg["rope_theta"])
    cos_sin = _cos_sin(cfg["rope_theta"], d, device) if rope else qkv
    scaled = cfg["kv_scale"] != 1.0
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    q8 = torch.empty((tokens, hq, d), dtype=FP8, device=device)
    _prologue[(tokens, hq + hkv)](qkv, data["positions"], data["cache_loc"], cos_sin, k_cache, v_cache, q8,
                                  data["k_scale"], data["v_scale"], HQ=hq, HKV=hkv, D=d,
                                  ROPE=rope, SCALED=scaled, num_warps=1)

    # Split each request's key range into NS static chunks (shapes only: the graph is replayed).
    window = cfg["window"]
    page_table = data["page_table"]
    span = window + length - 1 if window else page_table.shape[1] * page
    group = hq // hkv
    kpp = 2 if length * group < 16 else 1
    pairs = hkv // kpp
    rows = max(16, triton.next_power_of_2(length * kpp * group))
    block_n = 64
    target = 4 * _sms(device)
    ns = max(1, min(triton.cdiv(span, block_n), triton.cdiv(target, batch * pairs)))
    chunk = triton.cdiv(triton.cdiv(span, ns), block_n) * block_n
    ns = triton.cdiv(span, chunk)

    part_acc = torch.empty((tokens, hq, ns, d), dtype=torch.float32, device=device)
    part_ml = torch.empty((tokens, hq, ns, 2), dtype=torch.float32, device=device)
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    _attend[(batch, pairs, ns)](q8, k_cache, v_cache, page_table, data["cache_seqlens"], data["k_scale"],
                                part_acc, part_ml, scale, page_table.stride(0), chunk,
                                L=length, HQ=hq, HKV=hkv, G=group, PAGE=page, WINDOW=window, NS=ns,
                                KPP=kpp, M=rows, BLOCK_N=block_n, num_warps=4 if rows <= 64 else 8, num_stages=3)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=device)
    _combine[(tokens * hq,)](part_acc, part_ml, qkv, data["v_scale"], out, HQ=hq, HKV=hkv, NS=ns,
                             SB=min(64, triton.next_power_of_2(ns)), num_warps=2)
    return out.view(batch, length, hq * d)
