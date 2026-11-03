"""Builds muse_decode_layer/submission.py (and, with --split, submission_split.py): SGLang's layer
(baseline.py) with the muse_bf16_proj kernels for the dense projections and, with --split,
muse_attention_decode's Triton attention, spliced in as one file (the judge takes one), plus
the layer-only fusions below (norm-sum kernel, one-launch local ids, Marlin block size)."""
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent.parent
proj = (root / "muse_bf16_proj/submission.py").read_text()
layer = (root / "muse_decode_layer/baseline.py").read_text()
import sys
SPLIT = "--split" in sys.argv  # attention on the split-KV Triton kernels instead of FA3
attn = (root / "muse_attention_decode/submission.py").read_text()

# The projections module: everything after its problem line, prepare/kernel_fn renamed.
proj = proj.split("\n", 1)[1]
proj = proj.replace("def prepare(weights: Weights):", "def _proj_prepare(weights):")
proj = proj.replace("def kernel_fn(data: Inputs, state) -> torch.Tensor:", "def _proj_fn(data, state):")
proj = proj.replace("_tune_w12(state, t0 + 200)", "_tune_w12(state, t0 + 150)")
for name in ("Weights", "Config", "Inputs"):
    proj = proj.replace(f"class {name}(TypedDict):", f"class _Proj{name}(TypedDict):")
    proj = proj.replace(f": {name}", f": _Proj{name}")
assert "def _proj_prepare(" in proj and "def _proj_fn(" in proj

kernel = attn[attn.index("@triton.jit"):attn.index("def kernel_fn")]

doc_end = layer.index('"""', 3) + 3
body = layer[doc_end:]
body = body.replace("import torch\n", "import torch\nimport triton\nimport triton.language as tl\n", 1)
body = body.replace("from sglang.kernels.ops.kvcache.kvcache import store_cache\n", "")
body = body.replace("from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm\n", "")
body = body.replace('        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),\n', "")
body = body.replace("\ndef _gemm(x, w):", "\n" + kernel + '''def _dense(state, x, layer, n):
    """x @ W.T on the muse_bf16_proj kernels (W12 or the tuned cuBLAS variant), row-major: the
    tuned "nt" variant returns a transposed view, and the Triton kernels index rows."""
    return _proj_fn({"x": x, "config": {"m": x.shape[0], "n": n, "k": x.shape[1], "layer": layer}},
                    state["proj"]).contiguous()


def _gemm(x, w):''')
body = body.replace('''    state = {
        "dense":''', '''    proj = _proj_prepare({name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler", "router")})
    state = {
        "proj": proj,
        "dense":''')

s = body.index('    qkv = _gemm(data["attn_input"], dense["qkv_gate"])')
e = body.index('    window = cfg["window"]', s)
body = body[:s] + '''    qkv = _dense(state, data["attn_input"], "qkv_gate", (2 * hq + 2 * hkv) * d)
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    rope = bool(cfg["rope_theta"])
    q8 = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
    _prologue[(tokens, hq + hkv)](qkv, data["positions"], data["cache_loc"], state["cos_sin"] if rope else qkv,
                                  k_cache, v_cache, q8, data["k_scale"], data["v_scale"], HQ=hq, HKV=hkv, D=d,
                                  ROPE=rope, SCALED=scaled, num_warps=1)
''' + body[e:]
body = body.replace("        q=q.view(tokens, hq, d).to(FP8),", "        q=q8,")
if SPLIT:
    # The split-KV kernels, except verify blocks on full layers with many requests, where FA3
    # (and SGLang's norm gate) measured faster.
    kf = body.index("def kernel_fn(")
    s0 = body.index('    window = cfg["window"]', kf)
    s1 = body.index('    attn_output = _gemm(o, dense["o"])', kf)
    fa3 = "\n".join("    " + line if line else line for line in body[s0:s1].rstrip("\n").split("\n"))
    fa3 = "\n".join("    " + line if line else line for line in fa3.split("\n"))
    body = body[:s0] + (
        "        if cfg[\"window\"] or batch <= 4:\n"
        "            o = _attention(data, cfg, qkv, q8, batch, length, tokens, hq, hkv, d)\n"
        "        else:\n" + fa3 + "\n") + body[s1:]
    # Decode steps: muse_attention_decode's _decode (queries and the new token's cache store
    # inside the attention kernel, so no prologue launch); verify blocks: the prologue first.
    p0 = body.index("    q8 = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)", kf)
    p1 = body.index("        if cfg[\"window\"] or batch <= 4:", kf)
    pro = "\n".join("    " + line if line else line for line in body[p0:p1].rstrip("\n").split("\n"))
    body = body[:p0] + (
        "    if length == 1:\n"
        "        o = _decode(data, cfg, qkv, state[\"cos_sin\"] if rope else qkv, rope, scaled, batch, hq, hkv, d,\n"
        "                    cfg[\"page_size\"], qkv.device).view(tokens, hq * d)\n"
        "    else:\n" + pro + "\n") + body[p1:]
    helpers = ""  # _sms comes with the kernels
    att_fn = '''def _attention(data, cfg, qkv, q8, batch, length, tokens, hq, hkv, d):
    """Split-KV flash decode and the gated output norm (muse_attention_decode's kernels)."""
    page = cfg["page_size"]
    device = qkv.device
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
    _attend[(batch, pairs, ns)](q8, data["k_cache"], data["v_cache"], page_table, data["cache_seqlens"],
                                data["k_scale"], part_acc, part_ml, scale, page_table.stride(0), chunk,
                                L=length, HQ=hq, HKV=hkv, G=group, PAGE=page, WINDOW=window, NS=ns,
                                KPP=kpp, M=rows, BLOCK_N=block_n, num_warps=4 if rows <= 64 else 8, num_stages=3)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=device)
    _combine[(tokens * hq,)](part_acc, part_ml, qkv, data["v_scale"], out, HQ=hq, HKV=hkv, NS=ns,
                             SB=min(64, triton.next_power_of_2(ns)), num_warps=2)
    return out


'''
    body = body.replace("\ndef _gemm(x, w):", "\n" + helpers + att_fn + "def _gemm(x, w):", 1)
    body = body.replace("ROPE_THETA = 500_000.0\n", "ROPE_THETA = 500_000.0\nLOG2E = 1.4426950408889634\n_SMS = {}\n_COUNTERS = {}\nFUSED_SPLITS = 8\n", 1)
body = body.replace('attn_output = _gemm(o, dense["o"])', 'attn_output = _dense(state, o, "o", HIDDEN)')
body = body.replace('''router_dtype=None if split_router else torch.float32)''', '''router_dtype=None)''')
body = body.replace('expert_input = _gemm(ffn_input, dense["pooler"])', 'expert_input = _dense(state, ffn_input, "pooler", D_EXPERT)')
s = body.index("        if split_router:\n            logits = torch.mm(")
e = body.index("        topk_weights, topk_ids = state[\"routing\"]")
body = body[:s] + '        logits = _dense(state, ffn_input, "router", N_EXPERTS)\n' + body[e:]
body = body.replace('unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])',
                    'unpooled = _dense(state, expert_output, "unpooler", UNPOOL_SLICE)')
body = body.replace('ffn_output = _gemm(expert_output, dense["unpooler"])',
                    'ffn_output = _dense(state, expert_output, "unpooler", HIDDEN)')
body = body.replace("    split_router = tokens > ROUTER_FP32_MAX_TOKENS\n", "")
assert "= _gemm(" not in body, "a cuBLAS call left"

# The norm-sum as SGLang's Triton kernel at every token count (its wrapper runs ~8 eager kernels
# below 32 tokens), and the local expert ids in one launch instead of three.
LOCAL_IDS_KERNEL = """
@triton.jit
def _local_ids(ids, start, out, n, N: tl.constexpr, LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    # SGLang's expert_map on this EP rank: (id - local_start) mod N if that is < LOCAL, else -1.
    i = tl.arange(0, BLOCK)
    e = (tl.load(ids + i, mask=i < n, other=0) - tl.load(start) + N) % N
    tl.store(out + i, tl.where(e < LOCAL, e, -1), mask=i < n)

"""
body = body.replace("    fused_post_expert_norm_weighted_sum,\n)", "    _post_expert_norm_weighted_sum_kernel,\n)", 1)
s0 = body.index("    return fused_post_expert_norm_weighted_sum(")
s1 = body.index("slot_expert_ids=local_ids)", s0) + len("slot_expert_ids=local_ids)")
body = body[:s0] + """    out = torch.empty((num_tokens, size_k), dtype=x.dtype, device=x.device)
    _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
        y, state["post_expert_w"], topk_weights, out, local_ids, TOPK=TOP_K, H=size_k, eps=POST_NORM_EPS,
        APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=size_k,
        num_warps=8 if size_k >= 4096 else 4, num_stages=2)
    return out""" + body[s1:]
body = body.replace('    local_ids = bank["expert_map"][topk_ids].contiguous()\n', "    local_ids = topk_ids\n", 1)
body = body.replace('    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS\n', """    local_ids = torch.empty_like(topk_ids)
    _local_ids[(1,)](topk_ids, data["local_start"], local_ids, tokens * TOP_K, N=N_EXPERTS, LOCAL=LOCAL_EXPERTS,
                     BLOCK=triton.next_power_of_2(tokens * TOP_K), num_warps=4)
""", 1)
body = body.replace("\ndef _dense(state, x, layer, n):", "\n" + LOCAL_IDS_KERNEL + "\ndef _dense(state, x, layer, n):", 1)
# Marlin's moe_block_size: 16 from 16 tokens up (SGLang picks up to 64), as timed on
# muse_moe_w4a16 under the judge's routings (bs 128: 302 vs 453 us, bs 256: 438 vs 640 us).
old_bm = """    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
"""
assert old_bm in body
body = body.replace(old_bm, "    block_size_m = 8 if num_tokens < 16 else 16\n", 1)
assert "fused_post_expert_norm_weighted_sum(" not in body and "_local_ids[" in body and "def _local_ids" in body

doc = '''"""SGLang's decoder layer (MuseSpark1xDecoderLayer.forward, as baseline.py runs it) with:

- the dense projections (qkv_gate, o, pooler, unpooler, router) on the muse_bf16_proj kernels:
  BF16 weights stored losslessly in 12 bits (W12) and decoded in registers, cuBLAS where
  that's faster, and the router as a BF16 hi + lo pair at every token count (see that
  module's notes below);
- the attention prologue fused into one Triton kernel: per-head QK RMSNorm, interleaved
  RoPE, the FP8 casts of q, k and v and the scattered cache store (SGLang: FusedQKNorm, two
  casts, store_cache, a q cast).
- the post-expert norm-sum always on SGLang's Triton kernel (its wrapper runs ~8 eager
  kernels below 32 tokens), and the local expert ids in one launch instead of three.
The rest (norms, routing, Marlin experts) is SGLang's."""
'''
for prob in ("muse_decode_layer",):
    out = f"# okc: problem={prob}\n" + doc + proj.replace('"""Dense projections', '_PROJ_NOTES = """Dense projections', 1) + "\n" + body
    compile(out, prob, "exec")
    if SPLIT:
        out = out.replace("- the attention prologue fused into one Triton kernel:", "- attention as three Triton launches (muse_attention_decode's): split-KV flash decode over\n  the paged FP8 cache with both KV heads per program, a combine fused with the gated output\n  norm, and a prologue fusing", 1)
    (root / prob / ("submission_split.py" if SPLIT else "submission.py")).write_text(out)
    print(prob, len(out.splitlines()), "lines")
