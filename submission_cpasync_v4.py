# okc: problem=muse_bf16_proj
"""Skinny BF16 GEMMs for the dense projections: one split-K CUDA kernel.

out[m, n] = x[m, k] @ W[n, k].T. At these token counts the projections are bound by reading
the weights, so the kernel streams W from HBM into mma.sync as fast as it can:

- Each warp owns 16 * RT rows of W (the MMA's M side); tokens are the N side, in tiles of 8.
- Per 64-wide k step a thread copies 8 BF16 at k 8c and 8 at k 32 + 8c of each of its rows
  (so each warp copy is 64 contiguous bytes per row), and the same 16 k of x. A dot product
  doesn't care about k's order, so the thread's words go to m16n8k16 in a permuted k order
  that A and B share, instead of the fragment layout's scattered 4-byte columns.
- The copies are cp.async into a STAGES-deep ring in shared memory. Each thread reads back
  only what it copied itself, so the ring needs cp.async.wait_group and no CTA barriers, and
  keeps STAGES - 1 steps of W in flight per thread whatever the register pressure.
- K is split over CTAs to fill the GPU. Partial sums go to an FP32 workspace with atomics;
  the last CTA of each row tile (a counter) writes the output and zeroes the workspace and
  counter for the next call, so a call is one launch and CUDA-graph safe.

The router's FP32 weights run as a BF16 hi + lo pair with FP32 accumulation (allowed at any
batch size), stored as one [256, 2 * 8192] matrix [hi | lo]: one GEMM over K = 16384, with
x's column taken modulo 8192.
"""

import time
from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


KERNEL_SRC = r"""
#include <cuda_bf16.h>
#include <stdint.h>

#define THREADS 128
#define K_STEP 64

__device__ __forceinline__ void cp_async_cg(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_ca(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// x: [M, KX] BF16, w: [N, K] BF16, out: [M, N] (BF16, or FP32 when OUT_F32).
// grid (N / (64 RT), S, ceil(M / (8 NT))), THREADS threads, STAGES * SLOTS * 16 * THREADS
// bytes of dynamic shared memory. ws: FP32 [M, N]; cnt: an int per (z, x) tile; both zero
// between calls.
template <int NT, int RT, int STAGES, bool OUT_F32>
__global__ void __launch_bounds__(THREADS) skinny_gemm(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, void* __restrict__ out,
    float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int KX, int S) {
  // A thread's 16-byte slots per stage: W rows (tile r, half h = row g or g + 8, part q = k 8c or
  // 32 + 8c) at 4r + 2h + q, then x (token tile t, part q) at 4 RT + 2t + q.
  constexpr int SLOTS = 4 * RT + 2 * NT;
  extern __shared__ uint4 smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c = lane & 3;
  const int row0 = blockIdx.x * 64 * RT + warp * 16 * RT + g;
  const int kc = K / S;
  const int k0 = blockIdx.y * kc;
  const int tok0 = blockIdx.z * 8 * NT;
  const int iters = kc / K_STEP;

  const __nv_bfloat16* wp = w + (size_t)row0 * K + k0 + 8 * c;
  const __nv_bfloat16* xp[NT];
  bool xv[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tok = tok0 + t * 8 + g;
    xv[t] = tok < M;
    xp[t] = x + (size_t)(xv[t] ? tok : 0) * KX + (k0 % KX) + 8 * c;
  }
  const uint32_t s_base = (uint32_t)__cvta_generic_to_shared(smem) + tid * 16;

  auto issue = [&](int step) {
    const int off = step * K_STEP;
    const uint32_t s = s_base + (step % STAGES) * SLOTS * THREADS * 16;
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int q = 0; q < 2; ++q)
          cp_async_cg(s + (4 * r + 2 * h + q) * THREADS * 16,
                      wp + (size_t)(16 * r + 8 * h) * K + off + 32 * q);
#pragma unroll
    for (int t = 0; t < NT; ++t)
      if (xv[t]) {
#pragma unroll
        for (int q = 0; q < 2; ++q)
          cp_async_ca(s + (4 * RT + 2 * t + q) * THREADS * 16, xp[t] + off + 32 * q);
      }
  };

  float acc[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[r][t][j] = 0.f;

#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    if (st < iters) issue(st);
    cp_async_commit();
  }
  for (int it = 0; it < iters; ++it) {
    if (it + STAGES - 1 < iters) issue(it + STAGES - 1);
    cp_async_commit();
    cp_async_wait<STAGES - 1>();
    const uint4* s = smem + (it % STAGES) * SLOTS * THREADS + tid;
    uint4 wr[RT][2][2], xr[NT][2];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int q = 0; q < 2; ++q) wr[r][h][q] = s[(4 * r + 2 * h + q) * THREADS];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int q = 0; q < 2; ++q)
        xr[t][q] = xv[t] ? s[(4 * RT + 2 * t + q) * THREADS] : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      // k step ks takes the thread's words 2 (ks & 1) and 2 (ks & 1) + 1 of part q = ks >> 1.
      const int q = ks >> 1, i0 = (ks & 1) * 2;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const uint32_t a0 = word(wr[r][0][q], i0), a2 = word(wr[r][0][q], i0 + 1);
        const uint32_t a1 = word(wr[r][1][q], i0), a3 = word(wr[r][1][q], i0 + 1);
#pragma unroll
        for (int t = 0; t < NT; ++t)
          mma_bf16(acc[r][t], a0, a1, a2, a3, word(xr[t][q], i0), word(xr[t][q], i0 + 1));
      }
    }
  }

  // acc[r][t][j]: row row0 + 16r + 8 (j >> 1), token tok0 + 8t + 2c + (j & 1).
  if (S == 1) {
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int tok = tok0 + t * 8 + 2 * c + (j & 1);
          if (tok < M) {
            const size_t o = (size_t)tok * N + row0 + 16 * r + 8 * (j >> 1);
            if (OUT_F32) reinterpret_cast<float*>(out)[o] = acc[r][t][j];
            else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(acc[r][t][j]);
          }
        }
    return;
  }

#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int tok = tok0 + t * 8 + 2 * c + (j & 1);
        if (tok < M) atomicAdd(ws + (size_t)tok * N + row0 + 16 * r + 8 * (j >> 1), acc[r][t][j]);
      }
  __threadfence();
  __syncthreads();
  __shared__ int last;
  const int tile = blockIdx.z * gridDim.x + blockIdx.x;
  if (tid == 0) last = atomicAdd(cnt + tile, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int tok = tok0 + t * 8 + 2 * c + (j & 1);
        if (tok < M) {
          const size_t o = (size_t)tok * N + row0 + 16 * r + 8 * (j >> 1);
          const float v = __ldcg(ws + o);
          ws[o] = 0.f;
          if (OUT_F32) reinterpret_cast<float*>(out)[o] = v;
          else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(v);
        }
      }
  if (tid == 0) cnt[tile] = 0;
}
"""

LAUNCH_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <int NT, int RT, int ST, bool F32>
static void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                   torch::Tensor& ws, torch::Tensor& cnt, int M, int N, int K, int KX, int S) {
  constexpr int smem = ST * (4 * RT + 2 * NT) * 16 * THREADS;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(skinny_gemm<NT, RT, ST, F32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  dim3 grid(N / (64 * RT), S, (M + 8 * NT - 1) / (8 * NT));
  skinny_gemm<NT, RT, ST, F32><<<grid, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), out.data_ptr(),
      ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, KX, S);
}

// (NT, RT, STAGES); keep in sync with VARIANTS in Python.
#define SKINNY_VARIANTS(X)                                                                 \
  X(1, 1, 4) X(1, 1, 8) X(1, 2, 4) X(1, 2, 8) X(2, 1, 6) X(2, 2, 4) X(2, 2, 6) X(2, 4, 4) \
  X(4, 2, 4) X(4, 2, 6) X(4, 4, 3) X(4, 4, 4) X(8, 1, 4) X(8, 2, 3) X(8, 2, 4)

template <bool F32>
static void dispatch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                     torch::Tensor& ws, torch::Tensor& cnt, int M, int N, int K, int KX, int S,
                     int NT, int RT, int ST) {
#define CASE(nt, rt, st) \
  if (NT == nt && RT == rt && ST == st) return launch<nt, rt, st, F32>(x, w, out, ws, cnt, M, N, K, KX, S);
  SKINNY_VARIANTS(CASE)
#undef CASE
  TORCH_CHECK(false, "no skinny_gemm variant NT=", NT, " RT=", RT, " ST=", ST);
}

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws,
            torch::Tensor cnt, int64_t S, int64_t NT, int64_t RT, int64_t ST) {
  const int M = x.size(0), KX = x.size(1), N = out.size(1), K = w.size(1);
  if (out.scalar_type() == torch::kFloat32)
    dispatch<true>(x, w, out, ws, cnt, M, N, K, KX, S, NT, RT, ST);
  else
    dispatch<false>(x, w, out, ws, cnt, M, N, K, KX, S, NT, RT, ST);
}
"""

CPP_SRC = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, "
           "torch::Tensor cnt, int64_t S, int64_t NT, int64_t RT, int64_t ST);")

# (NT token tiles of 8, RT row tiles of 16 per warp, STAGES), as LAUNCH_SRC instantiates them.
VARIANTS = [(1, 1, 4), (1, 1, 8), (1, 2, 4), (1, 2, 8), (2, 1, 6), (2, 2, 4), (2, 2, 6), (2, 4, 4),
            (4, 2, 4), (4, 2, 6), (4, 4, 3), (4, 4, 4), (8, 1, 4), (8, 2, 3), (8, 2, 4)]
SPLITS = (1, 2, 4, 8, 16, 32, 64, 128)
WS_FLOATS = 1 << 21
N_COUNTERS = 1 << 12

LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (8192, 4096), "router": (256, 8192)}

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline("muse_skinny_gemm", cpp_sources=CPP_SRC,
                           cuda_sources=KERNEL_SRC + LAUNCH_SRC, functions=["skinny"],
                           extra_cuda_cflags=["-O3"], verbose=False)
    return _ext


def _nt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def _valid(m, n, k, router, cfg):
    nt, rt, _, split = cfg
    if n % (64 * rt) or k % split or (k // split) % 64:
        return False
    if router and split % 2:  # each split must sit inside one half of [hi | lo]
        return False
    if split > 1 and (m * n > WS_FLOATS or n // (64 * rt) * -(-m // (8 * nt)) > N_COUNTERS):
        return False
    return True


def _ctas(m, n, cfg):
    nt, rt, _, split = cfg
    return n // (64 * rt) * split * -(-m // (8 * nt))


def _heuristic(m, n, k, router):
    """A config giving a few hundred CTAs, for shapes without a tuned entry."""
    nt = _nt(m)
    best = None
    for vnt, rt, st in VARIANTS:
        if vnt != nt:
            continue
        for split in SPLITS:
            cfg = (nt, rt, st, split)
            if _valid(m, n, k, router, cfg):
                score = abs(_ctas(m, n, cfg) - 400)
                if best is None or score < best[0]:
                    best = (score, cfg)
    return best[1]


def _launch(state, x, layer, n, cfg):
    nt, rt, st, split = cfg
    out = torch.empty((x.shape[0], n), device=x.device,
                      dtype=torch.float32 if layer == "router" else torch.bfloat16)
    state["ext"].skinny(x, state["w"][layer], out, state["ws"], state["cnt"], split, nt, rt, st)
    return out


def _cublas(state, x, layer, n):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return x @ state["w"][layer][:n].T


def _n(layer, m):
    return 1024 if layer == "unpooler" and m <= 8 else LAYERS[layer][0]


def _time(fn, flush, reps=25):
    """Median µs of one CUDA-graph replay of fn after a cold L2, as the judge times it."""
    fn()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


SCORED_M = [1, 8, 16, 32, 128, 256]


def _tune(state):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    table = {}
    t_start = time.time()
    for m in SCORED_M:
        for layer, (_, kk) in LAYERS.items():
            router = layer == "router"
            n = _n(layer, m)
            k = state["w"][layer].shape[1]
            x = torch.randn((m, kk), device="cuda").to(torch.bfloat16)
            ref = _cublas(state, x, layer, n).float()
            res = [("cublas", _time(lambda: _cublas(state, x, layer, n), flush))]
            if m <= 64 or router:
                nt = _nt(m)
                for vnt, rt, st in VARIANTS:
                    if vnt != nt:
                        continue
                    for split in SPLITS:
                        cfg = (nt, rt, st, split)
                        if not _valid(m, n, k, router, cfg) or not 32 <= _ctas(m, n, cfg) <= 4096:
                            continue
                        err = float((_launch(state, x, layer, n, cfg).float() - ref).norm() / ref.norm())
                        if err > 1e-2:
                            print(f"  BAD {m} {layer} {cfg}: err {err:.2e}", flush=True)
                            continue
                        res.append((cfg, _time(lambda: _launch(state, x, layer, n, cfg), flush)))
            res.sort(key=lambda r: r[1])
            table[(m, layer)] = res[0][0]
            cub = [t for c, t in res if c == "cublas"][0]
            top = ", ".join(f"{c}:{t:.1f}" for c, t in res[:4])
            print(f"TUNE m={m} {layer}: cublas {cub:.1f}us | best {top}", flush=True)
    print(f"TUNE took {time.time() - t_start:.0f}s", flush=True)
    del flush
    return table


def prepare(weights: Weights):
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    t0 = time.time()
    ext = _module()
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)
    state = {
        "ext": ext,
        "w": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "ws": torch.zeros(WS_FLOATS, dtype=torch.float32, device=router.device),
        "cnt": torch.zeros(N_COUNTERS, dtype=torch.int32, device=router.device),
    }
    state["w"]["router"] = torch.cat([hi, lo], dim=1).contiguous()
    state["table"] = _tune(state)
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg_ = data["config"]
    layer, n, m = cfg_["layer"], cfg_["n"], cfg_["m"]
    cfg = state["table"].get((m, layer))
    if cfg is None:
        if m > 64 and layer != "router":
            cfg = "cublas"
        else:
            cfg = _heuristic(m, n, state["w"][layer].shape[1], layer == "router")
    if cfg == "cublas":
        return _cublas(state, x, layer, n)
    return _launch(state, x, layer, n, cfg)
