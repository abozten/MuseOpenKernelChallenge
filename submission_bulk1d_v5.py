# okc: problem=muse_bf16_proj
"""Skinny BF16 GEMMs for the dense projections: one stream-K CUDA kernel fed by TMA.

out[m, n] = x[m, k] @ W[n, k].T. At these token counts the projections are bound by reading
the weights, so the kernel is built to stream W from HBM as fast as the GPU allows:

- A CTA (4 warps) owns a tile of 64 * RT rows of W (the MMA's M side) and 8 * NT tokens (the
  N side). A unit of work is one tile over BKS of k.
- Bulk copies (cp.async.bulk, the 1D TMA) bring each row's BKS-wide chunk of W and of x into
  a STAGES-deep ring in shared memory, signalled with mbarriers. TMA requests aren't limited
  by the SM's per-thread load tracking, which is what holds 16-byte loads back here. Rows are
  padded by 64 bytes so the fragment reads below don't conflict on banks.
- Per 64 of k a thread reads 8 BF16 at k 8c and 8 at k 32 + 8c of its rows, and the same k of
  x. A dot product doesn't care about k's order, so the thread's words go to m16n8k16 in a
  permuted k order that A and B share, instead of the fragment layout's scattered columns.
- Stream-K: the grid is persistent, and CTA b takes an equal contiguous share of the
  flattened (tile, k unit) list, so every SM streams the same number of bytes whatever the
  shape. A tile a CTA covers whole is stored directly; a tile split between CTAs is summed in
  an FP32 workspace with atomics, and the CTA that completes its k (a counter) writes it and
  zeroes the workspace and counter for the next call. A call is one launch, CUDA-graph safe.

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
#define WARPS 4

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_copy(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

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
// Persistent grid of THREADS-thread CTAs with STAGES * (64 RT + 8 NT) * (2 BKS + 64) bytes of
// dynamic shared memory. ws: FP32 [M, N]; cnt: an int per tile; both zero between calls.
template <int NT, int RT, int BKS, int STAGES, bool OUT_F32>
__global__ void __launch_bounds__(THREADS) skinny_gemm(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, void* __restrict__ out,
    float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int KX) {
  constexpr int ROWS = 64 * RT, TOKS = 8 * NT;
  constexpr int ROW_BYTES = 2 * BKS + 64;  // == 64 mod 128: rows g and g + 1 hit disjoint banks
  constexpr int STAGE_BYTES = (ROWS + TOKS) * ROW_BYTES;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
  __shared__ int s_last;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c = lane & 3;
  const int n_blocks = N / ROWS;
  const int per_tile = K / BKS;  // k units per tile
  const int units = n_blocks * ((M + TOKS - 1) / TOKS) * per_tile;
  const int u_begin = (int)((long long)units * blockIdx.x / gridDim.x);
  const int u_end = (int)((long long)units * (blockIdx.x + 1) / gridDim.x);
  const int n_units = u_end - u_begin;
  if (n_units <= 0) return;

  const uint32_t s_base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(full);
  const uint32_t empty0 = (uint32_t)__cvta_generic_to_shared(empty);
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  // Warp 0 fills stage (i % STAGES) with unit u_begin + i: 64 RT rows of W, then the tile's
  // valid tokens of x, each a BKS-wide chunk.
  auto issue = [&](int i) {
    const int u = u_begin + i;
    const int tile = u / per_tile, kk = (u % per_tile) * BKS;
    const int nb = tile % n_blocks, tok0 = (tile / n_blocks) * TOKS;
    const int toks = min(TOKS, M - tok0);
    const uint32_t stage = s_base + (i % STAGES) * STAGE_BYTES;
    const uint32_t bar = full0 + 8 * (i % STAGES);
    if (lane == 0) mbar_expect_tx(bar, (ROWS + toks) * BKS * 2);
    __syncwarp();
    for (int r = lane; r < ROWS + toks; r += 32) {
      const __nv_bfloat16* src = r < ROWS ? w + (size_t)(nb * ROWS + r) * K + kk
                                          : x + (size_t)(tok0 + r - ROWS) * KX + kk % KX;
      bulk_copy(stage + r * ROW_BYTES, src, BKS * 2, bar);
    }
  };

  if (warp == 0)
    for (int i = 0; i < STAGES && i < n_units; ++i) issue(i);

  float acc[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[r][t][j] = 0.f;

  int steps = 0;  // k units of the current tile done by this CTA
  for (int i = 0; i < n_units; ++i) {
    const int u = u_begin + i;
    const int tile = u / per_tile;
    const int tok0 = (tile / n_blocks) * TOKS;
    const int st = i % STAGES;
    const uint32_t parity = (i / STAGES) & 1;
    mbar_wait(full0 + 8 * st, parity);

    const uint8_t* sw = smem + st * STAGE_BYTES + (warp * 16 * RT + g) * ROW_BYTES + 16 * c;
    const uint8_t* sx = smem + st * STAGE_BYTES + (ROWS + g) * ROW_BYTES + 16 * c;
    bool xv[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) xv[t] = tok0 + 8 * t + g < M;
#pragma unroll
    for (int kb = 0; kb < BKS / 64; ++kb) {
      uint4 wr[RT][2][2], xr[NT][2];
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < 2; ++q)
            wr[r][h][q] = *reinterpret_cast<const uint4*>(sw + (16 * r + 8 * h) * ROW_BYTES + 128 * kb + 64 * q);
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int q = 0; q < 2; ++q)
          xr[t][q] = xv[t] ? *reinterpret_cast<const uint4*>(sx + 8 * t * ROW_BYTES + 128 * kb + 64 * q)
                           : make_uint4(0, 0, 0, 0);
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
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * st);
    if (warp == 0 && i + STAGES < n_units) {
      mbar_wait(empty0 + 8 * st, parity);
      issue(i + STAGES);
    }

    ++steps;
    if ((u + 1) % per_tile != 0 && i + 1 < n_units) continue;

    // The tile ends here for this CTA: flush acc. acc[r][t][j] is row
    // nb ROWS + warp 16 RT + 16 r + g + 8 (j >> 1), token tok0 + 8 t + 2 c + (j & 1).
    const int row0 = (tile % n_blocks) * ROWS + warp * 16 * RT + g;
    if (steps == per_tile) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int tok = tok0 + 8 * t + 2 * c + (j & 1);
            if (tok < M) {
              const size_t o = (size_t)tok * N + row0 + 16 * r + 8 * (j >> 1);
              if (OUT_F32) reinterpret_cast<float*>(out)[o] = acc[r][t][j];
              else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(acc[r][t][j]);
            }
          }
    } else {
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int tok = tok0 + 8 * t + 2 * c + (j & 1);
            if (tok < M) atomicAdd(ws + (size_t)tok * N + row0 + 16 * r + 8 * (j >> 1), acc[r][t][j]);
          }
      __threadfence();
      __syncthreads();
      if (tid == 0) s_last = atomicAdd(cnt + tile, steps) + steps == per_tile;
      __syncthreads();
      if (s_last) {
        __threadfence();
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int tok = tok0 + 8 * t + 2 * c + (j & 1);
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
    }
    steps = 0;
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][t][j] = 0.f;
  }
}
"""

LAUNCH_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <int NT, int RT, int BKS, int ST, bool F32>
static void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                   torch::Tensor& ws, torch::Tensor& cnt, int M, int N, int K, int KX, int G) {
  constexpr int smem = ST * (64 * RT + 8 * NT) * (2 * BKS + 64);
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(skinny_gemm<NT, RT, BKS, ST, F32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  skinny_gemm<NT, RT, BKS, ST, F32><<<G, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), out.data_ptr(),
      ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, KX);
}

// (NT, RT, BKS, STAGES); keep in sync with VARIANTS in Python.
#define SKINNY_VARIANTS(X)                                                       \
  X(1, 1, 128, 4) X(1, 1, 128, 6) X(1, 1, 256, 4) X(1, 2, 128, 4)                \
  X(2, 1, 128, 4) X(2, 1, 128, 6) X(2, 1, 256, 4) X(2, 2, 128, 4)                \
  X(4, 1, 128, 4) X(4, 1, 128, 6) X(4, 2, 128, 3) X(4, 2, 128, 4)                \
  X(8, 1, 128, 4) X(8, 2, 128, 3)

template <bool F32>
static void dispatch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                     torch::Tensor& ws, torch::Tensor& cnt, int M, int N, int K, int KX, int G,
                     int NT, int RT, int BKS, int ST) {
#define CASE(nt, rt, bks, st)                                     \
  if (NT == nt && RT == rt && BKS == bks && ST == st)             \
    return launch<nt, rt, bks, st, F32>(x, w, out, ws, cnt, M, N, K, KX, G);
  SKINNY_VARIANTS(CASE)
#undef CASE
  TORCH_CHECK(false, "no skinny_gemm variant NT=", NT, " RT=", RT, " BKS=", BKS, " ST=", ST);
}

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws,
            torch::Tensor cnt, int64_t G, int64_t NT, int64_t RT, int64_t BKS, int64_t ST) {
  const int M = x.size(0), KX = x.size(1), N = out.size(1), K = w.size(1);
  if (out.scalar_type() == torch::kFloat32)
    dispatch<true>(x, w, out, ws, cnt, M, N, K, KX, G, NT, RT, BKS, ST);
  else
    dispatch<false>(x, w, out, ws, cnt, M, N, K, KX, G, NT, RT, BKS, ST);
}
"""

CPP_SRC = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, "
           "torch::Tensor cnt, int64_t G, int64_t NT, int64_t RT, int64_t BKS, int64_t ST);")

# (NT token tiles of 8, RT row tiles of 16 per warp, BKS k per unit, STAGES), as LAUNCH_SRC
# instantiates them.
VARIANTS = [(1, 1, 128, 4), (1, 1, 128, 6), (1, 1, 256, 4), (1, 2, 128, 4),
            (2, 1, 128, 4), (2, 1, 128, 6), (2, 1, 256, 4), (2, 2, 128, 4),
            (4, 1, 128, 4), (4, 1, 128, 6), (4, 2, 128, 3), (4, 2, 128, 4),
            (8, 1, 128, 4), (8, 2, 128, 3)]
SMEM_PER_SM = 228 * 1024
WS_FLOATS = 1 << 21
N_COUNTERS = 1 << 12

LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (8192, 4096), "router": (256, 8192)}

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline("muse_skinny_gemm_tma", cpp_sources=CPP_SRC,
                           cuda_sources=KERNEL_SRC + LAUNCH_SRC, functions=["skinny"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                           verbose=False)
    return _ext


def _nt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def _smem(v):
    nt, rt, bks, st = v
    return st * (64 * rt + 8 * nt) * (2 * bks + 64)


def _ctas_per_sm(v):
    return max(1, SMEM_PER_SM // (_smem(v) + 1024))


def _valid(m, n, k, router, v):
    nt, rt, bks, st = v
    if n % (64 * rt) or k % bks or (router and (k // 2) % bks):
        return False
    tiles = n // (64 * rt) * -(-m // (8 * nt))
    return m * n <= WS_FLOATS and tiles <= N_COUNTERS


def _units(m, n, k, v):
    nt, rt, bks, _ = v
    return n // (64 * rt) * -(-m // (8 * nt)) * (k // bks)


def _grids(m, n, k, v, sms):
    """Grid sizes to try: whole waves of resident CTAs, at most one unit each."""
    units = _units(m, n, k, v)
    return sorted({min(units, sms * j) for j in range(1, _ctas_per_sm(v) + 1)})


def _default(state, m, n, k, router):
    """For shapes without a tuned entry: the first fitting variant, one full wave."""
    nt = _nt(m)
    for v in VARIANTS:
        if v[0] == nt and _valid(m, n, k, router, v):
            return v + (_grids(m, n, k, v, state["sms"])[-1],)
    raise ValueError(f"no variant for m={m} n={n} k={k}")


def _launch(state, x, layer, n, cfg):
    nt, rt, bks, st, grid = cfg
    out = torch.empty((x.shape[0], n), device=x.device,
                      dtype=torch.float32 if layer == "router" else torch.bfloat16)
    state["ext"].skinny(x, state["w"][layer], out, state["ws"], state["cnt"], grid, nt, rt, bks, st)
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
                for v in VARIANTS:
                    if v[0] != nt or not _valid(m, n, k, router, v):
                        continue
                    for grid in _grids(m, n, k, v, state["sms"]):
                        cfg = v + (grid,)
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
        "sms": torch.cuda.get_device_properties(router.device).multi_processor_count,
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
            cfg = _default(state, m, n, state["w"][layer].shape[1], layer == "router")
    if cfg == "cublas":
        return _cublas(state, x, layer, n)
    return _launch(state, x, layer, n, cfg)
