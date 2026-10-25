# okc: problem=muse_bf16_proj
"""Skinny BF16 GEMMs for the dense projections: one stream-K CUDA kernel fed by TMA.

out[m, n] = x[m, k] @ W[n, k].T. At these token counts the projections are bound by reading
the weights, so the kernel is built to stream W from HBM as fast as the GPU allows:

- A CTA (4 warps) owns a tile of 64 * RT rows of W (the MMA's M side) and 8 * NT tokens (the
  N side). A unit of work is one tile over BKS of k.
- TMA brings each unit's rows of W and of x into a STAGES-deep ring in shared memory,
  signalled with mbarriers, 128-byte swizzled. (16-byte loads, and 1D bulk copies of a row at
  a time, are held back by per-request overheads at these sizes.) The boxes are either 2D, 64
  of k by the tile's rows, BKS / 64 of them a unit, or (BOX3D) one 3D box per unit whose rows
  are each BKS of k, contiguous in memory. TMA zero-fills tokens past M.
- Per 64 of k a thread reads 16 BF16 of each of its rows, and the same k of x. A dot product
  doesn't care about k's order, so the thread's words go to m16n8k16 in a permuted k order
  that A and B share, instead of the fragment layout's scattered columns. The permutation is
  picked so that, with the swizzle, the reads are free of bank conflicts.
- Stream-K: the grid is persistent, and CTA b takes an equal contiguous share of the
  flattened (tile, k unit) list, so every SM streams the same number of bytes whatever the
  shape. A tile a CTA covers whole is stored directly. A tile split between CTAs is reduced
  without atomics on the data: each contributor stores its FP32 partial in its own slot (its
  rank among the tile's CTAs, which the split formula gives) and counts its k units on the
  tile's counter; the one that completes the count sums the slots and writes the tile, and
  resets the counter for the next call. A CTA does its last tile's share first, so that share
  is published early, and the tile it finishes on usually needs no publishing at all: when
  the counter already holds every other share, it sums them at once. A call is one launch,
  CUDA-graph safe.

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
#include <cuda.h>
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
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                       uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
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

// One region of a stage (W's rows or x's tokens) as TMA lays it out, 128-byte lines swizzled
// by XOR-ing a line's 16-byte chunk index with its line index mod 8.
template <int BKS, bool BOX3D>
struct Region {
  // Byte offset of 16-byte chunk q (0 or 1) of thread column c, for 64-k block kb of row rr.
  // 2D boxes: [kb][row][128 B], a thread takes chunks 2c + q (32 contiguous bytes); rows g and
  // g + 1 differ in the XOR's low bit, so a quarter warp hits all 32 banks once.
  // 3D boxes: [row][kb][128 B], a thread takes chunks c + 4q; with BKS = 256 rows g and g + 1
  // differ in the XOR's bit 2, to the same effect.
  template <int R>
  static __device__ __forceinline__ int off(int rr, int kb, int c, int q) {
    if (BOX3D) {
      const int line = rr * (BKS / 64) + kb;
      return line * 128 + (((c + 4 * q) ^ (line & 7)) << 4);
    }
    return (kb * R + rr) * 128 + (((2 * c + q) ^ (rr & 7)) << 4);
  }
};

template <int NT, int RT, int BKS>
struct Layout {
  static constexpr int ROWS = 64 * RT, TOKS = 8 * NT;
  static constexpr int X_OFF = ROWS * BKS * 2;
  static constexpr int STAGE = X_OFF + TOKS * BKS * 2;  // a multiple of 1024
};

// wmap: TMA map of w [N, K] BF16; xmap: of x [M, KX] BF16 (boxes as Region describes, ROWS
// or TOKS rows). out: [M, N] (BF16, or FP32 when OUT_F32). Persistent grid of THREADS-thread
// CTAs with STAGES * Layout::STAGE + 1024 bytes of dynamic shared memory. ws: FP32 partials,
// [tile][MAXC slots][THREADS][RT NT 4]; cnt: an int per tile, zero between calls.
template <int NT, int RT, int BKS, int STAGES, bool BOX3D, bool OUT_F32>
__global__ void __launch_bounds__(THREADS) skinny_gemm(
    const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap,
    void* __restrict__ out, float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K,
    int KX, int MAXC, unsigned long long* __restrict__ dbg) {
  // dbg (tuning only, else null): per CTA, ns at start, first data, last unit done, end.
  const unsigned long long t_start = dbg ? global_ns() : 0;
  using L = Layout<NT, RT, BKS>;
  using Rg = Region<BKS, BOX3D>;
  constexpr int ROWS = L::ROWS, TOKS = L::TOKS, X_OFF = L::X_OFF, STAGE_BYTES = L::STAGE;
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
  __shared__ int s_flag;
  constexpr int FRAG = RT * NT * 4;  // a thread's accumulators

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c = lane & 3;
  const int n_blocks = N / ROWS;
  const int per_tile = K / BKS;  // k units per tile
  const int units = n_blocks * ((M + TOKS - 1) / TOKS) * per_tile;
  const int u_begin = (int)((long long)units * blockIdx.x / gridDim.x);
  const int u_end = (int)((long long)units * (blockIdx.x + 1) / gridDim.x);
  const int n_units = u_end - u_begin;
  if (n_units <= 0) return;
  // Work order: the share of the last tile first, then the rest in order.
  const int last_start = max(u_begin, (u_end - 1) / per_tile * per_tile);
  const int n_last = u_end - last_start;
  auto unit_of = [&](int i) { return i < n_last ? last_start + i : u_begin + (i - n_last); };

  // The 128-byte swizzle repeats every 1024 bytes: align the ring to that.
  const uint32_t raw_base = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t s_base = (raw_base + 1023) & ~1023u;
  uint8_t* smem = smem_raw + (s_base - raw_base);
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

  // Thread 0 fills stage (i % STAGES) with unit u_begin + i: the tile's rows of W, then its
  // tokens of x (zero past M), BKS of k each.
  auto issue = [&](int i) {
    const int u = unit_of(i);
    const int tile = u / per_tile, kk = (u % per_tile) * BKS;
    const int nb = tile % n_blocks, tok0 = (tile / n_blocks) * TOKS;
    const uint32_t stage = s_base + (i % STAGES) * STAGE_BYTES;
    const uint32_t bar = full0 + 8 * (i % STAGES);
    const int kx = kk % KX;
    mbar_expect_tx(bar, STAGE_BYTES);
    if (BOX3D) {
      tma_3d(stage, &wmap, 0, kk / 64, nb * ROWS, bar);
      tma_3d(stage + X_OFF, &xmap, 0, kx / 64, tok0, bar);
    } else {
#pragma unroll
      for (int kb = 0; kb < BKS / 64; ++kb) {
        tma_2d(stage + kb * ROWS * 128, &wmap, kk + 64 * kb, nb * ROWS, bar);
        tma_2d(stage + X_OFF + kb * TOKS * 128, &xmap, kx + 64 * kb, tok0, bar);
      }
    }
  };

  if (tid == 0)
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
    const int u = unit_of(i);
    const int tile = u / per_tile;
    const int tok0 = (tile / n_blocks) * TOKS;
    const int st = i % STAGES;
    const uint32_t parity = (i / STAGES) & 1;
    mbar_wait(full0 + 8 * st, parity);
    if (dbg && i == 0 && threadIdx.x == 0) {
      dbg[4 * blockIdx.x] = t_start;
      dbg[4 * blockIdx.x + 1] = global_ns();
    }

    const uint8_t* sw = smem + st * STAGE_BYTES;
    const uint8_t* sx = sw + X_OFF;
#pragma unroll
    for (int kb = 0; kb < BKS / 64; ++kb) {
      uint4 wr[RT][2][2], xr[NT][2];
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < 2; ++q)
            wr[r][h][q] = *reinterpret_cast<const uint4*>(
                sw + Rg::template off<ROWS>(warp * 16 * RT + 16 * r + 8 * h + g, kb, c, q));
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int q = 0; q < 2; ++q)
          xr[t][q] = *reinterpret_cast<const uint4*>(sx + Rg::template off<TOKS>(8 * t + g, kb, c, q));
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        // k step ks takes the thread's words 2 (ks & 1) and 2 (ks & 1) + 1 of chunk q = ks >> 1.
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
    if (tid == 0 && i + STAGES < n_units) {
      mbar_wait(empty0 + 8 * st, parity);
      issue(i + STAGES);
    }

    ++steps;
    if (dbg && i + 1 == n_units && threadIdx.x == 0) dbg[4 * blockIdx.x + 2] = global_ns();
    if (i + 1 < n_units && unit_of(i + 1) / per_tile == tile) continue;

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
      // The tile's contributors are CTAs first..last of the split; this one is rank b - first.
      const long long G = gridDim.x, U = units, t0u = (long long)tile * per_tile;
      const int first = (int)(((t0u + 1) * G + U - 1) / U) - 1;
      const int last = (int)(((t0u + per_tile) * G + U - 1) / U) - 1;
      float* slots = ws + (size_t)tile * MAXC * THREADS * FRAG + (size_t)tid * FRAG;
      // If every other share is already counted, finish the tile without publishing ours.
      if (tid == 0) s_flag = ld_acquire(cnt + tile) == per_tile - steps;
      __syncthreads();
      bool finish = s_flag;
      if (!finish) {
        float* mine = slots + (size_t)(blockIdx.x - first) * THREADS * FRAG;
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int t = 0; t < NT; ++t)
            *reinterpret_cast<float4*>(mine + (r * NT + t) * 4) =
                make_float4(acc[r][t][0], acc[r][t][1], acc[r][t][2], acc[r][t][3]);
        __threadfence();
        __syncthreads();
        if (tid == 0) s_flag = atomicAdd(cnt + tile, steps) + steps == per_tile;
        __syncthreads();
        finish = s_flag;
        if (finish) __threadfence();
      }
      if (finish) {
        for (int b = first; b <= last; ++b) {
          if (b == (int)blockIdx.x) continue;
          const float* other = slots + (size_t)(b - first) * THREADS * FRAG;
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              const float4 v = __ldcg(reinterpret_cast<const float4*>(other + (r * NT + t) * 4));
              acc[r][t][0] += v.x; acc[r][t][1] += v.y; acc[r][t][2] += v.z; acc[r][t][3] += v.w;
            }
        }
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
        if (tid == 0) cnt[tile] = 0;
      }
      // s_flag is rewritten at the next flush only after a barrier every thread passes here.
      __syncthreads();
    }
    steps = 0;
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][t][j] = 0.f;
  }
  if (dbg && threadIdx.x == 0) dbg[4 * blockIdx.x + 3] = global_ns();
}
"""

LAUNCH_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cudaTypedefs.h>

// A TMA map of row-major BF16 [rows, cols] at ptr, 128-byte swizzled: 2D boxes of 64 cols x
// box_rows, or (box3d) 3D boxes of box_k cols x box_rows, the cols split into 64-wide blocks.
static CUtensorMap encode_map(void* ptr, int64_t rows, int64_t cols, int64_t box_rows,
                              int64_t box_k, bool box3d) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
  if (!encode) {
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&encode,
                                        cudaEnableDefault, &q) == cudaSuccess && encode);
  }
  CUtensorMap map;
  CUresult r;
  cuuint32_t elem[3] = {1, 1, 1};
  if (box3d) {
    cuuint64_t dims[3] = {64, (cuuint64_t)cols / 64, (cuuint64_t)rows};
    cuuint64_t strides[2] = {128, (cuuint64_t)cols * 2};
    cuuint32_t box[3] = {64, (cuuint32_t)box_k / 64, (cuuint32_t)box_rows};
    r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr, dims, strides, box, elem,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  } else {
    cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
    cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
    r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, elem,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return map;
}

// W's map, made once: its bytes in a CPU tensor.
torch::Tensor make_wmap(torch::Tensor w, int64_t box_rows, int64_t box_k, bool box3d) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.is_contiguous());
  CUtensorMap map = encode_map(w.data_ptr(), w.size(0), w.size(1), box_rows, box_k, box3d);
  auto host = torch::empty({(int64_t)sizeof(CUtensorMap)}, torch::kUInt8);
  memcpy(host.data_ptr(), &map, sizeof(map));
  return host;
}

template <int NT, int RT, int BKS, int ST, bool B3, bool F32>
static void launch(const CUtensorMap& wmap, const CUtensorMap& xmap, torch::Tensor& out,
                   torch::Tensor& ws, torch::Tensor& cnt, int M, int N, int K, int KX, int G,
                   int MAXC, unsigned long long* dbg) {
  constexpr int smem = ST * Layout<NT, RT, BKS>::STAGE + 1024;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(skinny_gemm<NT, RT, BKS, ST, B3, F32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  skinny_gemm<NT, RT, BKS, ST, B3, F32><<<G, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      wmap, xmap, out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, KX, MAXC, dbg);
}

// (NT, RT, BKS, STAGES, BOX3D); keep in sync with VARIANTS in Python.
#define SKINNY_VARIANTS(X)                                                                     \
  X(1, 1, 64, 16, 0) X(1, 1, 128, 8, 0) X(1, 2, 64, 12, 0) X(1, 1, 256, 4, 1)                    \
  X(2, 1, 64, 16, 0) X(2, 1, 128, 8, 0) X(2, 2, 64, 10, 0) X(2, 1, 256, 4, 1)                    \
  X(4, 1, 64, 12, 0) X(4, 1, 128, 6, 0) X(4, 2, 64, 8, 0) X(4, 1, 256, 4, 1)                     \
  X(8, 1, 64, 12, 0) X(8, 2, 64, 8, 0) X(8, 1, 256, 3, 1)

void skinny(torch::Tensor x, torch::Tensor wmap_bytes, torch::Tensor out, torch::Tensor ws,
            torch::Tensor cnt, int64_t K, int64_t G, int64_t NT, int64_t RT, int64_t BKS,
            int64_t ST, bool B3, int64_t MAXC, c10::optional<torch::Tensor> dbg_t) {
  unsigned long long* dbg =
      dbg_t ? reinterpret_cast<unsigned long long*>(dbg_t->data_ptr<int64_t>()) : nullptr;
  const int M = x.size(0), KX = x.size(1), N = out.size(1);
  CUtensorMap wmap;
  memcpy(&wmap, wmap_bytes.data_ptr(), sizeof(wmap));
  const CUtensorMap xmap = encode_map(x.data_ptr(), M, KX, 8 * NT, BKS, B3);
  const bool f32 = out.scalar_type() == torch::kFloat32;
#define CASE(nt, rt, bks, st, b3)                                                         \
  if (NT == nt && RT == rt && BKS == bks && ST == st && B3 == (bool)b3) {                 \
    if (f32) launch<nt, rt, bks, st, (bool)b3, true>(wmap, xmap, out, ws, cnt, M, N, K, KX, G, MAXC, dbg); \
    else launch<nt, rt, bks, st, (bool)b3, false>(wmap, xmap, out, ws, cnt, M, N, K, KX, G, MAXC, dbg);    \
    return;                                                                               \
  }
  SKINNY_VARIANTS(CASE)
#undef CASE
  TORCH_CHECK(false, "no skinny_gemm variant NT=", NT, " RT=", RT, " BKS=", BKS, " ST=", ST);
}
"""

CPP_SRC = ("void skinny(torch::Tensor x, torch::Tensor wmap_bytes, torch::Tensor out, "
           "torch::Tensor ws, torch::Tensor cnt, int64_t K, int64_t G, int64_t NT, int64_t RT, "
           "int64_t BKS, int64_t ST, bool B3, int64_t MAXC, c10::optional<torch::Tensor> dbg_t);\n"
           "torch::Tensor make_wmap(torch::Tensor w, int64_t box_rows, int64_t box_k, bool box3d);")

# (NT token tiles of 8, RT row tiles of 16 per warp, BKS k per unit, STAGES, BOX3D), as
# LAUNCH_SRC instantiates them.
VARIANTS = [(1, 1, 64, 16, 0), (1, 1, 128, 8, 0), (1, 2, 64, 12, 0), (1, 1, 256, 4, 1),
            (2, 1, 64, 16, 0), (2, 1, 128, 8, 0), (2, 2, 64, 10, 0), (2, 1, 256, 4, 1),
            (4, 1, 64, 12, 0), (4, 1, 128, 6, 0), (4, 2, 64, 8, 0), (4, 1, 256, 4, 1),
            (8, 1, 64, 12, 0), (8, 2, 64, 8, 0), (8, 1, 256, 3, 1)]
SMEM_PER_SM = 228 * 1024
WS_FLOATS = 1 << 22
N_COUNTERS = 1 << 12

LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (8192, 4096), "router": (256, 8192)}

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline("muse_skinny_gemm_tma4", cpp_sources=CPP_SRC,
                           cuda_sources=KERNEL_SRC + LAUNCH_SRC, functions=["skinny", "make_wmap"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                           verbose=False)
    return _ext


def _nt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def _smem(v):
    """Dynamic shared memory of a variant, as Layout and launch compute it."""
    nt, rt, bks, st, _ = v
    return st * (64 * rt + 8 * nt) * bks * 2 + 1024


def _ctas_per_sm(v):
    return max(1, SMEM_PER_SM // (_smem(v) + 1024))


def _valid(m, n, k, router, v):
    nt, rt, bks, st, _ = v
    if n % (64 * rt) or k % bks or (router and (k // 2) % bks):
        return False
    return n // (64 * rt) * -(-m // (8 * nt)) <= N_COUNTERS


def _maxc(m, n, k, v, grid):
    """Most CTAs sharing a tile under the kernel's split: the partial slots per tile."""
    nt, rt, bks = v[:3]
    tiles = n // (64 * rt) * -(-m // (8 * nt))
    per_tile = k // bks
    units = tiles * per_tile
    return max(-(-((t * per_tile + per_tile) * grid) // units) - -(-((t * per_tile + 1) * grid) // units) + 1
               for t in range(tiles))


def _fits(m, n, k, v, grid):
    nt, rt = v[:2]
    tiles = n // (64 * rt) * -(-m // (8 * nt))
    return tiles * _maxc(m, n, k, v, grid) * 128 * rt * nt * 4 <= WS_FLOATS


def _units(m, n, k, v):
    nt, rt, bks = v[:3]
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
            for grid in reversed(_grids(m, n, k, v, state["sms"])):
                if _fits(m, n, k, v, grid):
                    return v + (grid,)
    raise ValueError(f"no variant for m={m} n={n} k={k}")


def _launch(state, x, layer, n, cfg, dbg=None):
    nt, rt, bks, st, b3, grid = cfg
    out = torch.empty((x.shape[0], n), device=x.device,
                      dtype=torch.float32 if layer == "router" else torch.bfloat16)
    k = state["w"][layer].shape[1]
    maxc = state["maxc"].get((x.shape[0], n, k, cfg))
    if maxc is None:
        maxc = state["maxc"][(x.shape[0], n, k, cfg)] = _maxc(x.shape[0], n, k, cfg, grid)
    state["ext"].skinny(x.contiguous(), state["wmap"][(layer, rt, bks, b3)], out, state["ws"], state["cnt"],
                        k, grid, nt, rt, bks, st, bool(b3), maxc, dbg)
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
    # The fixed cost of an empty graph replay, to read the times below as kernel time.
    nop = torch.zeros(1, device="cuda")
    print(f"TUNE empty graph {_time(lambda: nop.add_(1), flush):.1f}us", flush=True)
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
                        if not _fits(m, n, k, v, grid):
                            continue
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
    for m, layer in [(1, "pooler"), (1, "o"), (16, "o"), (16, "pooler"), (32, "qkv_gate"), (1, "router")]:
        _phases(state, m, layer, flush)
    del flush
    return table


def _phases(state, m, layer, flush):
    """Print where a tuned config's time goes, from the kernel's per-CTA timestamps."""
    n = _n(layer, m)
    k = state["w"][layer].shape[1]
    nt = _nt(m)
    for v in VARIANTS:
        if v[0] != nt or not _valid(m, n, k, layer == "router", v):
            continue
        cfg = v + (_grids(m, n, k, v, state["sms"])[0],)
        x = torch.randn((m, LAYERS[layer][1]), device="cuda").to(torch.bfloat16)
        dbg = torch.zeros((cfg[-1], 4), dtype=torch.int64, device="cuda")
        rows = []
        for _ in range(5):
            flush.add_(1)
            _launch(state, x, layer, n, cfg, dbg)
            torch.cuda.synchronize()
            d = dbg[dbg[:, 0] > 0].double()
            t0 = d[:, 0].min()
            rows.append([float(d[:, 3].max() - t0) / 1e3, float((d[:, 0] - t0).max()) / 1e3,
                         float((d[:, 1] - d[:, 0]).median()) / 1e3, float((d[:, 2] - d[:, 1]).median()) / 1e3,
                         float((d[:, 2] - d[:, 1]).max()) / 1e3, float((d[:, 3] - d[:, 2]).median()) / 1e3,
                         float((d[:, 3] - d[:, 2]).max()) / 1e3])
        r = sorted(rows)[2]
        mb = (n * k * 2 + m * k * 2) / 1e6
        print(f"PHASES m={m} {layer} {cfg}: span {r[0]:.1f}us ({mb / r[0]:.2f} TB/s) | start skew {r[1]:.1f}"
              f" | first data {r[2]:.1f} | stream med {r[3]:.1f} max {r[4]:.1f} | flush med {r[5]:.1f}"
              f" max {r[6]:.1f}", flush=True)


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
        "maxc": {},
        "cnt": torch.zeros(N_COUNTERS, dtype=torch.int32, device=router.device),
    }
    state["w"]["router"] = torch.cat([hi, lo], dim=1).contiguous()
    state["wmap"] = {(name, rt, bks, b3): ext.make_wmap(w, 64 * rt, bks, bool(b3))
                     for name, w in state["w"].items()
                     for rt, bks, b3 in {(v[1], v[2], v[4]) for v in VARIANTS}}
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
