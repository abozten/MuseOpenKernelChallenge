# okc: problem=muse_bf16_proj
"""Dense projections: BF16 weights stored losslessly in 12 bits, a dedicated kernel for the router.

At up to 32 tokens the BF16 projections (qkv_gate, o, pooler, unpooler) are bound by reading
the weights, and cuBLAS already reads them near what an H200 streams. So the weights are read
in fewer bytes: prepare() stores each BF16 matrix, losslessly, in 12 bits per weight (W12
below), and the kernels decode it in registers straight into mma.sync. Above 32 tokens, and
wherever it is faster, the BF16 projections stay on cuBLAS.

W12. A BF16 value is [S E7..E1 | E0 M6..M0]. Its low byte is kept as is. The high byte (sign,
E >> 1) takes 4 bits: the sign and a 3-bit index idx = (E >> 1) - b into the window
[b, b + 7] (exponents 2b to 2b + 15) that holds the most weights of the matrix: for these
normal weights all but ~0.015 %. The kernel writes the high byte back as S << 7 | (b' | idx),
b' = b & ~7, a bitwise OR: that is the weight times 2^-2d (d = b & 7), and the FP32 result is
scaled by 2^2d, exactly. Weights outside the window (escapes, and any zero) are stored with
idx 0 and listed apart with their true value: the kernel subtracts what it decoded there and
adds the true weight, in FP32, so every product is the original weight's.

Layout (fragment order, as the router's): a 16-row x 64-k tile, lane (g, c) holding rows
g + 8h and k 32q + 8c + e (h, q in 0..1, e in 0..7), is 3 warp loads of 512 contiguous bytes
(1536 B, 2048 B in BF16), and the tiles are stored k-major ([k tile][row tile]):
- uint4 h (0, 1): the low bytes of row g + 8h, byte 8q + e;
- uint4 2: word 2h + q is the code word of (h, q). Its byte j holds, from bit 7 down: S of
  element j, idx of element 4 + (j + 1) % 4, S of element 4 + j, idx of element j. Then
  (word & 0x87878787) | b' is the high bytes of elements 0..3, and the same on the word
  rotated by 4 those of elements 4..7: LOP3, SHF, LOP3 for 8 weights, and one PRMT per pair
  builds the BF16x2 register mma.sync takes.
A dot product doesn't care about k's order, so x is read in the same permuted order.

The W12 kernels (prepare() picks one and its parameters per shape, or cuBLAS, by timing
CUDA-graph replays on a cold L2 as the judge times them):

- TMA kernel. A CTA computes R = 16 NW RT rows over one k slice. A producer warp fills an
  ST-deep ring of stages, each KS k tiles: the CTA's weight tiles by one TMA bulk copy per k
  tile (k-major tiles make them contiguous) and x's rows by cp.async, both completing on the
  stage's full mbarrier; it refills a stage once the NW computing warps have released it (its
  empty mbarrier). A computing warp takes RT row tiles that share its x fragments, with two
  chains of mma when there are few token tiles. Escapes are corrected as their k tile passes,
  their entries held one per lane and x read from the stage, into separate registers added
  once at the end. The S k slices of a row block are one cluster: each CTA pushes its FP32
  partial of the others' rows into their shared memory, and each writes R / S rows. One
  launch, no workspace.
- Register kernel. Each warp streams its 16-row tile's slice into registers U tiles ahead, x
  through L1; each k slice stores its FP32 partial in its own workspace slot, and the last CTA
  of a row block (by its counter) sums the slots in a fixed order, writes the BF16 rows and
  clears the counter.

The router is where SGLang leaves the most time: out = x @ W.T with W FP32 [256, 8192] runs as
an FP32 GEMV up to 8 tokens and as two BF16 GEMMs plus an add above that. Its kernel here:

- Runs the FP32 weights as a BF16 hi + lo pair with FP32 accumulation (allowed at any batch
  size, as SGLang does above 8 tokens), stored as one [256, 2 * 8192] matrix [hi | lo]: one
  GEMM over K = 16384, with x's column taken modulo 8192.
- Repacks that matrix (losslessly) into mma.sync fragment order, so every warp load is 512
  contiguous bytes.
- A CTA's 4 warps take 4 consecutive k slices of one 16-row tile (for 8 * NT tokens),
  streaming them into registers U tiles ahead; the CTA sums its warps in shared memory and
  adds the result into the FP32 output with red.global.add.
- The output is zeroed in the same launch by the first CTA to arrive. Every CTA counts its
  arrival on a counter that only grows (a call adds exactly one per CTA, so the call's epoch
  is the count over the grid size); the first publishes the epoch on a flag once the output is
  zero, and the others wait for it before adding (by then it has long been set). A call is
  one launch, CUDA-graph safe.

For more tokens prepare() times that kernel against router_big and cuBLAS paths:

- router_big: a shared-memory GEMM of BM tokens x BN outputs per CTA, x's tile loaded once per
  k step and multiplied by both halves; K split over the 8 or 16 CTAs of a cluster, which sum
  their partials through distributed shared memory in a fixed order (no atomics, no zeroing).
- stack: one cuBLAS GEMM against [hi; lo] stacked (N = 512, FP32 out), then the two halves
  added: no copy of x, as [x | x] @ [hi | lo].T would need.
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
#define ROUTER_N 256

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_acquire32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
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

// wp: the router's [hi | lo] weight repacked, [256 / 16][K / 64][4][32] uint4, K = 2 KX.
// x: [M, KX] BF16. out: [M, 256] FP32, zeroed here. Grid: 16 row tiles x KS k slices x
// ceil(M / (8 NT)) token tiles, KS = K / 64 / (WARPS KTW). sync: this config's two counters,
// [arrivals, zeroed epoch], which only grow.
template <int NT, int KTW, int U>
__global__ void __launch_bounds__(THREADS) router_gemm(
    const uint4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, float* __restrict__ out,
    uint32_t* __restrict__ sync, int M, int KX) {
  constexpr int TOKS = 8 * NT;
  __shared__ float red[WARPS - 1][32][NT * 4];
  __shared__ uint32_t s_epoch;
  __shared__ int s_first;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c = lane & 3;
  const int ktiles = (2 * KX) >> 6;
  constexpr int KS = 64 / KTW;
  const int rt = blockIdx.x & 15;
  const int b_div16 = blockIdx.x >> 4;
  const int ks = b_div16 & (KS - 1);
  const int zt = b_div16 / KS;
  const int kt0 = (ks * WARPS + warp) * KTW, tok0 = zt * TOKS;

  if (tid == 0) {
    const uint32_t v = atomicAdd(sync, 1u);
    s_epoch = v / gridDim.x + 1;
    s_first = (v % gridDim.x == 0);
  }
  __syncthreads();
  if (s_first) {
    for (int i = tid; i < M * ROUTER_N; i += THREADS) out[i] = 0.f;
    __threadfence();
    __syncthreads();
    if (tid == 0) st_release32(sync + 1, s_epoch);
  }

  auto load = [&](int j, uint4 (&wb)[4], uint4 (&xb)[NT][2]) {
    const int kt = kt0 + j;
    const uint4* p = wp + (rt * ktiles + kt) * 128 + lane;
#pragma unroll
    for (int l = 0; l < 4; ++l) wb[l] = ld_stream(p + l * 32);
    const int kx = (kt * 64) & (KX - 1);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int tok = min(tok0 + 8 * t + g, M - 1);
      const __nv_bfloat16* px = x + tok * KX + kx + 8 * c;
#pragma unroll
      for (int q = 0; q < 2; ++q)
        xb[t][q] = __ldg(reinterpret_cast<const uint4*>(px + 32 * q));
    }
  };

  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[t][e] = 0.f;

  uint4 wb[U][4], xb[U][NT][2];
#pragma unroll
  for (int j = 0; j < U; ++j) load(j, wb[j], xb[j]);
  for (int j0 = 0; j0 < KTW; j0 += U) {
#pragma unroll
    for (int j = 0; j < U; ++j) {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        // k step s takes words 2 (s & 1) and 2 (s & 1) + 1 of the lane's chunk q = s >> 1:
        // load 2h + q holds row 8h + g.
        const int q = s >> 1, w0 = (s & 1) * 2;
        const uint32_t a0 = word(wb[j][q], w0), a2 = word(wb[j][q], w0 + 1);
        const uint32_t a1 = word(wb[j][2 + q], w0), a3 = word(wb[j][2 + q], w0 + 1);
#pragma unroll
        for (int t = 0; t < NT; ++t)
          mma_bf16(acc[t], a0, a1, a2, a3, word(xb[j][t][q], w0), word(xb[j][t][q], w0 + 1));
      }
      if (j0 + j + U < KTW) load(j0 + j + U, wb[j], xb[j]);
    }
  }

  // Sum the CTA's warps (same rows and tokens, consecutive k), then add into out.
  if (warp > 0)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[warp - 1][lane][t * 4 + e] = acc[t][e];
  __syncthreads();
  if (warp > 0) return;
#pragma unroll
  for (int w = 0; w < WARPS - 1; ++w)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[t][e] += red[w][lane][t * 4 + e];
  if (lane == 0)
    while (ld_acquire32(sync + 1) < s_epoch) {}
  __syncwarp();
  // acc[t][e] is row rt 16 + g + 8 (e >> 1), token tok0 + 8 t + 2 c + (e & 1).
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int tok = tok0 + 8 * t + 2 * c + (e & 1);
      if (tok < M) atomicAdd(out + tok * ROUTER_N + rt * 16 + g + 8 * (e >> 1), acc[t][e]);
    }
}

__global__ void __launch_bounds__(256) router_sum_halves(const float4* __restrict__ in, float4* __restrict__ out, int M) {
  const int total = M * 64;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    const int row = idx >> 6;
    const int col = idx & 63;
    const float4 a = in[row * 128 + col];
    const float4 b = in[row * 128 + 64 + col];
    out[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
}
"""

W12_KERNEL_SRC = r"""
// 12-bit lossless BF16 weights (W12 in the docstring): a 16-row x 64-k tile is 3 warp loads of
// 512 contiguous bytes, stored k-major, [kt][rt][3][32 lanes] uint4: warps on the same k step
// of neighbouring row tiles read neighbouring memory.

struct W12Esc {
  int k, r;      // k, row within its 16-row tile
  float wt, wd;  // the true weight * 2^-2d and the value the kernel decoded there
};

struct W12Args {
  const uint4* wp;
  const __nv_bfloat16* x;
  __nv_bfloat16* out;
  float* ws;
  unsigned* cnt;
  const int* esc_off;
  const W12Esc* esc;
  int M, N, K, ktc, nrb;
  float scale;
  uint32_t bb;
};

__device__ __forceinline__ uint4 w12_ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 w12_lds(uint32_t a) {
  uint4 r;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
  return r;
}

__device__ __forceinline__ void w12_cp16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void w12_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void w12_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void w12_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t w12_word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// One 16 x 64 tile: decode the lane's 32 weights (wb) and multiply with x (xb[t][q]: token
// 8 t + g, k 32 q + 8 c to + 8, in the same permuted k order).
// acc takes the k steps over words 0, 1 of each chunk, acc2 those over words 2, 3: two chains of
// dependent mma instead of one (pass the same array twice for one).
template <int NT>
__device__ __forceinline__ void w12_tile(float (&acc)[NT][4], float (&acc2)[NT][4], const uint4 (&wb)[3],
                                         const uint4 (&xb)[NT][2], uint32_t bb) {
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    // code words (h, q): high bytes of elements 0..3, then (rotated) of 4..7
    const uint32_t n0 = q == 0 ? wb[2].x : wb[2].y;
    const uint32_t n1 = q == 0 ? wb[2].z : wb[2].w;
    const uint32_t hA0 = (n0 & 0x87878787u) | bb, hA1 = (n1 & 0x87878787u) | bb;
    const uint32_t hB0 = (__funnelshift_l(n0, n0, 4) & 0x87878787u) | bb;
    const uint32_t hB1 = (__funnelshift_l(n1, n1, 4) & 0x87878787u) | bb;
    const uint32_t lA0 = q == 0 ? wb[0].x : wb[0].z, lB0 = q == 0 ? wb[0].y : wb[0].w;
    const uint32_t lA1 = q == 0 ? wb[1].x : wb[1].z, lB1 = q == 0 ? wb[1].y : wb[1].w;
    // k step (q, 0): words 0, 1 of the chunk (elements 0..3); (q, 1): words 2, 3
    {
      const uint32_t a0 = __byte_perm(lA0, hA0, 0x5140), a2 = __byte_perm(lA0, hA0, 0x7362);
      const uint32_t a1 = __byte_perm(lA1, hA1, 0x5140), a3 = __byte_perm(lA1, hA1, 0x7362);
#pragma unroll
      for (int t = 0; t < NT; ++t) w12_mma(acc[t], a0, a1, a2, a3, xb[t][q].x, xb[t][q].y);
    }
    {
      const uint32_t a0 = __byte_perm(lB0, hB0, 0x5140), a2 = __byte_perm(lB0, hB0, 0x7362);
      const uint32_t a1 = __byte_perm(lB1, hB1, 0x5140), a3 = __byte_perm(lB1, hB1, 0x7362);
#pragma unroll
      for (int t = 0; t < NT; ++t) w12_mma(acc2[t], a0, a1, a2, a3, xb[t][q].z, xb[t][q].w);
    }
  }
}

// The escapes of row tile rt over k tiles kt0 .. kt0 + ktc: subtract what was decoded, add the
// true weight (exact for a probe row: the first FMA cancels its product). Then store: with one
// k slice the BF16 output directly, else k slice s's own FP32 partial, ws[s][M][N] (plain
// stores, no atomics). acc[t][e] is row rt 16 + g + 8 (e >> 1), token 8 t + 2 c + (e & 1).
template <int NT>
__device__ __forceinline__ void w12_finish(float (&acc)[NT][4], const W12Args& a, int rt, int kt0, int S,
                                           int s, int g, int c) {
  const int KT = a.K / 64;
  const int e0 = a.esc_off[rt * KT + kt0], e1 = a.esc_off[rt * KT + kt0 + a.ktc];
  for (int i = e0; i < e1; ++i) {
    const W12Esc e = a.esc[i];
    if ((e.r & 7) != g) continue;
    const int h = e.r >> 3;
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int tok = 8 * t + 2 * c + u;
        if (tok < a.M) {
          const float xv = __bfloat162float(a.x[tok * a.K + e.k]);
          acc[t][2 * h + u] = fmaf(xv, e.wt, fmaf(-xv, e.wd, acc[t][2 * h + u]));
        }
      }
  }
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int tok = 8 * t + 2 * c + (e & 1);
      if (tok >= a.M) continue;
      const int o = tok * a.N + rt * 16 + g + 8 * (e >> 1);
      if (S == 1)
        a.out[o] = __float2bfloat16(acc[t][e] * a.scale);
      else
        __stcg(a.ws + s * (a.M * a.N) + o, acc[t][e]);
    }
}

// With S > 1: the last CTA of row block rb (R rows) to arrive sums the S partials of its rows in
// a fixed order (so the result is deterministic), writes them as BF16 and clears the counter for
// the next call. Every partial is rewritten each call, so ws needs no clearing.
__device__ __forceinline__ void w12_reduce(const W12Args& a, int rb, int R, int S) {
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomicAdd(a.cnt + rb, 1u) == (unsigned)(S - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  const int r0 = rb * R;
  const int step = a.M * a.N;
  for (int i = threadIdx.x; i < a.M * R; i += blockDim.x) {
    const int tok = i / R, row = r0 + i % R;
    const float* p = a.ws + tok * a.N + row;
    float v = 0.f;
#pragma unroll 8
    for (int j = 0; j < S; ++j) v += __ldcg(p + j * step);
    a.out[tok * a.N + row] = __float2bfloat16(v * a.scale);
  }
  if (threadIdx.x == 0) a.cnt[rb] = 0;
}

// Contiguous kernel: one CTA per RT 16-row tiles over all of K, weights row-tile-major
// ([rt][kt][3][32] uint4), so a CTA reads RT contiguous ranges. Its NWK warps split K (warp w
// takes k tiles w, w + NWK, ...: on each step the CTA's warps read one contiguous chunk per row
// tile), stream them into registers U ahead (one x fragment serving the RT tiles) and sum in
// shared memory in a fixed order: no k slices across CTAs, no workspace. The tiles' escapes are
// copied to shared memory while the first tiles load. NWR groups of NWK warps take consecutive
// RT-tile groups: x then comes from L2 once per NWR RT tiles and the other groups hit L1 (x's L2
// traffic, one x per CTA, is what slows m >= 16).
template <int NT, int NWK, int U, int RT, int NWR>
__global__ void __launch_bounds__(NWR * NWK * 32) w12c_gemm(const W12Args a) {
  constexpr int RTC = NWR * RT, NTH = NWR * NWK * 32, ESC_CAP = 128 * RTC;
  __shared__ float red[NWR][NWK][RT][NT][4][32];
  __shared__ W12Esc s_esc[ESC_CAP];
  const int tid = threadIdx.x, wr = (tid >> 5) / NWK, warp = (tid >> 5) % NWK, lane = tid & 31;
  const int g = lane >> 2, c = lane & 3;
  const int KT = a.K / 64, rt0 = blockIdx.x * RTC, J = KT / NWK;
  // each CTA starts its k steps at its own offset, so the CTAs don't all read the same x lines
  // (one L2 slice) at once
  const int rot = blockIdx.x & (J - 1);
  const uint4* wbase = a.wp + ((rt0 + wr * RT) * KT + warp) * 96 + lane;

  int toff[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tok = min(8 * t + g, a.M - 1);
    toff[t] = tok * a.K;
  }
  auto load = [&](int j, uint4 (&wb)[RT][3], uint4 (&xb)[NT][2]) {
    j = j + rot >= J ? j + rot - J : j + rot;
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint4* p = wbase + (r * KT + j * NWK) * 96;
#pragma unroll
      for (int l = 0; l < 3; ++l) wb[r][l] = w12_ld_stream(p + 32 * l);
    }
    const int kx = (warp + j * NWK) * 64 + 8 * c;
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int q = 0; q < 2; ++q)
        xb[t][q] = __ldg(reinterpret_cast<const uint4*>(a.x + toff[t] + kx + 32 * q));
  };

  float acc[RT][NT][4], acc2[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][t][e] = acc2[r][t][e] = 0.f;
  uint4 wb[U][RT][3], xb[U][NT][2];
#pragma unroll
  for (int j = 0; j < U; ++j) load(j, wb[j], xb[j]);
  // after the first loads are in flight
  const int e0 = a.esc_off[rt0 * KT], ne = a.esc_off[(rt0 + RTC) * KT] - e0;
  for (int i = tid; i < min(ne, ESC_CAP); i += NTH) s_esc[i] = a.esc[e0 + i];
  for (int j0 = 0; j0 < J; j0 += U) {
#pragma unroll
    for (int j = 0; j < U; ++j) {
#pragma unroll
      for (int r = 0; r < RT; ++r) w12_tile<NT>(acc[r], acc2[r], wb[j][r], xb[j], a.bb);
      if (j0 + j + U < J) load(j0 + j + U, wb[j], xb[j]);
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[wr][warp][r][t][e][lane] = acc[r][t][e] + acc2[r][t][e];
  __syncthreads();
  // thread i: row i % (16 RTC) of the CTA (tile r, row rr), token i / (16 RTC); its fragment slot:
  // lane 4 (rr & 7) + (tok & 7) / 2, element 2 (rr >> 3) + (tok & 1), token tile tok >> 3
  for (int i = tid; i < 16 * RTC * 8 * NT; i += NTH) {
    const int row = i % (16 * RTC), tok = i / (16 * RTC), r = row >> 4, rr = row & 15;
    if (tok >= a.M) continue;
    const int ln = 4 * (rr & 7) + ((tok & 7) >> 1), e = 2 * (rr >> 3) + (tok & 1), t = tok >> 3;
    float v = 0.f;
    const int wr_idx = (RT == 1) ? r : (r >> 1);
    const int rt_idx = (RT == 1) ? 0 : (r & 1);
#pragma unroll
    for (int w = 0; w < NWK; ++w) v += red[wr_idx][w][rt_idx][t][e][ln];
    const int k1 = a.esc_off[(rt0 + r) * KT] - e0, k2e = a.esc_off[(rt0 + r + 1) * KT] - e0;
    for (int k2 = k1; k2 < k2e; ++k2) {
      const W12Esc es = k2 < ESC_CAP ? s_esc[k2] : a.esc[e0 + k2];
      if (es.r != rr) continue;
      const float xv = __bfloat162float(a.x[tok * a.K + es.k]);
      v = fmaf(xv, es.wt, fmaf(-xv, es.wd, v));
    }
    a.out[tok * a.N + (rt0 + r) * 16 + rr] = __float2bfloat16(v * a.scale);
  }
}

// Register ring. Grid: nrb row blocks (NW row tiles) x S k slices of ktc tiles; warp w takes
// row tile rb NW + w and streams its tiles into registers U ahead, x through L1.
template <int NT, int U, int NW>
__global__ void __launch_bounds__(NW * 32) w12r_gemm(const W12Args a) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c = lane & 3;
  const int KT = a.K / 64, S = gridDim.x / a.nrb, ktc = a.ktc;
  const int rb = blockIdx.x % a.nrb, s = blockIdx.x / a.nrb;
  const int rt = rb * NW + warp, kt0 = s * ktc;
  const int NRT = a.N / 16;
  const uint4* wbase = a.wp + (kt0 * NRT + rt) * 96 + lane;

  int toff[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tok = min(8 * t + g, a.M - 1);
    toff[t] = tok * a.K;
  }
  auto load = [&](int j, uint4 (&wb)[3], uint4 (&xb)[NT][2]) {
    const uint4* p = wbase + j * NRT * 96;
#pragma unroll
    for (int l = 0; l < 3; ++l) wb[l] = w12_ld_stream(p + 32 * l);
    const int kx = (kt0 + j) * 64 + 8 * c;
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int q = 0; q < 2; ++q)
        xb[t][q] = __ldg(reinterpret_cast<const uint4*>(a.x + toff[t] + kx + 32 * q));
  };

  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[t][e] = 0.f;
  uint4 wb[U][3], xb[U][NT][2];
#pragma unroll
  for (int j = 0; j < U; ++j) load(j, wb[j], xb[j]);
  for (int j0 = 0; j0 < ktc; j0 += U) {
#pragma unroll
    for (int j = 0; j < U; ++j) {
      w12_tile<NT>(acc, acc, wb[j], xb[j], a.bb);
      if (j0 + j + U < ktc) load(j0 + j + U, wb[j], xb[j]);
    }
  }
  w12_finish<NT>(acc, a, rt, kt0, S, s, g, c);
  if (S > 1) w12_reduce(a, rb, 16 * NW, S);
}

__device__ __forceinline__ uint32_t w12_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void w12_cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void w12_st_peer(uint32_t addr, uint32_t rank, float v) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(addr), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}

__device__ __forceinline__ void w12_mbar_init(uint32_t bar, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(n) : "memory");
}
__device__ __forceinline__ void w12_mbar_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void w12_mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void w12_mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nW12W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W12W_%=;\n}\n"
      ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void w12_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// TMA kernel. A stage is KS k tiles: the CTA's R = 16 NW RT rows of weights (k-major, so one
// bulk copy per k tile) and x's rows for those k, [8 NT][KS 64 + 32] BF16 (rows padded by 64 B so
// a quarter-warp's 16 B reads hit distinct banks; rows past M stay zero).
__host__ __device__ constexpr int w12t_xs(int KS) { return KS * 64 + 32; }
__host__ __device__ inline int w12t_stage_bytes(int NT, int NW, int RT, int KS) {
  return 8 * NT * w12t_xs(KS) * 2 + KS * NW * RT * 1536;
}
__host__ __device__ inline int w12t_smem_bytes(int NT, int ST, int NW, int RT, int KS) {
  const int a = ST * w12t_stage_bytes(NT, NW, RT, KS), b = 8 * NT * 16 * NW * RT * 4;
  return a > b ? a : b;
}

// Grid: nrb row blocks (R rows) x S k slices of ktc tiles, the S slices of a row block one
// cluster (blockIdx = rb S + rank). Warps 0 .. NW - 1 compute, warp w taking RT row tiles that
// share its x fragments. Warp NW produces: it fills an ST-deep ring of stages (weights by TMA bulk
// copies, x by cp.async), each stage on a full mbarrier, and refills a stage once the computing
// warps have released it (its empty mbarrier). Escapes are corrected as their k tile passes
// (their entries held one per lane, x from the stage), into cor, added once at the end. With
// S > 1 the partials are summed through distributed shared memory, pushed to the CTA of the
// cluster that writes their R / S rows.
template <int NT, int ST, int NW, int RT, int KS>
__global__ void __launch_bounds__((NW + 1) * 32) w12t_gemm(const W12Args a) {
  extern __shared__ __align__(128) uint8_t w12_smem[];
  __shared__ __align__(8) uint64_t bars[2 * ST];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c = lane & 3;
  const int KT = a.K / 64, ktc = a.ktc, NRT = a.N / 16, nst = ktc / KS;
  const int S = KT / ktc, s = blockIdx.x % S, rb = blockIdx.x / S;
  constexpr int R = 16 * NW * RT, XS = w12t_xs(KS), XB = 8 * NT * XS * 2, SB = XB + KS * NW * RT * 1536;
  const int kt0 = s * ktc;
  const uint32_t sm = (uint32_t)__cvta_generic_to_shared(w12_smem);
  const uint32_t full = (uint32_t)__cvta_generic_to_shared(bars), empty = full + 8 * ST;

  // zero x's rows past M in every stage (TMA writes only rows < M), init the barriers
  for (int i = tid; i < ST * (8 * NT - a.M) * XS / 2; i += blockDim.x) {
    const int per = (8 * NT - a.M) * XS / 2, st = i / per, j = i % per;
    reinterpret_cast<uint32_t*>(w12_smem + st * SB)[a.M * XS / 2 + j] = 0u;
  }
  if (tid == 0) {
    for (int i = 0; i < ST; ++i) {
      w12_mbar_init(full + 8 * i, 1 + 32);
      w12_mbar_init(empty + 8 * i, NW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  // acc2: the second chain of mma (NT <= 2; with more token tiles there are chains enough)
  constexpr int DUAL = NT <= 2;
  float acc[RT][NT][4], acc2[RT][DUAL ? NT : 1][4], cor[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][t][e] = cor[r][t][e] = 0.f;
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < (DUAL ? NT : 1); ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc2[r][t][e] = 0.f;

  if (warp == NW) {
    // lane 0 copies the weights (TMA bulk, counted in bytes); the warp's lanes copy x's rows in
    // 16 B pieces (cp.async, each lane's completion one arrival: the full barrier counts 1 + 32)
    const int chunks = KS * 8;
    constexpr int shift = 3 + (KS == 1 ? 0 : KS == 2 ? 1 : KS == 4 ? 2 : 3);
    for (int i = 0; i < nst; ++i) {
      const int st = i % ST;
      if (i >= ST) w12_mbar_wait(empty + 8 * st, ((i / ST) - 1) & 1);
      const uint32_t bar = full + 8 * st, dst = sm + st * SB;
      const int k0 = kt0 + i * KS;
      if (lane == 0) {
        w12_mbar_tx(bar, KS * NW * RT * 1536);
        for (int kk = 0; kk < KS; ++kk)
          w12_bulk(dst + XB + kk * NW * RT * 1536, a.wp + ((k0 + kk) * NRT + rb * NW * RT) * 96,
                   NW * RT * 1536, bar);
      }
      for (int j = lane; j < a.M * chunks; j += 32) {
        const int r = j >> shift, ch = j & (chunks - 1);
        w12_cp16(dst + (r * XS + ch * 8) * 2, a.x + r * a.K + k0 * 64 + ch * 8, 16);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
    }
    __syncwarp();
  } else {
    // escapes: entries base .. base + 31 held one per lane, p the next one
    int p[RT], e1[RT], base[RT], nkt[RT];
    W12Esc mine[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const int rt = rb * NW * RT + warp * RT + r;
      p[r] = base[r] = a.esc_off[rt * KT + kt0];
      e1[r] = a.esc_off[rt * KT + kt0 + ktc];
      if (base[r] + lane < e1[r]) mine[r] = a.esc[base[r] + lane];
      nkt[r] = p[r] < e1[r] ? __shfl_sync(0xffffffffu, mine[r].k, 0) >> 6 : 1 << 30;
    }
    for (int i = 0; i < nst; ++i) {
      const int st = i % ST;
      w12_mbar_wait(full + 8 * st, (i / ST) & 1);
      const uint32_t xs = sm + st * SB, ws = xs + XB;
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
        uint4 xb[NT][2];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int q = 0; q < 2; ++q) xb[t][q] = w12_lds(xs + ((8 * t + g) * XS + kk * 64 + 32 * q + 8 * c) * 2);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          uint4 wb[3];
#pragma unroll
          for (int l = 0; l < 3; ++l)
            wb[l] = w12_lds(ws + ((kk * NW * RT + warp * RT + r) * 96 + 32 * l + lane) * 16);
          if constexpr (DUAL)
            w12_tile<NT>(acc[r], acc2[r], wb, xb, a.bb);
          else
            w12_tile<NT>(acc[r], acc[r], wb, xb, a.bb);
        }
      }
      // escapes in this stage's k tiles
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        while (nkt[r] < kt0 + (i + 1) * KS) {
          const int src = p[r] - base[r];
          const int ek = __shfl_sync(0xffffffffu, mine[r].k, src), er = __shfl_sync(0xffffffffu, mine[r].r, src);
          const float wt = __shfl_sync(0xffffffffu, mine[r].wt, src), wd = __shfl_sync(0xffffffffu, mine[r].wd, src);
          if ((er & 7) == g) {
            const int h = er >> 3;
            const __nv_bfloat16* xr =
                reinterpret_cast<const __nv_bfloat16*>(w12_smem + st * SB) + (ek - (kt0 + i * KS) * 64);
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
              for (int u = 0; u < 2; ++u) {
                const float xv = __bfloat162float(xr[(8 * t + 2 * c + u) * XS]);
                cor[r][t][2 * h + u] = fmaf(xv, wt, fmaf(-xv, wd, cor[r][t][2 * h + u]));
              }
          }
          if (++p[r] - base[r] == 32) {  // the next 32 (rare)
            base[r] = p[r];
            if (base[r] + lane < e1[r]) mine[r] = a.esc[base[r] + lane];
          }
          nkt[r] = p[r] < e1[r] ? __shfl_sync(0xffffffffu, mine[r].k, p[r] - base[r]) >> 6 : 1 << 30;
        }
      }
      // The producer refills this stage with TMA (the async proxy): order this warp's reads of
      // it (generic proxy) before the release.
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) w12_mbar_arrive(empty + 8 * st);
    }
  }

  if (DUAL)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[r][t][e] += acc2[r][DUAL ? t : 0][e];
  if (S == 1) {
    if (warp < NW) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int tok = 8 * t + 2 * c + (e & 1);
            if (tok < a.M)
              a.out[tok * a.N + rb * R + (warp * RT + r) * 16 + g + 8 * (e >> 1)] =
                  __float2bfloat16((acc[r][t][e] + cor[r][t][e]) * a.scale);
          }
    }
    return;
  }
  // The cluster's sum, rows split by rank: once every CTA is done with its ring, each pushes its
  // partial of rank q's rows into q's shared memory, recv[source][8 NT][R / S], then q adds them.
  const int rows = R / S;
  w12_cluster_sync();
  if (warp < NW) {
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int tok = 8 * t + 2 * c + (e & 1), row = (warp * RT + r) * 16 + g + 8 * (e >> 1);
          if (tok < a.M)
            w12_st_peer(sm + ((s * 8 * NT + tok) * rows + row % rows) * 4, row / rows, acc[r][t][e] + cor[r][t][e]);
        }
  }
  w12_cluster_sync();
  const float* recv = reinterpret_cast<const float*>(w12_smem);
  for (int i = tid; i < a.M * rows; i += blockDim.x) {
    const int tok = i / rows, row = i % rows;
    float v = 0.f;
    for (int q = 0; q < S; ++q) v += recv[(q * 8 * NT + tok) * rows + row];
    a.out[tok * a.N + rb * R + s * rows + row] = __float2bfloat16(v * a.scale);
  }
}
"""

ROUTER_BIG_SRC = r"""
// The router for more tokens: out [M, 256] = x @ hi.T + x @ lo.T with hi, lo the FP32 weight's
// BF16 halves ([256, 8192] row-major), FP32 accumulation. A CTA computes a BM-token x BN-output
// tile over K / CS: the K = 8192 is split over the CS CTAs of a cluster, which sum their
// partials through DSMEM in a fixed order (no atomics, no zeroing, deterministic). x's tile is
// loaded once per k step and multiplied by both halves. Stages of 64 k: rows of 128 B with their
// 16 B chunks XOR-swizzled by row, filled by cp.async in an ST-deep ring, read by ldmatrix into
// mma.sync. WGM x WGN warps, each a (BM / WGM) x (BN / WGN) piece.
__device__ __forceinline__ void rb_ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ float4 rb_ld_peer4(uint32_t addr, uint32_t rank) {
  uint32_t a;
  float4 v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(addr), "r"(rank));
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}

template <int BM, int BN, int CS, int ST, int WGM, int WGN>
__host__ __device__ constexpr int rb_smem_bytes() {
  return ST * (BM + 2 * BN) * 128 > BM * (BN + 8) * 4 ? ST * (BM + 2 * BN) * 128 : BM * (BN + 8) * 4;
}

template <int BM, int BN, int CS, int ST, int WGM, int WGN>
__global__ void __launch_bounds__(32 * WGM * WGN) router_big(const __nv_bfloat16* __restrict__ hi,
                                                  const __nv_bfloat16* __restrict__ lo,
                                                  const __nv_bfloat16* __restrict__ x,
                                                  float* __restrict__ out, int M) {
  constexpr int NTH = 32 * WGM * WGN, WM = BM / WGM, WN = BN / WGN, MI = WM / 16, NI = WN / 8;
  constexpr int A_B = BM * 128, B_B = BN * 128, STAGE = A_B + 2 * B_B;
  constexpr int KC = 8192 / CS, NK = KC / 64, LD = BN + 8;
  static_assert(NI % 2 == 0 && MI >= 1 && BM % CS == 0 && NK >= ST && (BM * 8) % NTH == 0 && (BN * 8) % NTH == 0,
                "router_big tile");
  extern __shared__ __align__(128) uint8_t rb_smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WGN, wn = warp % WGN, g = lane >> 2, c = lane & 3;
  const uint32_t rank = w12_rank();
  const int tile = blockIdx.x / CS, mt = tile / (256 / BN), nt = tile % (256 / BN);
  const int m0 = mt * BM, n0 = nt * BN, k0 = rank * KC;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(rb_smem);

  auto issue = [&](int i) {
    if (i < NK) {
      const uint32_t st = sbase + (i % ST) * STAGE;
      const int k = k0 + i * 64;
#pragma unroll
      for (int j = 0; j < BM * 8 / NTH; ++j) {
        const int q = tid + NTH * j, r = q >> 3, ch = q & 7, tok = m0 + r;
        w12_cp16(st + r * 128 + ((ch ^ (r & 7)) << 4), x + (tok < M ? tok : 0) * 8192 + k + ch * 8,
                 tok < M ? 16 : 0);
      }
#pragma unroll
      for (int j = 0; j < BN * 8 / NTH; ++j) {
        const int q = tid + NTH * j, r = q >> 3, ch = q & 7;
        const int o = (n0 + r) * 8192 + k + ch * 8;
        const uint32_t d = st + A_B + r * 128 + ((ch ^ (r & 7)) << 4);
        w12_cp16(d, hi + o, 16);
        w12_cp16(d + B_B, lo + o, 16);
      }
    }
    w12_commit();
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
  for (int i = 0; i < ST - 1; ++i) issue(i);
  for (int i = 0; i < NK; ++i) {
    w12_wait<ST - 2>();
    __syncthreads();
    issue(i + ST - 1);  // into the stage everyone finished at i - 1
    const uint32_t st = sbase + (i % ST) * STAGE;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t a[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r = wm * WM + mi * 16 + (lane & 15), ch = 2 * kk + (lane >> 4);
        rb_ldsm4(a[mi], st + r * 128 + ((ch ^ (r & 7)) << 4));
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t b[NI][2];
#pragma unroll
        for (int nj = 0; nj < NI; nj += 2) {
          const int r = wn * WN + nj * 8 + (lane & 7) + ((lane >> 4) << 3), ch = 2 * kk + ((lane >> 3) & 1);
          uint32_t t[4];
          rb_ldsm4(t, st + A_B + h * B_B + r * 128 + ((ch ^ (r & 7)) << 4));
          b[nj][0] = t[0];
          b[nj][1] = t[1];
          b[nj + 1][0] = t[2];
          b[nj + 1][1] = t[3];
        }
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
          for (int ni = 0; ni < NI; ++ni)
            mma_bf16(acc[mi][ni], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b[ni][0], b[ni][1]);
      }
    }
  }

  // The partial into this CTA's shared memory, [BM][LD] FP32; then rank r sums rows
  // r BM / CS .. of all CS partials, peer 0 first.
  w12_wait<0>();
  __syncthreads();
  float* red = reinterpret_cast<float*>(rb_smem);
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = wm * WM + mi * 16 + g + 8 * hh, col = wn * WN + ni * 8 + 2 * c;
        *reinterpret_cast<float2*>(red + r * LD + col) = make_float2(acc[mi][ni][2 * hh], acc[mi][ni][2 * hh + 1]);
      }
  w12_cluster_sync();
  constexpr int RR = BM / CS, ITEMS = RR * BN / 4;
  for (int it = tid; it < ITEMS; it += NTH) {
    const int r = rank * RR + it / (BN / 4), c4 = (it % (BN / 4)) * 4;
    const uint32_t off = sbase + (r * LD + c4) * 4;
    float4 s = rb_ld_peer4(off, 0);
#pragma unroll
    for (int p = 1; p < CS; ++p) {
      const float4 v = rb_ld_peer4(off, p);
      s.x += v.x;
      s.y += v.y;
      s.z += v.z;
      s.w += v.w;
    }
    const int tok = m0 + r;
    if (tok < M) *reinterpret_cast<float4*>(out + tok * 256 + n0 + c4) = s;
  }
  w12_cluster_sync();  // peers may still be reading this CTA's partial
}
"""

LAUNCH_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// (NT, KTW, U); keep in sync with ROUTER_VARIANTS in Python.
#define ROUTER_VARIANTS(X)                                                                 \
  X(1, 1, 1) X(1, 2, 2) X(1, 4, 4) X(2, 1, 1) X(2, 2, 2) X(2, 4, 4) X(4, 1, 1) X(4, 2, 2) X(4, 4, 2) \
  X(8, 2, 2) X(8, 4, 2)

void router(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor sync,
            int64_t slot, int64_t NT, int64_t KTW, int64_t U) {
  const int M = x.size(0), KX = x.size(1);
  const int KS = 2 * KX / 64 / (4 * KTW);
  const int G = 16 * KS * ((M + 8 * NT - 1) / (8 * NT));
  auto* s = reinterpret_cast<uint32_t*>(sync.data_ptr<int32_t>()) + 2 * slot;
  auto stream = c10::cuda::getCurrentCUDAStream();
#define CASE(nt, ktw, u)                                                                    \
  if (NT == nt && KTW == ktw && U == u) {                                                   \
    router_gemm<nt, ktw, u><<<G, THREADS, 0, stream>>>(                                     \
        reinterpret_cast<const uint4*>(wp.data_ptr()),                                      \
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), s, M, KX); \
    return;                                                                                 \
  }
  ROUTER_VARIANTS(CASE)
#undef CASE
  TORCH_CHECK(false, "no router_gemm variant NT=", NT, " KTW=", KTW, " U=", U);
}

void router_sum(torch::Tensor in, torch::Tensor out) {
  const int M = in.size(0);
  const int total = M * 64;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  auto stream = c10::cuda::getCurrentCUDAStream();
  router_sum_halves<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float4*>(in.data_ptr()),
      reinterpret_cast<float4*>(out.data_ptr()), M);
}
"""

W12_LAUNCH_SRC = r"""
// Variants; keep in sync with W12R / W12T in Python.
// Register ring: (NT, U, NW).
#define W12R_VARIANTS(X) X(1, 2, 8) X(1, 2, 4) X(2, 2, 8) X(2, 2, 4) X(4, 2, 8) X(4, 2, 4) \
  X(1, 4, 8) X(1, 4, 4) X(1, 8, 8) X(1, 8, 4) X(2, 4, 8) X(2, 4, 4)
// Contiguous kernel: (NT, NWK, U, RT, NWR).
#define W12C_VARIANTS(X) X(1, 4, 4, 1, 1) X(1, 8, 2, 1, 1) X(1, 8, 4, 1, 1) X(1, 16, 2, 1, 1) \
  X(1, 16, 4, 1, 1) X(1, 4, 2, 1, 1) X(1, 2, 4, 1, 1) X(1, 8, 2, 2, 1) X(1, 16, 2, 2, 1) \
  X(2, 8, 2, 1, 1) X(2, 8, 4, 1, 1) X(2, 16, 2, 1, 1) X(2, 4, 2, 1, 1) X(2, 8, 2, 2, 1) \
  X(2, 16, 2, 2, 1) X(2, 8, 1, 2, 1) X(4, 8, 2, 1, 1) X(4, 4, 2, 1, 1) X(4, 8, 1, 2, 1) \
  X(4, 8, 2, 2, 1) X(2, 4, 2, 1, 4) X(2, 8, 2, 1, 4) X(2, 2, 2, 1, 8) X(2, 4, 2, 2, 2) \
  X(2, 8, 2, 1, 2) X(2, 4, 2, 1, 2) X(4, 4, 2, 1, 4) X(4, 2, 2, 1, 8) X(4, 4, 2, 2, 2) \
  X(4, 8, 2, 1, 2) X(4, 4, 2, 1, 2) X(4, 2, 2, 2, 4) X(2, 4, 4, 1, 1) X(4, 4, 4, 1, 1)
// TMA kernel: (NT, ST, NW, RT, KS).
#define W12T_VARIANTS(X) \
  X(1, 8, 4, 1, 1) X(1, 6, 4, 1, 2) X(1, 16, 4, 1, 1) X(1, 24, 4, 1, 1) X(1, 16, 2, 1, 1) \
  X(1, 12, 2, 2, 1) X(1, 8, 4, 2, 1) X(2, 8, 4, 1, 1) X(2, 6, 4, 1, 2) X(2, 16, 4, 1, 1) \
  X(2, 16, 2, 1, 1) X(2, 8, 8, 1, 1) X(2, 8, 4, 2, 1) X(4, 8, 4, 1, 1) X(4, 6, 4, 1, 2) \
  X(4, 12, 4, 1, 1) X(4, 8, 8, 1, 1) X(4, 6, 8, 1, 1) X(4, 8, 4, 2, 1) X(4, 4, 8, 2, 1) \
  X(1, 3, 4, 1, 4) X(1, 4, 4, 1, 4) X(1, 6, 4, 1, 4) X(1, 3, 4, 1, 8) X(2, 4, 4, 1, 4) \
  X(2, 6, 4, 1, 4) X(4, 4, 4, 1, 4)

static int w12_max_smem = 0;  // set by w12_init

// kind 0: register ring (A = U, B = NW); kind 1: TMA kernel (A = ST, B = NW, C = RT, D = KS,
// S <= 8 the cluster size).
void w12(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
         torch::Tensor esc_off, torch::Tensor esc, int64_t N, int64_t S, double scale, int64_t bb,
         int64_t kind, int64_t NT, int64_t A, int64_t B, int64_t C, int64_t D) {
  W12Args a;
  a.wp = reinterpret_cast<const uint4*>(wp.data_ptr());
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.ws = ws.data_ptr<float>();
  a.cnt = reinterpret_cast<unsigned*>(cnt.data_ptr<int>());
  a.esc_off = esc_off.data_ptr<int>();
  a.esc = reinterpret_cast<const W12Esc*>(esc.data_ptr());
  a.M = x.size(0);
  a.N = N;
  a.K = x.size(1);
  a.ktc = a.K / 64 / S;
  a.scale = (float)scale;
  a.bb = (uint32_t)bb;
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (kind == 0) {
    TORCH_CHECK(S == 1 || S * a.M * N <= ws.numel(), "w12 workspace too small");
#define CASE(nt, u, w)                                                \
  if (NT == nt && A == u && B == w) {                                 \
    a.nrb = N / 16 / w;                                               \
    w12r_gemm<nt, u, w><<<a.nrb * S, w * 32, 0, stream>>>(a);         \
    return;                                                           \
  }
    W12R_VARIANTS(CASE)
#undef CASE
  } else if (kind == 1) {
#define CASE(nt, st, w, rt, ks)                                                              \
  if (NT == nt && A == st && B == w && C == rt && D == ks) {                                 \
    const int smem = w12t_smem_bytes(nt, st, w, rt, ks);                                     \
    TORCH_CHECK(smem <= w12_max_smem && S <= 8 && a.ktc % ks == 0, "w12t smem ", smem, " S ", S); \
    cudaLaunchConfig_t cfg = {};                                                             \
    cudaLaunchAttribute attr[1];                                                             \
    attr[0].id = cudaLaunchAttributeClusterDimension;                                        \
    attr[0].val.clusterDim.x = S;                                                            \
    attr[0].val.clusterDim.y = 1;                                                            \
    attr[0].val.clusterDim.z = 1;                                                            \
    cfg.gridDim = dim3(N / (16 * w * rt) * S);                                               \
    cfg.blockDim = dim3((w + 1) * 32);                                                       \
    cfg.dynamicSmemBytes = smem;                                                             \
    cfg.stream = stream;                                                                     \
    cfg.attrs = attr;                                                                        \
    cfg.numAttrs = 1;                                                                        \
    const cudaError_t e = cudaLaunchKernelEx(&cfg, w12t_gemm<nt, st, w, rt, ks>, a);         \
    TORCH_CHECK(e == cudaSuccess, "w12t launch: ", cudaGetErrorString(e));                   \
    return;                                                                                  \
  }
    W12T_VARIANTS(CASE)
#undef CASE
  } else if (kind == 2) {
#define CASE(nt, nwk, u, rt, nwr)                                                       \
  if (NT == nt && B == nwk && A == u && C == rt && D == nwr) {                          \
    w12c_gemm<nt, nwk, u, rt, nwr><<<N / 16 / rt / nwr, nwr * nwk * 32, 0, stream>>>(a); \
    return;                                                                             \
  }
    W12C_VARIANTS(CASE)
#undef CASE
  }
  TORCH_CHECK(false, "no w12 variant kind=", kind, " NT=", NT, " ", A, " ", B, " ", C, " ", D);
}

// (BM, BN, CS, ST, WGM, WGN); keep in sync with ROUTER_BIG in Python.
#define ROUTER_BIG_VARIANTS(X)                                                                 \
  X(32, 32, 16, 4, 2, 2) X(64, 32, 8, 4, 2, 2) X(64, 32, 16, 4, 2, 2) X(64, 64, 8, 4, 2, 2)      \
  X(64, 64, 16, 4, 2, 2) X(128, 32, 8, 4, 2, 2) X(64, 64, 16, 4, 2, 4) X(64, 64, 8, 4, 2, 4)     \
  X(128, 64, 16, 3, 4, 2) X(128, 64, 8, 3, 4, 2) X(128, 32, 16, 4, 4, 2) X(64, 32, 16, 4, 4, 1)

void router_big_run(torch::Tensor hi, torch::Tensor lo, torch::Tensor x, torch::Tensor out, int64_t BM,
                    int64_t BN, int64_t CS, int64_t ST, int64_t WGM, int64_t WGN) {
  const int M = x.size(0);
#define CASE(bm, bn, cs, st, wgm, wgn)                                                       \
  if (BM == bm && BN == bn && CS == cs && ST == st && WGM == wgm && WGN == wgn) {            \
    const int smem = rb_smem_bytes<bm, bn, cs, st, wgm, wgn>();                              \
    TORCH_CHECK(smem <= w12_max_smem, "router_big smem ", smem);                             \
    cudaLaunchConfig_t cfg = {};                                                             \
    cudaLaunchAttribute attr[1];                                                             \
    attr[0].id = cudaLaunchAttributeClusterDimension;                                        \
    attr[0].val.clusterDim.x = cs;                                                           \
    attr[0].val.clusterDim.y = 1;                                                            \
    attr[0].val.clusterDim.z = 1;                                                            \
    cfg.gridDim = dim3((M + bm - 1) / bm * (256 / bn) * cs);                                 \
    cfg.blockDim = dim3(32 * wgm * wgn);                                                     \
    cfg.dynamicSmemBytes = smem;                                                             \
    cfg.stream = c10::cuda::getCurrentCUDAStream();                                          \
    cfg.attrs = attr;                                                                        \
    cfg.numAttrs = 1;                                                                        \
    static int fits = -1; /* can this GPU run the cluster at all (asked once, eagerly) */   \
    if (fits < 0) {                                                                          \
      int n = 0;                                                                             \
      fits = cudaOccupancyMaxActiveClusters(&n, router_big<bm, bn, cs, st, wgm, wgn>, &cfg) == cudaSuccess && n > 0; \
      cudaGetLastError();                                                                    \
    }                                                                                        \
    TORCH_CHECK(fits, "router_big: clusters of ", cs, " do not fit this GPU");               \
    const cudaError_t e = cudaLaunchKernelEx(&cfg, router_big<bm, bn, cs, st, wgm, wgn>,               \
        reinterpret_cast<const __nv_bfloat16*>(hi.data_ptr()),                               \
        reinterpret_cast<const __nv_bfloat16*>(lo.data_ptr()),                               \
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), M);     \
    TORCH_CHECK(e == cudaSuccess, "router_big launch: ", cudaGetErrorString(e));             \
    return;                                                                                  \
  }
  ROUTER_BIG_VARIANTS(CASE)
#undef CASE
  TORCH_CHECK(false, "no router_big variant ", BM, " ", BN, " ", CS, " ", ST, " ", WGM, " ", WGN);
}

// Once, before any capture: let the shared-ring variants take all the shared memory a block may
// opt into (less room for static shared memory); returns that.
int64_t w12_init() {
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  w12_max_smem = optin - 1024;
  // The router_big variants: their shared memory, and clusters of 16 (not portable; a GPU
  // without them fails the launch, and the tuner skips that variant).
#define INIT_RB(bm, bn, cs, st, wgm, wgn)                                                                         \
  if (rb_smem_bytes<bm, bn, cs, st, wgm, wgn>() <= w12_max_smem) {                                                \
    cudaFuncSetAttribute(router_big<bm, bn, cs, st, wgm, wgn>, cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                         rb_smem_bytes<bm, bn, cs, st, wgm, wgn>());                                              \
    if (cs > 8) cudaFuncSetAttribute(router_big<bm, bn, cs, st, wgm, wgn>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); \
  }
  ROUTER_BIG_VARIANTS(INIT_RB)
#undef INIT_RB
  cudaGetLastError();  // clear an unsupported attribute's error
#define INIT(nt, st, w, rt, ks)                                                              \
  {                                                                                          \
    const cudaError_t e = cudaFuncSetAttribute(w12t_gemm<nt, st, w, rt, ks>,                 \
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,  \
                                               w12_max_smem);                                \
    TORCH_CHECK(e == cudaSuccess, "w12_init: ", cudaGetErrorString(e));                      \
  }
  W12T_VARIANTS(INIT)
#undef INIT
  return w12_max_smem;
}
"""

CPP_SRC = ("void router(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor sync, "
           "int64_t slot, int64_t NT, int64_t KTW, int64_t U);\n"
           "void router_sum(torch::Tensor in, torch::Tensor out);\n"
           "void w12(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, "
           "torch::Tensor esc_off, torch::Tensor esc, int64_t N, int64_t S, double scale, int64_t bb, "
           "int64_t kind, int64_t NT, int64_t A, int64_t B, int64_t C, int64_t D);\n"
           "int64_t w12_init();\n"
           "void router_big_run(torch::Tensor hi, torch::Tensor lo, torch::Tensor x, torch::Tensor out, "
           "int64_t BM, int64_t BN, int64_t CS, int64_t ST, int64_t WGM, int64_t WGN);")

# (NT token tiles of 8, KTW k tiles of 64 per warp, U tiles loaded ahead), as LAUNCH_SRC
# instantiates them.
ROUTER_VARIANTS = [(1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 1, 1), (2, 2, 2), (2, 4, 4),
                   (4, 1, 1), (4, 2, 2), (4, 4, 2), (8, 2, 2), (8, 4, 2)]
N_SLOTS = 256

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline("muse_w12_router", cpp_sources=CPP_SRC,
                           cuda_sources=KERNEL_SRC + W12_KERNEL_SRC + ROUTER_BIG_SRC + LAUNCH_SRC + W12_LAUNCH_SRC,
                           functions=["router", "router_sum", "w12", "w12_init", "router_big_run"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-use_fast_math"], verbose=False)
    return _ext


def repack(w):
    """W [N, K] -> fragment order [N/16][K/64][load 2h + q][lane 4g + c][8]: row 16 rt + 8h + g,
    k 64 kt + 32q + 8c + e. A permutation: same BF16 values, same bytes."""
    n, k = w.shape
    v = w.reshape(n // 16, 2, 8, k // 64, 2, 4, 8)  # rt, h, g, kt, q, c, e
    return v.permute(0, 3, 1, 4, 2, 5, 6).contiguous()


def _router_nt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 64 else 8


def _router(state, x, cfg):
    nt, ktw, u = cfg
    out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
    # Each (m, config) has its own counters: a call's epoch is its arrivals over its grid.
    key = (x.shape[0], cfg)
    slot = state["slots"].get(key)
    if slot is None:
        slot = state["slots"][key] = len(state["slots"])
        assert slot < N_SLOTS, "out of router counters"
    state["ext"].router(state["router_p"], x.contiguous(), out, state["sync"], slot, nt, ktw, u)
    return out


def _router_cublas(state, x):
    """SGLang's split path: two BF16 GEMMs on the hi / lo halves, FP32 accumulation."""
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _router_cat(state, x):
    """One cuBLAS GEMM over K = 16384: [x | x] @ [hi | lo].T, FP32 accumulation."""
    return torch.mm(torch.cat([x, x], dim=1), state["router_cat_t"], out_dtype=torch.float32)


def _router_bmm(state, x):
    """hi and lo as a batch of two GEMMs on the same x (a stride-0 view), FP32, then summed."""
    return torch.bmm(x.expand(2, *x.shape), state["router_hilo_t"], out_dtype=torch.float32).sum(0)


def _router_stack(state, x):
    """One cuBLAS GEMM against [hi; lo] stacked (N = 512, FP32 out), then the two halves added:
    as SGLang's two products, without concatenating x."""
    o = torch.mm(x, state["router_stack_t"], out_dtype=torch.float32)
    out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
    state["ext"].router_sum(o, out)
    return out


ROUTER_LIBS = {"cublas": _router_cublas, "cat": _router_cat, "bmm": _router_bmm, "stack": _router_stack}

# router_big's (BM, BN, CS, ST, WGM, WGN), as W12_LAUNCH_SRC instantiates them; tuned from 16
# tokens up.
ROUTER_BIG = [(32, 32, 16, 4, 2, 2), (64, 32, 8, 4, 2, 2), (64, 32, 16, 4, 2, 2), (64, 64, 8, 4, 2, 2),
              (64, 64, 16, 4, 2, 2), (128, 32, 8, 4, 2, 2), (64, 64, 16, 4, 2, 4), (64, 64, 8, 4, 2, 4),
              (128, 64, 16, 3, 4, 2), (128, 64, 8, 3, 4, 2), (128, 32, 16, 4, 4, 2), (64, 32, 16, 4, 4, 1)]
ROUTER_BIG_MIN_M = 16


def _router_big(state, x, cfg):
    out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
    state["ext"].router_big_run(state["router_hi"], state["router_lo"], x.contiguous(), out, *cfg[1:])
    return out


def _time(fn, flush, reps=30):
    """Median µs of one CUDA-graph replay of fn after reading a buffer larger than L2, as the
    judge times it."""
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
        flush.sum(dtype=torch.int64)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


# Router token counts to pick a variant for: the scored ones and the tests'.
TUNE_M = [1, 5, 8, 9, 16, 32, 128, 256]


def _tune(state):
    """Pick the router variant per token count by timing CUDA-graph replays, as the judge does."""
    flush = torch.ones(256 << 20, dtype=torch.uint8, device="cuda")
    nop = torch.zeros(1, device="cuda")
    print(f"TUNE empty graph {_time(lambda: nop.add_(1), flush):.1f}us", flush=True)
    table = {}
    for m in TUNE_M:
        x = torch.randn((m, 8192), device="cuda").to(torch.bfloat16)
        ref = x.double() @ state["router_f64_t"]
        res = []
        for name, fn in ROUTER_LIBS.items():
            try:
                err = float((fn(state, x).double() - ref).norm() / ref.norm())
            except Exception as e:  # e.g. no out_dtype for bmm in this torch
                print(f"  {name}: {type(e).__name__}: {e}", flush=True)
                continue
            if err > 1e-5:
                print(f"  BAD m={m} {name}: err {err:.2e}", flush=True)
                continue
            res.append((name, _time(lambda: fn(state, x), flush)))
        for v in ROUTER_VARIANTS:
            if v[0] != _router_nt(m):
                continue
            err = float((_router(state, x, v).double() - ref).norm() / ref.norm())
            if err > 1e-5:
                print(f"  BAD m={m} {v}: err {err:.2e}", flush=True)
                continue
            res.append((v, _time(lambda: _router(state, x, v), flush)))
        for v in ROUTER_BIG if m >= ROUTER_BIG_MIN_M else []:
            v = ("big",) + v
            try:
                err = float((_router_big(state, x, v).double() - ref).norm() / ref.norm())
            except RuntimeError as e:  # e.g. too much shared memory, or no clusters of 16
                print(f"  {v}: {e}".splitlines()[0], flush=True)
                continue
            if err > 1e-5:
                print(f"  BAD m={m} {v}: err {err:.2e}", flush=True)
                continue
            res.append((v, _time(lambda: _router_big(state, x, v), flush)))
        res.sort(key=lambda r: r[1])
        table[m] = res[0][0]
        print("TUNE router m=%d: " % m + ", ".join(f"{c}:{t:.1f}" for c, t in res), flush=True)
    del flush
    return table


def w12_pack(w):
    """W BF16 [N, K] -> W12 (see the docstring): the packed tiles uint8 [K/64, N/16, 3, 32, 16],
    the OR-ed exponent bits bb, the scale 2^2d, and the escapes, sorted by tile, as rows
    (k, row in tile, true weight * 2^-2d, decoded value) with the offsets esc_off[rt * KT + kt]."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    eh = (bits >> 8) & 0x7F
    e = (bits >> 7) & 0xFF
    hist = torch.bincount(eh[e > 0].flatten(), minlength=136)
    b = int(torch.stack([hist[i:i + 8].sum() for i in range(121)]).argmax())
    bp, d = b & ~7, b & 7
    s = (bits >> 15) & 1
    lo = bits & 0xFF
    ok = (eh >= b) & (eh <= b + 7) & (e > 0)
    idx = torch.where(ok, eh - b, torch.zeros_like(eh))
    dec = (s << 15) | ((bp + idx) << 8) | lo

    def frag(a):  # [N, K] -> [rt, kt, h, g, c, q, e]
        return a.reshape(n // 16, 2, 8, k // 64, 2, 4, 8).permute(0, 3, 1, 2, 5, 4, 6)
    L, S, I = frag(lo), frag(s), frag(idx)
    wp = torch.empty((n // 16, k // 64, 3, 32, 16), dtype=torch.uint8, device=w.device)
    wp[:, :, 0:2] = L.reshape(n // 16, k // 64, 2, 32, 16).to(torch.uint8)
    code = (S[..., 0:4] << 7) | (torch.roll(I[..., 4:8], -1, dims=-1) << 4) | (S[..., 4:8] << 3) | I[..., 0:4]
    wp[:, :, 2] = code.permute(0, 1, 3, 4, 2, 5, 6).reshape(n // 16, k // 64, 32, 16).to(torch.uint8)

    rows, cols = torch.nonzero(~ok, as_tuple=True)
    key = (rows // 16) * (k // 64) + cols // 64
    order = torch.argsort(key, stable=True)
    rows, cols, key = rows[order], cols[order], key[order]
    as_f32 = lambda v: (v.to(torch.int32) << 16).view(torch.float32)
    true = as_f32(bits[rows, cols])
    wt = (true.double() * 2.0 ** (-2 * d)).float()
    assert torch.equal(wt.double() * 2.0 ** (2 * d), true.double()), "an escape is not exact in FP32"
    esc = torch.stack([cols.to(torch.int32), (rows % 16).to(torch.int32), wt.view(torch.int32),
                       as_f32(dec[rows, cols]).view(torch.int32)], 1)
    off = torch.zeros((n // 16) * (k // 64) + 1, dtype=torch.int64, device=w.device)
    off[1:] = torch.cumsum(torch.bincount(key, minlength=(n // 16) * (k // 64)), 0)
    if not len(esc):
        esc = torch.zeros((1, 4), dtype=torch.int32, device=w.device)

    # Check the decode, as the kernel does it, against the weights: W * 2^-2d, or dec at escapes.
    wq = wp.view(torch.int32).reshape(n // 16, k // 64, 3, 32, 4)
    mask = 0x87878787 - (1 << 32)  # as an int32
    words = []  # (h, q, e) order
    for h in range(2):
        for q in range(2):
            nw = wq[:, :, 2, :, 2 * h + q]
            rot = (nw << 4) | ((nw >> 28) & 0xF)
            for grp, word in ((0, nw), (1, rot)):
                hh = (word & mask) | (bp * 0x01010101)
                lw = wq[:, :, h, :, 2 * q + grp]
                for j in range(4):
                    words.append(((hh >> (8 * j)) & 0xFF) << 8 | ((lw >> (8 * j)) & 0xFF))
    got = torch.stack(words, -1).reshape(n // 16, k // 64, 8, 4, 2, 2, 8)  # rt, kt, g, c, h, q, e
    got = got.permute(0, 4, 2, 1, 5, 3, 6).reshape(n, k)
    ref = torch.where(ok, bits - (2 * d << 7), dec)
    assert torch.equal(got, ref), "W12 decode mismatch"
    return {"wp": wp.transpose(0, 1).contiguous(), "wpr": wp, "bb": bp * 0x01010101, "scale": 2.0 ** (2 * d),
            "esc": esc.contiguous(), "esc_off": off.to(torch.int32), "n_esc": int(rows.numel()), "b": b}


W12_MAX_TOKENS = 32
W12_WS_FLOATS = 32 * W12_MAX_TOKENS * 8192  # the register kernel's partials: S <= 32 slices
# (NT, U, NW) and (NT, ST, NW, RT, KS), as W12_LAUNCH_SRC instantiates them.
W12R = [(1, 2, 8), (1, 2, 4), (2, 2, 8), (2, 2, 4), (4, 2, 8), (4, 2, 4),
        (1, 4, 8), (1, 4, 4), (1, 8, 8), (1, 8, 4), (2, 4, 8), (2, 4, 4)]  # deeper rings: more bytes in flight
W12C = [(1, 4, 4, 1, 1), (1, 8, 2, 1, 1), (1, 8, 4, 1, 1), (1, 16, 2, 1, 1), (1, 16, 4, 1, 1), (1, 4, 2, 1, 1), (1, 2, 4, 1, 1), (1, 8, 2, 2, 1), (1, 16, 2, 2, 1), (2, 8, 2, 1, 1), (2, 8, 4, 1, 1), (2, 16, 2, 1, 1), (2, 4, 2, 1, 1), (2, 8, 2, 2, 1), (2, 16, 2, 2, 1), (2, 8, 1, 2, 1), (4, 8, 2, 1, 1), (4, 4, 2, 1, 1), (4, 8, 1, 2, 1), (4, 8, 2, 2, 1), (2, 4, 2, 1, 4), (2, 8, 2, 1, 4), (2, 2, 2, 1, 8), (2, 4, 2, 2, 2), (2, 8, 2, 1, 2), (2, 4, 2, 1, 2), (4, 4, 2, 1, 4), (4, 2, 2, 1, 8), (4, 4, 2, 2, 2), (4, 8, 2, 1, 2), (4, 4, 2, 1, 2), (4, 2, 2, 2, 4), (2, 4, 4, 1, 1), (4, 4, 4, 1, 1)]  # (NT, NWK, U, RT, NWR)
W12T = ([(1, st, nw, rt, ks) for st, nw, rt, ks in
         ((8, 4, 1, 1), (6, 4, 1, 2), (16, 4, 1, 1), (24, 4, 1, 1), (16, 2, 1, 1), (12, 2, 2, 1), (8, 4, 2, 1),
          (3, 4, 1, 4), (4, 4, 1, 4), (6, 4, 1, 4), (3, 4, 1, 8))]
        + [(2, st, nw, rt, ks) for st, nw, rt, ks in
           ((8, 4, 1, 1), (6, 4, 1, 2), (16, 4, 1, 1), (16, 2, 1, 1), (8, 8, 1, 1), (8, 4, 2, 1), (4, 4, 1, 4),
            (6, 4, 1, 4))]
        + [(4, st, nw, rt, ks) for st, nw, rt, ks in
           ((8, 4, 1, 1), (6, 4, 1, 2), (12, 4, 1, 1), (8, 8, 1, 1), (6, 8, 1, 1), (8, 4, 2, 1), (4, 8, 2, 1),
            (4, 4, 1, 4))])


def _w12_nt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4


def _w12(state, x, layer, n, cfg):
    kind, a, b, c, d, s = cfg
    p = state["w12"][(layer, n)]
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    state["ext"].w12(p["wpr"] if kind == 2 else p["wp"], x, out, state["ws"], state["cnt"], p["esc_off"], p["esc"], n, s,
                     p["scale"], p["bb"], kind, _w12_nt(x.shape[0]), a, b, c, d)
    return out


def _w12_configs(n, k, m, max_smem):
    """(kind, A, B, C, D, S): kind 0 the register kernel (A = U, B = NW), kind 1 the TMA kernel
    (A = ST, B = NW, C = RT, D = KS; S <= 8 a cluster), kind 2 the contiguous kernel (A = U,
    B = NWK, C = RT, D = NWR; S = 1); S k slices."""
    kt, nt, out = k // 64, _w12_nt(m), []
    for (t, nwk, u, rt, nwr) in W12C:
        if t == nt and kt % nwk == 0 and (kt // nwk) % u == 0 and (n // 16) % (rt * nwr) == 0:
            out.append((2, u, nwk, rt, nwr, 1))
    for s in (1, 2, 4, 8, 16, 32):
        if kt % s:
            continue
        ktc = kt // s
        for (t, u, nw) in W12R:
            if t == nt and ktc % u == 0 and 48 <= n // 16 // nw * s <= 4224:
                out.append((0, u, nw, 0, 0, s))
        for (t, st, nw, rt, ks) in W12T:
            stage = 8 * nt * (ks * 64 + 32) * 2 + ks * nw * rt * 1536
            smem = max(st * stage, 8 * nt * 16 * nw * rt * 4)
            if (t == nt and s <= 8 and ktc % ks == 0 and (n // 16) % (nw * rt) == 0 and smem <= max_smem
                    and 16 <= n // (16 * nw * rt) * s <= 4224):
                out.append((1, st, nw, rt, ks, s))
    return out


# (layer, n) the judge runs at up to 32 tokens, and the token counts to tune: the scored ones;
# others take the next one up.
W12_SHAPES = [("qkv_gate", 2304), ("o", 8192), ("pooler", 4096), ("unpooler", 8192), ("unpooler", 1024)]
W12_MS = {("unpooler", 8192): [16, 32], ("unpooler", 1024): [1, 8]}

# Where cuBLAS runs, the same product asked four ways: cuBLAS's heuristics pick a different kernel
# for each (cuBLAS reads the pooler at ~2.5 TB/s as x @ W.T but the unpooler, as large, at ~3.3).
BLAS = {
    "tn": lambda s, x, l, n: x @ s["w"][l][:n].T,
    "nn": lambda s, x, l, n: x @ s["wt"][l][:, :n],
    "nt": lambda s, x, l, n: (s["w"][l][:n] @ x.T).T,
    "bias": lambda s, x, l, n: torch.nn.functional.linear(x, s["w"][l][:n], s["zb"][:n]),  # cuBLASLt
    "bias_t": lambda s, x, l, n: torch.addmm(s["zb"][:n], x, s["wt"][l][:, :n]),
}
BLAS_MS = {("unpooler", 8192): [16, 32, 128, 256], ("unpooler", 1024): [1, 8]}


def _tune_blas(state, flush):
    """For each shape and token count the fastest BLAS variant; returns its times too."""
    table, times = {}, {}
    for layer, n in W12_SHAPES:
        w = state["w"][layer][:n]
        for m in BLAS_MS.get((layer, n), [1, 8, 16, 32, 128, 256]):
            x = torch.randn((m, w.shape[1]), device="cuda").to(torch.bfloat16)
            ref = x.float() @ w.float().T
            res = []
            for v, fn in BLAS.items():
                out = fn(state, x, layer, n)
                assert out.shape == ref.shape and float((out.float() - ref).norm() / ref.norm()) < 1e-2, v
                res.append((_time(lambda: fn(state, x, layer, n), flush, reps=40), v))
            res.sort()
            tn = next(t for t, v in res if v == "tn")
            # a small 0.05 µs margin to favor standard x @ W.T in ties
            times[(layer, n, m)], table[(layer, n, m)] = res[0] if res[0][0] < tn - 0.05 else (tn, "tn")
            print(f"TUNE blas {layer} n={n} m={m}: " + " ".join(f"{v}:{t:.1f}" for t, v in res), flush=True)
    return table, times


def _blas_cfg(state, layer, n, m):
    tab = state["blas"]
    ms = sorted(t[2] for t in tab if t[0] == layer and t[1] == n)
    return tab[(layer, n, next((u for u in ms if u >= m), ms[-1]))]


def _tune_w12(state, deadline):
    """For each shape and token count pick the fastest W12 config, or cuBLAS (None) if that's faster."""
    flush = torch.ones(256 << 20, dtype=torch.uint8, device="cuda")
    state["blas"], blas_t = _tune_blas(state, flush)
    table = {}
    for layer, n in W12_SHAPES:
        w = state["w"][layer][:n]
        for m in W12_MS.get((layer, n), [1, 8, 16, 32]):
            if time.time() > deadline:
                table[(layer, n, m)] = None
                continue
            x = torch.randn((m, w.shape[1]), device="cuda").to(torch.bfloat16)
            ref = (x.double() @ w.double().T).float()
            rn = float(ref.norm())
            base = blas_t[(layer, n, m)]
            res = []
            for cfg in _w12_configs(n, w.shape[1], m, state["w12_smem"]):
                err = float((_w12(state, x, layer, n, cfg).float() - ref).norm()) / rn
                if err > 5e-3:
                    print(f"  BAD {layer} n={n} m={m} {cfg}: err {err:.2e}", flush=True)
                    continue
                res.append((_time(lambda: _w12(state, x, layer, n, cfg), flush, reps=10), cfg))
            res.sort()
            best, bc = base, None
            for _, cfg in res[:3]:
                t = _time(lambda: _w12(state, x, layer, n, cfg), flush, reps=40)
                if t < best:
                    best, bc = t, cfg
            table[(layer, n, m)] = bc
            reg = [r for r in res if r[1][0] == 0][:3]
            shr = [r for r in res if r[1][0] == 1][:8]
            con = [r for r in res if r[1][0] == 2][:4]
            print(f"TUNE {layer} n={n} m={m}: cublas {base:.1f}us w12 {best:.1f}us {bc} | reg "
                  + " ".join(f"{c[1:]}:{t:.1f}" for t, c in reg) + " | tma "
                  + " ".join(f"{c}:{t:.1f}" for t, c in shr) + " | contig "
                  + " ".join(f"{c[1:5]}:{t:.1f}" for t, c in con), flush=True)
    del flush
    return table


def _w12_cfg(state, layer, n, m):
    if m > W12_MAX_TOKENS:
        return None
    tab = state["w12_table"]
    ups = sorted(t[2] for t in tab if t[0] == layer and t[1] == n and t[2] >= m)
    return tab[(layer, n, ups[0])] if ups else None


def prepare(weights: Weights):
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    t0 = time.time()
    ext = _module()
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)
    dev = router.device
    state = {
        "ext": ext,
        "w": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "wt": {name: weights[name].t().contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "zb": torch.zeros(8192, dtype=torch.bfloat16, device=router.device),
        "router_p": repack(torch.cat([hi, lo], dim=1)),
        "router_hi": hi.contiguous(),
        "router_lo": lo.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "router_cat_t": torch.cat([hi, lo], dim=1).t().contiguous(),
        "router_stack_t": torch.cat([hi, lo], dim=0).t().contiguous(),
        "router_hilo_t": torch.stack([hi.t(), lo.t()]).contiguous(),
        "router_f64_t": router.double().t(),
        "sync": torch.zeros(2 * N_SLOTS, dtype=torch.int32, device=dev),
        "slots": {},
    }
    state["w12_smem"] = ext.w12_init()
    state["table"] = _tune(state)
    del state["router_f64_t"]
    state["w12"] = {}
    for name, n in W12_SHAPES:  # the unpooler's first 1024 rows packed apart: tiles are k-major
        state["w12"][(name, n)] = p = w12_pack(state["w"][name][:n])
        print(f"W12 {name} n={n}: window b={p['b']}, {p['n_esc']} escapes "
              f"({p['n_esc'] / (n * state['w'][name].shape[1]):.1e})", flush=True)
    state["ws"] = torch.zeros(W12_WS_FLOATS, dtype=torch.float32, device=dev)
    state["cnt"] = torch.zeros(1 << 12, dtype=torch.int32, device=dev)
    print(f"packed at {time.time() - t0:.0f}s", flush=True)
    state["w12_table"] = _tune_w12(state, t0 + 200)
    print(f"tuned at {time.time() - t0:.0f}s", flush=True)
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n, m = cfg["layer"], cfg["n"], cfg["m"]
    if layer != "router":
        bc = _w12_cfg(state, layer, n, m)
        if bc is None:
            return BLAS[_blas_cfg(state, layer, n, m)](state, x, layer, n)
        return _w12(state, x.contiguous(), layer, n, bc)
    v = state["table"].get(m)
    if v is None:  # an untuned token count: the tuned choice for the next one up
        up = [t for t in TUNE_M if t >= m]
        v = state["table"][up[0]] if up else state["table"][TUNE_M[-1]]
        if v not in ROUTER_LIBS and v[0] != "big" and v[0] != _router_nt(m):
            v = {1: (1, 2, 2), 2: (2, 2, 2), 4: (4, 4, 2), 8: (8, 2, 2)}[_router_nt(m)]
    if v in ROUTER_LIBS:
        return ROUTER_LIBS[v](state, x)
    if v[0] == "big":
        return _router_big(state, x, v)
    return _router(state, x, v)
