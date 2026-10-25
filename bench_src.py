BENCH_SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <stdint.h>

__device__ __forceinline__ void b_init(uint32_t bar, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(n) : "memory"); }
__device__ __forceinline__ void b_tx(uint32_t bar, uint32_t b) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(b) : "memory"); }
__device__ __forceinline__ void b_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" ::"r"(bar), "r"(ph) : "memory"); }
__device__ __forceinline__ void t3(uint32_t dst, const CUtensorMap* m, int a, int b, int c, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(m), "r"(a), "r"(b), "r"(c), "r"(bar) : "memory"); }

// Each CTA streams `per` boxes of R rows x BK k (3D map) starting at box index b0, ST-deep ring.
// Box j of CTA: row block (j % RB), k block (j / RB)... we walk k-major inside a row block.
template <int ST>
__global__ void tma_stream(const __grid_constant__ CUtensorMap map, int R, int BK, int nkb, int nrb, int total, int* sink) {
  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ __align__(8) uint64_t full[ST];
  const int bytes = R * BK * 2;
  const int b0 = (long long)total * blockIdx.x / gridDim.x, b1 = (long long)total * (blockIdx.x + 1) / gridDim.x;
  const uint32_t base = ((uint32_t)__cvta_generic_to_shared(sm) + 1023) & ~1023u;
  const uint32_t f0 = (uint32_t)__cvta_generic_to_shared(full);
  if (threadIdx.x != 0) return;
  for (int s = 0; s < ST; ++s) b_init(f0 + 8 * s, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  auto issue = [&](int i) {
    const int j = b0 + i, rb = j / nkb, kb = j % nkb;
    b_tx(f0 + 8 * (i % ST), bytes);
    t3(base + (i % ST) * bytes, &map, 0, kb * (BK / 64), rb * R, f0 + 8 * (i % ST));
  };
  const int n = b1 - b0;
  for (int i = 0; i < ST && i < n; ++i) issue(i);
  int acc = 0;
  for (int i = 0; i < n; ++i) {
    b_wait(f0 + 8 * (i % ST), (i / ST) & 1);
    acc += *(volatile int*)(sm + (base - (uint32_t)__cvta_generic_to_shared(sm)) + (i % ST) * bytes);
    if (i + ST < n) issue(i + ST);
  }
  if (acc == 0x7fffffff) sink[0] = acc;
}

// Plain loads: each thread streams 16-byte loads, U in flight, over its contiguous chunk.
template <int U>
__global__ void ldg_stream(const uint4* __restrict__ p, long long n16, int* sink) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nt = (long long)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (long long i = tid; i < n16; i += nt * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long j = i + u * nt;
      if (j < n16) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + j));
      else v[u] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].w;
  }
  if (acc == 0x12345678) sink[0] = acc;
}

void bench_tma(torch::Tensor w, int64_t R, int64_t BK, int64_t ST, int64_t G, torch::Tensor sink) {
  static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
  if (!enc) { cudaDriverEntryPointQueryResult q; cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&enc, cudaEnableDefault, &q); }
  const int rows = w.size(0), cols = w.size(1);
  CUtensorMap map;
  cuuint64_t dims[3] = {64, (cuuint64_t)cols / 64, (cuuint64_t)rows};
  cuuint64_t strides[2] = {128, (cuuint64_t)cols * 2};
  cuuint32_t box[3] = {64, (cuuint32_t)BK / 64, (cuuint32_t)R};
  cuuint32_t el[3] = {1, 1, 1};
  TORCH_CHECK(enc(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, w.data_ptr(), dims, strides, box, el, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  const int nkb = cols / BK, nrb = rows / R, total = nkb * nrb;
  const int smem = ST * R * BK * 2 + 1024;
  auto s = c10::cuda::getCurrentCUDAStream();
#define L(st) if (ST == st) { cudaFuncSetAttribute(tma_stream<st>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
    tma_stream<st><<<G, 32, smem, s>>>(map, R, BK, nkb, nrb, total, sink.data_ptr<int>()); return; }
  L(2) L(3) L(4) L(6) L(8) L(12) L(16)
  TORCH_CHECK(false, "ST");
}

void bench_ldg(torch::Tensor w, int64_t U, int64_t G, int64_t T, torch::Tensor sink) {
  auto s = c10::cuda::getCurrentCUDAStream();
  const long long n16 = w.numel() * 2 / 16;
  if (U == 4) ldg_stream<4><<<G, T, 0, s>>>((const uint4*)w.data_ptr(), n16, sink.data_ptr<int>());
  else if (U == 8) ldg_stream<8><<<G, T, 0, s>>>((const uint4*)w.data_ptr(), n16, sink.data_ptr<int>());
  else ldg_stream<16><<<G, T, 0, s>>>((const uint4*)w.data_ptr(), n16, sink.data_ptr<int>());
}
'''


def _bench(state, flush):
    from torch.utils.cpp_extension import load_inline as _li
    b = _li("okc_stream_bench", cpp_sources="void bench_tma(torch::Tensor w, int64_t R, int64_t BK, int64_t ST, int64_t G, torch::Tensor sink);\n"
            "void bench_ldg(torch::Tensor w, int64_t U, int64_t G, int64_t T, torch::Tensor sink);",
            cuda_sources=BENCH_SRC, functions=["bench_tma", "bench_ldg"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    sms = state["sms"]
    for name in ("pooler", "qkv_gate"):
        w = state["w"][name]
        mb = w.numel() * 2 / 1e6
        res = []
        for R, BK in [(16, 256), (32, 256), (64, 256), (128, 256), (256, 256), (64, 512), (32, 1024), (64, 128), (128, 64), (256, 64)]:
            for ST in (2, 3, 4, 6, 8, 12, 16):
                smem = ST * R * BK * 2 + 1024
                if smem > 227 * 1024:
                    continue
                per_sm = max(1, (228 * 1024) // (smem + 1024))
                for cps in sorted({1, min(per_sm, 2), min(per_sm, 4), per_sm}):
                    G = sms * cps
                    t = _time(lambda: b.bench_tma(w, R, BK, ST, G, sink), flush)
                    res.append((t, f"tma R={R} BK={BK} ST={ST} G={G} stage={R*BK*2//1024}KB"))
        for U in (4, 8, 16):
            for T in (256, 512, 1024):
                for cps in (1, 2, 4):
                    t = _time(lambda: b.bench_ldg(w, U, sms * cps, T, sink), flush)
                    res.append((t, f"ldg U={U} T={T} G={sms*cps}"))
        res.sort()
        empty = _time(lambda: sink.add_(1), flush)
        print(f"BENCH {name} {mb:.0f}MB empty-graph {empty:.1f}us", flush=True)
        for t, d in res[:25]:
            print(f"BENCH {name} {t:.1f}us ({mb/(t-empty):.2f} TB/s net) {d}", flush=True)
        for t, d in res:
            if "BK=256 ST=4" in d and "R=64" in d:
                print(f"BENCH {name} [ref] {t:.1f}us {d}", flush=True)
