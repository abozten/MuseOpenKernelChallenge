# okc: problem=muse_bf16_proj
"""Skinny BF16 GEMMs for the dense projections: one CUDA kernel over repacked weights.

out[m, n] = x[m, k] @ W[n, k].T. At these token counts the projections are bound by reading
the weights, and on H200 a weight read is fastest when every warp load is 512 contiguous
bytes, as in a plain linear sweep. mma.sync's fragments put 8 different rows in every warp
load, so prepare() repacks W (losslessly, still BF16) into fragment order:

- A 16-row x 64-k tile of W is 2 KB, stored as 4 warp loads of 32 lanes x 16 bytes, lane
  (g, c) of load 2h + q holding row 8h + g, k 8c + 32q to + 8. A row tile's k tiles follow
  each other, so a warp streams its rows along k through contiguous memory, straight into
  registers, U k tiles ahead.
- A dot product doesn't care about k's order, so the lane's words go to m16n8k16 in that
  permuted k order; x, a few rows, is read in the same order (16-byte loads from L1/L2).

Work is shared out dynamically, a warp at a time, so SMs that get more bandwidth do more of
it: a warp owns 16 * RT rows and 8 * NT tokens of the output, and takes chunks of C k tiles
of such a tile from a ticket counter (64-bit, never reset: a call takes exactly chunks +
warps tickets, so a ticket's chunk is its value mod that). A tile's chunks are its k
segments, handed out in order. Every segment but the last stores its FP32 partial and sets a
flag; the warp with the last segment waits for the flags (their chunks were handed out
before its own, to warps that are running), adds the partials to its own, writes the output
and clears the flags. A call is one launch, CUDA-graph safe.

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

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
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

// wp: W repacked (see the module docstring), [N / 16][K / 64][4][32] uint4. x: [M, KX] BF16.
// out: [M, N] (BF16, or FP32 when OUT_F32). Any grid of THREADS-thread CTAs.
// part: FP32 partials, [tile][segment][32 lanes][RT NT 4]; flags: an int per (tile,
// segment), zero between calls; ticket: this config's counter. C: k tiles per chunk, a
// multiple of U dividing K / 64. dbg (tuning only, else null): per warp, ns at start, end.
template <int NT, int RT, int U, bool OUT_F32>
__global__ void __launch_bounds__(THREADS) skinny_gemm(
    const uint4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ out,
    float* __restrict__ part, int* __restrict__ flags, unsigned long long* __restrict__ ticket,
    int M, int N, int K, int KX, int C, unsigned long long* __restrict__ dbg) {
  constexpr int TOKS = 8 * NT;
  constexpr int FRAG = RT * NT * 4;  // a lane's accumulators
  const unsigned long long t_start = dbg ? global_ns() : 0;

  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c = lane & 3;
  const int n_rt = N / (16 * RT);  // row tiles of a warp
  const int ktiles = K / 64;
  const int S = ktiles / C;  // segments per tile
  const long long chunks = (long long)n_rt * ((M + TOKS - 1) / TOKS) * S;
  const unsigned long long period = chunks + (long long)gridDim.x * WARPS;

  auto grab = [&]() -> long long {
    unsigned long long v = 0;
    if (lane == 0) v = atomicAdd(ticket, 1ull);
    v = __shfl_sync(0xffffffffu, v, 0);
    return (long long)(v % period);
  };

  // The W fragments (RT row tiles) and x (NT token tiles) of k tile j of chunk ch.
  auto load = [&](long long ch, int j, uint4 (&wb)[RT][4], uint4 (&xb)[NT][2]) {
    const int tile = (int)(ch / S), kt = (int)(ch % S) * C + j;
    const int rt0 = (tile % n_rt) * RT, tok0 = (tile / n_rt) * TOKS;
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint4* p = wp + ((size_t)(rt0 + r) * ktiles + kt) * 128 + lane;
#pragma unroll
      for (int l = 0; l < 4; ++l) wb[r][l] = ld_stream(p + l * 32);
    }
    const int kx = (kt * 64) % KX;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int tok = tok0 + 8 * t + g;
      const __nv_bfloat16* px = x + (size_t)(tok < M ? tok : 0) * KX + kx + 8 * c;
#pragma unroll
      for (int q = 0; q < 2; ++q)
        xb[t][q] = tok < M ? __ldg(reinterpret_cast<const uint4*>(px + 32 * q)) : make_uint4(0, 0, 0, 0);
    }
  };

  long long cur = grab();
  if (cur >= chunks) return;
  long long nxt = grab();

  float acc[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][t][e] = 0.f;

  uint4 wb[U][RT][4], xb[U][NT][2];
#pragma unroll
  for (int j = 0; j < U; ++j) load(cur, j, wb[j], xb[j]);

  while (true) {
    for (int j0 = 0; j0 < C; j0 += U) {
#pragma unroll
      for (int j = 0; j < U; ++j) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          // k step ks takes words 2 (ks & 1) and 2 (ks & 1) + 1 of the lane's chunk q = ks >> 1:
          // load 2h + q holds row 8h + g.
          const int q = ks >> 1, w0 = (ks & 1) * 2;
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            const uint32_t a0 = word(wb[j][r][q], w0), a2 = word(wb[j][r][q], w0 + 1);
            const uint32_t a1 = word(wb[j][r][2 + q], w0), a3 = word(wb[j][r][2 + q], w0 + 1);
#pragma unroll
            for (int t = 0; t < NT; ++t)
              mma_bf16(acc[r][t], a0, a1, a2, a3, word(xb[j][t][q], w0), word(xb[j][t][q], w0 + 1));
          }
        }
        // Refill this slot with the k tile U ahead: later in this chunk, or early in the next.
        const int ahead = j0 + j + U;
        if (ahead < C) load(cur, ahead, wb[j], xb[j]);
        else if (nxt < chunks) load(nxt, ahead - C, wb[j], xb[j]);
      }
    }

    // The chunk is done: flush acc. acc[r][t][e] is row (row tile rt0 + r) 16 + g + 8 (e >> 1),
    // token tok0 + 8 t + 2 c + (e & 1).
    const int tile = (int)(cur / S), seg = (int)(cur % S);
    const int row0 = (tile % n_rt) * RT * 16 + g, tok0 = (tile / n_rt) * TOKS;
    int* tflags = flags + (size_t)tile * S;
    float* tpart = part + (size_t)tile * S * 32 * FRAG + lane * FRAG;
    bool write = true;
    if (seg < S - 1) {
      float* mine = tpart + (size_t)seg * 32 * FRAG;
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t)
          __stcg(reinterpret_cast<float4*>(mine + (r * NT + t) * 4),
                 make_float4(acc[r][t][0], acc[r][t][1], acc[r][t][2], acc[r][t][3]));
      __threadfence();
      __syncwarp();
      if (lane == 0) st_release(tflags + seg, 1);
      write = false;
    } else if (S > 1) {
      // The last segment: wait for the others' flags (lane s polls segment s), then add them.
      for (int s = lane; s < S - 1; s += 32)
        while (!ld_acquire(tflags + s)) {}
      __syncwarp();
      for (int s = 0; s < S - 1; ++s) {
        const float* other = tpart + (size_t)s * 32 * FRAG;
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            const float4 v = __ldcg(reinterpret_cast<const float4*>(other + (r * NT + t) * 4));
            acc[r][t][0] += v.x; acc[r][t][1] += v.y; acc[r][t][2] += v.z; acc[r][t][3] += v.w;
          }
      }
      for (int s = lane; s < S - 1; s += 32) tflags[s] = 0;
    }
    if (write) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int tok = tok0 + 8 * t + 2 * c + (e & 1);
            if (tok < M) {
              const size_t o = (size_t)tok * N + row0 + 16 * r + 8 * (e >> 1);
              if (OUT_F32) reinterpret_cast<float*>(out)[o] = acc[r][t][e];
              else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(acc[r][t][e]);
            }
          }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[r][t][e] = 0.f;

    cur = nxt;
    if (cur >= chunks) break;
    nxt = grab();
  }
  if (dbg && lane == 0) {
    const int w = blockIdx.x * WARPS + (threadIdx.x >> 5);
    dbg[2 * w] = t_start;
    dbg[2 * w + 1] = global_ns();
  }
}
"""

LAUNCH_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <int NT, int RT, int U, bool F32>
static void launch(const torch::Tensor& wp, const torch::Tensor& x, torch::Tensor& out,
                   torch::Tensor& part, torch::Tensor& flags, unsigned long long* ticket, int M,
                   int N, int K, int KX, int C, int G, unsigned long long* dbg) {
  skinny_gemm<NT, RT, U, F32><<<G, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr(),
      part.data_ptr<float>(), flags.data_ptr<int>(), ticket, M, N, K, KX, C, dbg);
}

// (NT, RT, U); keep in sync with VARIANTS in Python.
#define SKINNY_VARIANTS(X)                                                        \
  X(1, 1, 4) X(1, 1, 8) X(1, 2, 4) X(2, 1, 4) X(2, 1, 8) X(2, 2, 4)                \
  X(4, 1, 4) X(4, 2, 2) X(8, 1, 2) X(8, 1, 4)

void skinny(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor part,
            torch::Tensor flags, torch::Tensor tickets, int64_t slot, int64_t K, int64_t C,
            int64_t G, int64_t NT, int64_t RT, int64_t U, c10::optional<torch::Tensor> dbg_t) {
  unsigned long long* dbg =
      dbg_t ? reinterpret_cast<unsigned long long*>(dbg_t->data_ptr<int64_t>()) : nullptr;
  unsigned long long* ticket = reinterpret_cast<unsigned long long*>(tickets.data_ptr<int64_t>()) + slot;
  const int M = x.size(0), KX = x.size(1), N = out.size(1);
  const bool f32 = out.scalar_type() == torch::kFloat32;
#define CASE(nt, rt, u)                                                                       \
  if (NT == nt && RT == rt && U == u) {                                                       \
    if (f32) launch<nt, rt, u, true>(wp, x, out, part, flags, ticket, M, N, K, KX, C, G, dbg);  \
    else launch<nt, rt, u, false>(wp, x, out, part, flags, ticket, M, N, K, KX, C, G, dbg);     \
    return;                                                                                   \
  }
  SKINNY_VARIANTS(CASE)
#undef CASE
  TORCH_CHECK(false, "no skinny_gemm variant NT=", NT, " RT=", RT, " U=", U);
}
"""

CPP_SRC = ("void skinny(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor part, "
           "torch::Tensor flags, torch::Tensor tickets, int64_t slot, int64_t K, int64_t C, int64_t G, "
           "int64_t NT, int64_t RT, int64_t U, c10::optional<torch::Tensor> dbg_t);")

# (NT token tiles of 8, RT row tiles of 16 per warp, U k tiles loaded ahead), as LAUNCH_SRC
# instantiates them.
VARIANTS = [(1, 1, 4), (1, 1, 8), (1, 2, 4), (2, 1, 4), (2, 1, 8), (2, 2, 4),
            (4, 1, 4), (4, 2, 2), (8, 1, 2), (8, 1, 4)]
CHUNKS = (4, 8, 16, 32)
CTAS_PER_SM = (4, 8)
PART_FLOATS = 1 << 22
N_FLAGS = 1 << 16
N_TICKETS = 1 << 12

LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (8192, 4096), "router": (256, 8192)}

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline("muse_skinny_gemm_dyn", cpp_sources=CPP_SRC,
                           cuda_sources=KERNEL_SRC + LAUNCH_SRC, functions=["skinny"],
                           extra_cuda_cflags=["-O3"], verbose=False)
    return _ext


def repack(w):
    """W [N, K] -> fragment order [N/16][K/64][load 2h + q][lane 4g + c][8]: row 16 rt + 8h + g,
    k 64 kt + 32q + 8c + e. A permutation: same BF16 values, same bytes."""
    n, k = w.shape
    v = w.reshape(n // 16, 2, 8, k // 64, 2, 4, 8)  # rt, h, g, kt, q, c, e
    return v.permute(0, 3, 1, 4, 2, 5, 6).contiguous()


def _nt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def _valid(m, n, k, router, v, chunk):
    nt, rt, u = v
    ktiles = k // 64
    if n % (16 * rt) or ktiles % chunk or chunk % u or (router and (k // 2) % (64 * chunk)):
        return False
    tiles = n // (16 * rt) * -(-m // (8 * nt))
    segs = ktiles // chunk
    return tiles * segs <= N_FLAGS and tiles * segs * 32 * rt * nt * 4 <= PART_FLOATS


def _default(state, m, n, k, router):
    """For shapes without a tuned entry: the first fitting variant, 16 k tiles a chunk."""
    nt = _nt(m)
    for chunk in (16, 8, 4, 32):
        for v in VARIANTS:
            if v[0] == nt and _valid(m, n, k, router, v, chunk):
                return v + (chunk, 4 * state["sms"])
    raise ValueError(f"no variant for m={m} n={n} k={k}")


def _launch(state, x, layer, n, cfg, dbg=None):
    nt, rt, u, chunk, grid = cfg
    out = torch.empty((x.shape[0], n), device=x.device,
                      dtype=torch.float32 if layer == "router" else torch.bfloat16)
    # Each (shape, config) has its own ticket counter: a call takes a fixed number of tickets.
    key = (x.shape[0], n, layer, cfg)
    slot = state["slots"].get(key)
    if slot is None:
        slot = state["slots"][key] = len(state["slots"])
        assert slot < N_TICKETS, "out of ticket counters"
    state["ext"].skinny(state["wp"][layer], x.contiguous(), out, state["part"], state["flags"],
                        state["tickets"], slot, state["k"][layer], chunk, grid, nt, rt, u, dbg)
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


def _spread(state, m, layer, cfg, flush):
    """Print when the warps of a config finish, from the kernel's per-warp timestamps."""
    n = _n(layer, m)
    x = torch.randn((m, LAYERS[layer][1]), device="cuda").to(torch.bfloat16)
    dbg = torch.zeros((cfg[-1] * 4, 2), dtype=torch.int64, device="cuda")
    rows = []
    for _ in range(5):
        dbg.zero_()
        flush.sum(dtype=torch.int64)
        _launch(state, x, layer, n, cfg, dbg)
        torch.cuda.synchronize()
        d = dbg[dbg[:, 0] > 0].double()
        t0 = d[:, 0].min()
        end = (d[:, 1] - t0) / 1e3
        rows.append([float(end.max()), float(end.median()), float(end.min()), float((d[:, 0] - t0).max()) / 1e3])
    r = sorted(rows)[2]
    mb = n * state["k"][layer] * 2 / 1e6
    print(f"SPREAD m={m} {layer} {cfg}: span {r[0]:.1f}us ({mb / r[0]:.2f} TB/s) | warp end median {r[1]:.1f}"
          f" min {r[2]:.1f} | start skew {r[3]:.1f}", flush=True)


SCORED_M = [1, 8, 16, 32, 128, 256]


def _tune(state):
    flush = torch.ones(256 << 20, dtype=torch.uint8, device="cuda")
    table = {}
    t_start = time.time()
    # The fixed cost of an empty graph replay, to read the times below as kernel time.
    nop = torch.zeros(1, device="cuda")
    print(f"TUNE empty graph {_time(lambda: nop.add_(1), flush):.1f}us", flush=True)
    for m in SCORED_M:
        for layer, (_, kk) in LAYERS.items():
            router = layer == "router"
            n = _n(layer, m)
            k = state["k"][layer]
            x = torch.randn((m, kk), device="cuda").to(torch.bfloat16)
            ref = _cublas(state, x, layer, n).float()
            res = [("cublas", _time(lambda: _cublas(state, x, layer, n), flush))]
            if m <= 64 or router:
                nt = _nt(m)
                for v in VARIANTS:
                    if v[0] != nt:
                        continue
                    for chunk in CHUNKS:
                        if not _valid(m, n, k, router, v, chunk):
                            continue
                        for cps in CTAS_PER_SM:
                            cfg = v + (chunk, cps * state["sms"])
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
    for m, layer in [(1, "pooler"), (1, "qkv_gate"), (1, "unpooler"), (1, "router"), (16, "pooler")]:
        cfg = table[(m, layer)]
        if cfg != "cublas":
            _spread(state, m, layer, cfg, flush)
    del flush
    return table


def prepare(weights: Weights):
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    t0 = time.time()
    ext = _module()
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)
    w = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    w["router"] = torch.cat([hi, lo], dim=1).contiguous()
    dev = router.device
    state = {
        "ext": ext,
        "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
        "w": w,
        "k": {name: v.shape[1] for name, v in w.items()},
        "wp": {name: repack(v) for name, v in w.items()},
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "part": torch.empty(PART_FLOATS, dtype=torch.float32, device=dev),
        "flags": torch.zeros(N_FLAGS, dtype=torch.int32, device=dev),
        "tickets": torch.zeros(N_TICKETS, dtype=torch.int64, device=dev),
        "slots": {},
    }
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
            cfg = _default(state, m, n, state["k"][layer], layer == "router")
    if cfg == "cublas":
        return _cublas(state, x, layer, n)
    return _launch(state, x, layer, n, cfg)
