# okc: problem=muse_bf16_proj
"""Probe, not a submission: times a plain streaming read at the weights' sizes, cold L2 as the
judge times it, to see what bandwidth one kernel can reach per size. kernel_fn is cuBLAS."""

import time
from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
template <int U>
__global__ void rd(const uint4* __restrict__ p, long n, unsigned* out) {
  unsigned acc = 0;
  const long stride = (long)gridDim.x * blockDim.x;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + i + u * stride));
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n; i += stride) acc ^= p[i].x;
  if (acc == 0x12345678u) out[0] = acc;
}
// Each CTA reads one contiguous chunk (as a per-CTA-contiguous weight layout would).
template <int U>
__global__ void rdc(const uint4* __restrict__ p, long n, unsigned* out) {
  unsigned acc = 0;
  const long per = (n + gridDim.x - 1) / gridDim.x, b = blockIdx.x * per, e = min(n, b + per);
  long i = b + threadIdx.x;
  for (; i + (U - 1) * blockDim.x < e; i += U * blockDim.x) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + i + u * blockDim.x));
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < e; i += blockDim.x) acc ^= p[i].x;
  if (acc == 0x12345678u) out[0] = acc;
}
__global__ void nop() {}
void run(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t u, int64_t contig) {
  auto s = c10::cuda::getCurrentCUDAStream();
  const long n = buf.numel() / 16;
  auto* p = reinterpret_cast<const uint4*>(buf.data_ptr());
  auto* o = reinterpret_cast<unsigned*>(out.data_ptr());
  if (u == 0) { nop<<<1, 32, 0, s>>>(); return; }
#define L(U) if (u == U) { if (contig) rdc<U><<<grid, threads, 0, s>>>(p, n, o); else rd<U><<<grid, threads, 0, s>>>(p, n, o); return; }
  L(1) L(2) L(4) L(8)
}
"""


def _time(fn, flush, reps=40):
    fn()
    torch.cuda.synchronize()
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


def prepare(weights: Weights):
    t0 = time.time()
    ext = load_inline("probe_rd", cpp_sources="void run(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);",
                      cuda_sources=SRC, functions=["run"], extra_cuda_cflags=["-O3", "-arch=sm_90a"])
    print(f"compiled {time.time() - t0:.0f}s", flush=True)
    flush = torch.ones(256 << 20, dtype=torch.uint8, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    big = torch.empty(64 << 20, dtype=torch.uint8, device="cuda").random_(0, 255)
    empty = _time(lambda: ext.run(big, out, 1, 32, 0, 0), flush)
    print(f"PROBE empty-kernel graph {empty:.2f}us", flush=True)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for mb in (6, 12, 28, 50, 64):
        buf = big[: mb << 20]
        res = []
        for contig in (0, 1):
            for cpsm in (1, 2, 4, 8):
                for thr in (256, 512):
                    for u in (2, 4, 8):
                        t = _time(lambda: ext.run(buf, out, sms * cpsm, thr, u, contig), flush, reps=15)
                        res.append((t, contig, cpsm, thr, u))
        res.sort()
        best = res[0][0]
        print(f"PROBE {mb}MB best {best:.2f}us (net {best - empty:.2f}us = {(mb << 20) / (best - empty) / 1e6:.2f} TB/s) "
              + " ".join(f"{c}/{k}/{th}/{u}:{t:.2f}" for t, c, k, th, u in res[:6]), flush=True)
        # what the weights themselves cost read by torch (a reference point)
    w = weights["pooler"]
    print(f"PROBE pooler x@W.T m=1 {_time(lambda: torch.randn(1, 8192, device='cuda', dtype=torch.bfloat16) @ w.T, flush):.2f}", flush=True)
    return {name: weights[name] for name in weights}


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    if layer != "router":
        return x @ state[layer][:n].T
    return torch.nn.functional.linear(x.float(), state["router"])
