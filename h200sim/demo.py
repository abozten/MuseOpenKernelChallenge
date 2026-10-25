"""python3 -m h200sim.demo [--n 256] [--k 8192] [--unroll 8] [--trace] [--regs]"""

import argparse
import time

import numpy as np

from . import H200, Sim, to_bf16_bits, from_bf16_bits
from .kernels import gemm_tma_wgmma, gemv_bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--unroll", type=int, default=8)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--kernel", choices=["gemv", "gemm"], default="gemv",
                    help="gemv: CUDA-core GEMV (m=1); gemm: TMA + mbarrier + wgmma, m=64")
    ap.add_argument("--bn", type=int, default=16, help="gemm: output columns per CTA")
    ap.add_argument("--trace", action="store_true", help="print warp state transitions on SM 0")
    ap.add_argument("--regs", action="store_true", help="dump registers of one warp afterwards")
    args = ap.parse_args()

    sim = Sim(H200(), trace_sms=[0] if args.trace else [])
    if args.kernel == "gemm":
        return run_gemm(sim, args)
    rng = np.random.default_rng(0)
    W = from_bf16_bits(to_bf16_bits(rng.standard_normal((args.n, args.k), dtype=np.float32) / np.sqrt(args.k)))
    x = from_bf16_bits(to_bf16_bits(rng.standard_normal(args.k, dtype=np.float32)))
    pW, px, po = (sim.mem.alloc(W.size * 2), sim.mem.alloc(x.size * 2), sim.mem.alloc(args.n * 4))
    sim.mem.write(pW, to_bf16_bits(W))
    sim.mem.write(px, to_bf16_bits(x))

    prog = gemv_bf16(args.k, args.unroll, args.warps)
    print(f"program: {len(prog.instrs)} instrs, {prog.nregs} regs/thread")
    t0 = time.time()
    st = sim.launch(prog, args.n // args.warps, args.warps * 32, {"W": pW, "x": px, "out": po})
    print(f"(host wall time {time.time() - t0:.1f}s)")
    print(st.report(floor_bytes=W.size * 2))

    out = sim.mem.read(po, np.float32, args.n)
    ref = W.astype(np.float64) @ x.astype(np.float64)
    print(f"relative L2 error vs fp64: {np.linalg.norm(out - ref) / np.linalg.norm(ref):.2e}")

    if args.trace:
        print("\nfirst state transitions on SM 0 (cycle, smsp, cta, warp, from -> to, pc):")
        for e in sim.events[:25]:
            print(f"  {e[0]:>6}  smsp{e[2]} cta{e[3]} w{e[4]}  {e[5]} -> {e[6]}  pc={e[7]}")
    if args.regs:
        w = next(w for w in sim.retired if w.cta.sm.id == 0)
        print("\n" + sim.dump_regs(w, upto=16))


def run_gemm(sim, args):
    n, k, bn = args.n, args.k, args.bn
    rng = np.random.default_rng(0)
    x = from_bf16_bits(to_bf16_bits(rng.standard_normal((64, k), dtype=np.float32)))
    W = from_bf16_bits(to_bf16_bits(rng.standard_normal((n, k), dtype=np.float32) / np.sqrt(k)))
    px, pW, po = sim.mem.alloc(x.size * 2), sim.mem.alloc(W.size * 2), sim.mem.alloc(64 * n * 4)
    sim.mem.write(px, to_bf16_bits(x))
    sim.mem.write(pW, to_bf16_bits(W))
    ta = sim.tensor_map(px, 64, k, 2, 64, 64, swizzle=128)
    tb = sim.tensor_map(pW, n, k, 2, bn, 64, swizzle=128)
    prog, smem = gemm_tma_wgmma(n, k, bn, ta, tb)
    t0 = time.time()
    st = sim.launch(prog, n // bn, 128, {"out": po}, smem_bytes=smem)
    print(f"(host wall time {time.time() - t0:.1f}s)")
    print(st.report(floor_bytes=W.size * 2))
    out = sim.mem.read(po, np.float32, 64 * n).reshape(64, n)
    ref = x.astype(np.float64) @ W.astype(np.float64).T
    print(f"relative L2 error vs fp64: {np.linalg.norm(out - ref) / np.linalg.norm(ref):.2e}")


if __name__ == "__main__":
    main()
