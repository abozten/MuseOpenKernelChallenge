# h200sim

Cycle-driven, GPGPU-Sim-style model of an H200 (GH100): 132 SMs x 4 sub-partitions, a warp state
machine, real per-thread register contents, caches, TLBs, HBM, NVLink and InfiniBand. Kernels are written in
a SASS-like Python ISA (`isa.Asm`), run for real (results are checked against numpy), and timed in cycles.
A `Node` simulates several GPUs, each a full simulator, over shared memory.

```
cd MuseOpenKernelChallenge
h200sim/.venv/bin/python -m h200sim.test_all                              # 79 tests
h200sim/.venv/bin/python -m h200sim.demo --n 512 --trace --regs           # CUDA-core GEMV, state trace, reg dump
h200sim/.venv/bin/python -m h200sim.demo --kernel gemm --n 256 --k 1024 --bn 32   # TMA + mbarrier + wgmma
```

## What is modeled

| Area | Model |
|---|---|
| Issue | 4 sub-partitions per SM. The measured **greedy-then-youngest** policy; one instruction / cycle (Volta onward does not dual-issue); pipe occupancy for int, fp32, fp64, xu (shuffles / MUFU / conversions), tensor, TMA, uniform |
| Warp states | READY, STALL_DEP, STALL_MEM, STALL_PIPE, STALL_ASYNC, STALL_FETCH, BARRIER, EXITED, with timestamped transitions (`Sim(trace_sms=[0])`) and an issue log (`sim.log_issue`) |
| Dependence tracking | `Sim(dep_mode="scoreboard")` (default) or `"control"`: the real ISA's compiler-set **stall counts, six dependence counters, wait masks and the Yield bit** (`control.annotate` plays ptxas); shared counters create false waits |
| Registers | 32 lanes x 32 bit; 64 uniform registers and 7 predicates per warp; allocation limits per SM; `setmaxnreg` moves registers between warpgroups through a per-CTA pool; a warp cannot exit while loads are returning |
| Register file | 2 banks per sub-core, one read port each: same-bank operands cost issue bubbles; compiler-managed register-file cache with reuse bits (`a.reuse(...)`) |
| SIMT | active masks, reconvergence stack, `@P` / `@!P` guards (`with a.guard(p)`), divergent branches with a `reconv` label, lanes may `exit` early, votes / elect / activemask |
| Front end | 16 B instructions, per-sub-core L0 I-cache + 16-line stream-buffer prefetcher, per-SM L1 I-cache, code cached in L2 (a flushed L2 means a DRAM fetch); LDC through L0 / SM / L2 constant caches (miss +79 cycles); 64 KB constant bank (`ldc_mem`, divergent addresses serialize) |
| Memory pipeline | per-sub-core queue of 5, one memory instruction / 4 cycles per sub-core, one / 2 cycles into the shared stage, 128 B/cycle L1; WAR / RAW latencies by access width; uniform-register addresses are faster. Reproduces the measured issue-cycle table |
| Shared memory | 32 banks, conflict waves, 128-bit accesses per quarter-warp, `ldmatrix` / `stmatrix` (x1/x2/x4, `.trans`) |
| L1 / shared | 256 KB unified; shared-memory carve-out steps; L1 = min(208 KB, 256 KB - carve-out - reserve); the L1 is **not coherent** |
| L2 | 16-way set-associative, 128 B lines with 32 B sectors; **two NUMA nodes and four sub-NUMA nodes** (GPC pairs, 32/34/34/32 SMs) with the measured address mapping (64 KB pattern at 4 KB granularity), local / remote latencies, a crossing link; 64 B DRAM bursts; dirty-sector write-back |
| HBM | 48 channels, 256 B interleave, 4.8 TB/s x 0.92; `ldg(..., cache="cs", l2pf=256)` is `ld.global.nc.L1::no_allocate.L2::256B`; ECC: single-bit errors corrected (costing cycles), double-bit `EccError` (XID 48), silent corruption with ECC off (`sim.inject_ecc_error`) |
| Address translation | the Ampere hierarchy from the TLB reverse-engineering: 16-entry fully-associative L1 TLB per TPC, L2 TLB per GPC (2048 entries x 16 sub-entries), shared L3 TLB (4096 x 16), reinsertion on hits, 16 page walkers; 2 MB pages |
| Atomics | global and shared, add / min / max / and / or / xor / exch / cas, u32 / s32 / f32, return old values in lane order; latency vs same-address lanes from the Volta measurements |
| cp.async | `ldgsts` + commit / wait groups; data lands through a timed event, so an early read sees stale data |
| mbarrier | init / arrive / arrive_tx / wait(parity) with transaction counting, remote arrive in a cluster; named barriers (`bar id, count`, `bar_arrive`) |
| TMA | tensor maps of rank 1-5 (zero fill, clipped stores, full box counts toward tx), **im2col mode and the `::w`, `::w::128` and `no_offs::w` variants** (wHalo, wOffset), 32/64/128 B swizzle, fp32 reduce-add stores, prefetch to L2, multicast to a cluster mask, 1D bulk copies; ~170 cycles above a regular access |
| mma.sync | m16n8kK for f16, bf16, tf32, e4m3, e5m2, s8, u8 with f32 / f16 / s32 accumulate; PTX fragment layouts; measured latency and ~65 % of wgmma peak; **`mma.sp`** structured-sparse (2:4, and 1:2 for tf32) with the PTX selector rules and metadata encoding |
| wgmma | m64nNkK for the same types; A from shared memory or registers; B from shared memory; K-major or MN-major (16-bit); swizzle; **structured-sparse** (register metadata + selector, per the PTX ISA); async commit / wait groups; latency = max(N/2, floor) from the measured tables; power-limited derating for random data; shared-memory port contention |
| FP8 numerics | tensor-core FP8 accumulation keeps ~13 fraction bits after aligning products and a 13-bit-mantissa accumulator (the DeepSeek finding) |
| FP64 | `DFMA` / `DADD` / `DMUL` on register pairs (1 per 2 cycles per sub-core), FP64 tensor core `DMMA m8n8k4` |
| Scalar ISA | packed f16x2 / bf16x2 math, conversions (bf16, f16, e4m3, e5m2), MUFU, min / max, DPX (`VIMNMX3`, `VIADDMNMX` incl. s16x2 + relu), PRMT, LOP3, POPC, BREV, FLO, shuffles, redux |
| Clusters | placed on distinct SMs of one GPC (16/16/17/17/17/17/16/16), cluster barrier, distributed shared memory with latency 181 / 33 (local) and bandwidth that falls with cluster size |
| Launch chains | `sim.launch_chain([...])`: one `stream` runs in order with a launch gap (2.1 us stream / 1.3 us graph); different streams run **concurrently** and share the SMs; **programmatic dependent launch** (`pdl=True`, `griddep_launch`, `griddep_wait`) |
| Multi-GPU | `Node(n_gpus, hosts=[...])`: every GPU is a full `Sim` with its own SMs, caches and HBM over one shared array, advanced in global time order. Peer loads, stores, atomics and TMA copies cross **NVLink through four NVSwitch chips** (each GPU's egress and ingress ports and each chip's queue are shared by all flows) inside a host and an **oversubscribed InfiniBand** fabric between hosts; `sim.map_peer(n)` gives a single GPU a peer buffer |
| Collectives | `collectives.py`: NVSHMEM-style direct-push all-gather / all-reduce, and NCCL's ring all-gather / all-reduce (channels, per-step flags, remote `red.add`), as ordinary kernels over a `Node`; reports NCCL-style bus bandwidth |
| Memory ordering | a load reads memory where its request lands (about half an L2 hit after issue, later after a TLB miss), a store reaches L2 after the SM's write path and a one-way trip (remote-NUMA stores later), so other SMs see it late; same-SM forwarding; `fence.sc` stalls until the warp's stores are visible, `fence.acq_rel` only orders (later stores become visible no earlier, later accesses wait for earlier loads) so it allows store buffering as PTX does; `ld.acquire` (bypasses and invalidates this SM's L1, orders later accesses), `ld.relaxed`, `st.release`; shared-memory writes need `fence.proxy.async` before TMA stores / wgmma read them (`sim.hazards`, or raised with `strict_proxy_fence`). Checked by **litmus tests** (MP, SB, CoRR, LB) against the outcomes the PTX model allows |
| Power / DVFS | board power from L2/NoC, issue, tensor (data-dependent: `tensor_data_toggle`) and HBM activity; a lower `power_cap_w` lowers the SM clock down to the 345 MHz floor (`clock_scale`, `time_us_dvfs`); calibrated on the H200 power-capping study |
| Addressing | 32-bit pointers, or 64-bit register pairs (`ldc64`, `iadd64`, `imad_wide`, `wide=True`) with `H200(va_base=1 << 40)` |

## Where the numbers come from

| Quantity | Value | Source |
|---|---|---|
| L1 hit / shared (cycles) | 32 / 24-29 | Luo et al. 2025 (H800 PCIe), Tables 3-4 |
| L2 hit local / remote NUMA, DRAM local / remote (cycles) | 295 / 470, 554 / 728; sub-NUMA +32; 66 SMs per NUMA node; 64 B DRAM bursts; address-to-NUMA mapping | Liu, Chen, Carlson, *DGNA* (H100 SXM5) |
| H200 TDP 700 W; boost 1980 MHz, base 1665 MHz, floor 345 MHz; HBM3e power up to ~220 W (H100 ~120 W); a streaming kernel needs a >= 550 W cap; ~250 W drawn under a 200 W cap | | Power-capping study of H100 and H200 (arXiv 2604.11391) |
| TLB hierarchy: L1 16-way fully-associative per TPC, L2 per GPC, L3 shared; entries, sub-entries, sets, reinsertion, 2 MB pages | A30 / A100 | Zhang et al., *TunneLs for Bootlegging* (CCS 2023) |
| wgmma shared-memory wavefronts (80 / 48 / 32 / 24 / 20 / 18 for N = 256..8, SS) and Tensor Core utilisation (97 % for N >= 64, 65 / 39 / 21 % for N = 32 / 16 / 8 in SS) | H100 PCIe, zero data | A Chinese Nsight Compute analysis on Zhihu ("NVIDIA Hopper架构TensorCore分析(4)"), read through a reader proxy |
| FFMA / IADD3 / LOP3 / ISETP 4 cycles, IMAD 5, HFMA2 6, FP64 8, POPC 10, MUFU / FLO / BREV 14, conversions 14 (Pascal); atomic latency vs same-address lanes (shared 6..66, global 36..76) | V100 | Jia et al., Volta dissection, Tables 4.1-4.2 |
| GPU-initiated put-with-signal ping-pong latency 16.7 us round trip (InfiniBand, NCCL GIN / NVSHMEM 16-24 us); DeepEP RDMA 53-85 GB/s | H100 (EOS) | NCCL GIN paper (arXiv 2511.15076) |
| NVLink 4: 18 lanes, 25 GB/s per lane per direction, 900 GB/s bidirectional; NVSwitch with 64 NVLink-4 ports and 13.6 Tbit/s; SuperPOD two-level fat tree, level 1 2:1 oversubscribed | | Hopper architecture articles (Chinese: cnblogs, Zhihu) |
| NVLink effective bandwidth ~160 GB/s of 200 GB/s per direction (153 dispatch / 158 combine), 20-24 SMs used for communication; InfiniBand 43-58 GB/s per ConnectX-7 | H800 | DeepEP, as reported in a Chinese write-up (pinghaoyang.com.cn) and the DeepEP README |
| L1 and shared throughput, L2 throughput, global memory throughput (91 % of peak) | 128 B/clk/SM, ~4000 B/clk | Luo et al. (Table 5), RRZE gpu-benches |
| TMA ~170 cycles above a regular access; DSMEM latency 181 (local 33), ring throughput 3.28 TB/s (cluster 2) | | Luo et al. 2024 / 2025 |
| mma.sync and wgmma latency / throughput per type, shape, N, dense / sparse, SS / RS; random-data derating | Tables VII-X | Luo et al. 2024 / 2025 |
| Issue policy, control bits (Stall, Yield, six dependence counters), register file banks / ports / cache, memory queue, per-width latencies, constant-cache miss penalty, stream-buffer prefetcher | | Huerta et al., *Analyzing Modern NVIDIA GPU cores* (**Ampere**) |
| FP8 tensor-core accumulation precision | ~13-14 fraction bits | DeepSeek-V3 technical report, Colfax write-up |
| mma.sp and sparse wgmma: 2:4 (1:2 for tf32), valid index nibbles, who supplies the metadata for each shape (one thread / a pair / all four), the compressed-A fragment mapping; each warp supplies the metadata of its 16 rows | | PTX ISA 8.7 (read from the PDF); the community note agrees on the high-half rule |
| im2col and im2col::w semantics (bounding box, pixels per column, channels per pixel, offsets, traversal stride, wHalo, wOffset, zero fill) | | PTX ISA 8.7, sections 5.5.4-5.5.5 |
| `membar` is a synonym for `fence.sc` on sm_70 and later | | PTX ISA 8.7, 9.7.13.4 |
| Launch overhead 2.1 us (stream) / 1.3 us (graph) on H100 | | Hazy Research |
| Tensor power depends on the operand data; random data reaches the power limit | | Luo et al., Patel et al. (SC24 workshop), FAU blog |
| Litmus-test outcomes allowed by the PTX memory model | | Lustig, Sahasrabuddhe, Giroux (ASPLOS 2019) |
| 132 SMs, 4 sub-partitions, 64K registers, 228 KB shared memory, 4.8 TB/s, 989 TFLOPS | | NVIDIA H200 / Hopper documentation |

Sources:
[Luo et al., 2024](https://arxiv.org/abs/2402.13499) ·
[Luo et al., 2025](https://arxiv.org/abs/2501.12084) ·
[Huerta et al., 2025](https://arxiv.org/abs/2503.20481) ·
[Jia et al., Volta](https://arxiv.org/abs/1804.06826) ·
[Liu, Chen, Carlson: DGNA](https://arxiv.org/abs/2607.19922) ·
[H100 / H200 power capping](https://arxiv.org/abs/2604.11391) ·
[Zhang et al., GPU TLBs (CCS 2023)](https://casrl.ece.ucf.edu/wp-content/uploads/2023/05/2023-ccs.pdf) ·
[NCCL GPU-initiated networking](https://arxiv.org/abs/2511.15076) ·
[Chips and Cheese on H100](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth) ·
[DeepSeek-V3](https://arxiv.org/abs/2412.19437) ·
[Colfax on FP8 accumulation](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/) ·
[PTX ISA 8.7 (PDF)](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf) ·
[Hazy Research on launch overhead and PDL](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) ·
[Zhihu: Hopper TensorCore analysis (4), Chinese](https://zhuanlan.zhihu.com/p/654067822) ·
[Zhihu: Hopper architecture, Chinese](https://zhuanlan.zhihu.com/p/21111811606) ·
[Zhihu: GPU reverse-engineering survey, Chinese](https://zhuanlan.zhihu.com/p/2014778262367250116) ·
[DeepEP walkthrough (Chinese)](https://pinghaoyang.com.cn/aigc/posts/deepep/) ·
[Hopper architecture article (Chinese, cnblogs)](https://www.cnblogs.com/sexintercourse/p/18555048) ·
[pantheonsim notes on mma.sp and im2col](https://github.com/pantheongpu/pantheonsim/pull/162) (community, hardware-checked on an RTX 3060) ·
[Patel et al., input-dependent GPU power](https://homes.cs.washington.edu/~patelp1/papers/inputpower-sssc24.pdf) ·
[FAU: power vs data initialization](https://blogs.fau.de/blog/2025/11/power-frequency-characteristics-of-compute-bound-codes-on-a40-a100-and-h100-gpu-with-different-data-initialization/)

The tests check the model against these tables: `test_calibration` (queue issue cycles, pointer-chase latencies
local / remote, prefetch bursts), `test_tensor` (wgmma cycles and wavefronts, the Chinese Nsight data, mma fraction of
peak), `test_system` (the H200 power study's knee and 250 W point, NVLink / InfiniBand / switch queues, TLB levels),
`test_litmus` (PTX-model outcomes), `test_collectives` (correct results, bandwidth below the link rate).

## Caveats: what is assumed, not measured

- **No public H200 microbenchmark of the core, caches or interconnect exists.** The one H200-specific dataset is the
  power-capping study. L1 / shared / TMA / DSMEM / tensor numbers are from an H800 PCIe; L2 / DRAM latencies and the NUMA
  topology from an H100 SXM5 (the same GH100 die and 132 SMs); instruction latencies from Volta / Pascal; the core
  pipeline from Ampere; the TLB hierarchy from Ampere (A30 / A100). Expect the real H200 to differ.
- **Still assumed** (no measurement found): DPX and DMMA latencies; MUFU / conversion / DPX issue intervals; FP8
  `mma.sync` timing (taken equal to INT8); instruction-cache sizes; DSMEM bandwidth for clusters of 8 and 16
  (extrapolated); TLB hit and walk latencies (472 cycles is a Turing measurement); ECC correction latency; NVLink
  latency (1 us; a secondary source also says ~1 us for NVLink 4); the store-buffer drain (100 cycles) and one-way store
  latency; NVSwitch hop latency (0); the PDL staged-secondary gap (0.3 us). The InfiniBand latency is a measurement
  (8.35 us one way, from a put-with-signal ping-pong) but includes software, so it overstates a raw RDMA write.
- **Power**: the structure and four anchors are sourced (700 W TDP, the 550 W streaming knee, ~250 W at a 200 W cap, the
  345 MHz floor); the issue and tensor coefficients are assumptions. `time_us` is unscaled; `time_us_dvfs` applies the
  cap. The tensor derating from the Hopper papers is a separate, measured effect and is not combined with it.
- **Sparse metadata**: the encoding, selector rules and fragment mapping are the spec's. Where inside a lane's 32-bit
  word each chunk's nibble sits is drawn only in the spec's figures, which could not be read as text; the natural order
  (row g in the low half, row g+8 in the high half, chunks ascending) is used, so a kernel written against real
  hardware metadata could differ in bit placement.
- **Control bits**: the semantics (stall counter, six counters, wait masks, Yield) are the measured ones; `annotate`
  is a plausible ptxas, not the real one (conservative at branch targets, no stall-count cap of 15, Yield only when you
  set it).
- **Memory model**: operational and checked against the PTX model by litmus tests (MP, SB, CoRR, LB), not axiomatic. It
  is stronger than PTX in one place: load buffering's (1,1) is never produced, because a load reads before a
  program-later store of another thread can become visible. Those litmus tests also caught two real modeling bugs
  (TLB penalty not part of the fill time; a queue head-of-line effect).
- **Collectives**: direct-push all-gather / all-reduce and NCCL's **ring** algorithm (channels = CTAs, per-step flags, remote
  `red.add` reduce-scatter then all-gather) are kernels over a `Node`. They reproduce the algorithms' structure and their
  latency-bound-to-bandwidth-bound behaviour, but not NCCL's tuned kernels (LL / LL128 protocols, tree algorithm, SM-count
  and channel tuning), so their bus bandwidth (10-180 GB/s at the sizes simulated) is well below what NCCL reaches; NVSHMEM /
  NCCL as libraries (proxy threads) are not modeled. NVSwitch is a set of queues without a hop latency.
- **FP8 accumulation** alignment is applied over one whole instruction's K; the real granularity is unpublished.
- **PTX ISA**: the HTML copy the fetch tool returns is truncated, so the PDF was downloaded and read as text; its figures
  (bit-level layouts) are not text.
- **More Chinese pages read** (through the proxy): a CSDN reading note on the Hopper benchmarking paper (no numbers), a Zhihu
  cycle-level WGMMA pipeline article (self-described personal speculation; it reproduces the N=64 -> 32 and N=32 -> 24
  cycle figures already used), and a Zhihu Tensor Core overview (several claims there conflict with measurements: two
  instructions per scheduler per cycle contradicts the single-issue measurement, and FP32 accumulation for FP8
  contradicts DeepSeek's finding, so neither is used).
- **Chinese-language sources**: Zhihu answers automated fetches with HTTP 403, but the pages can be read through a
  reader proxy, which is how the Hopper TensorCore analysis, the Hopper architecture article and the reverse-engineering
  survey were read. The Nsight Compute data in the first is used and tested; the second and third gave NVLink / NVSwitch /
  SuperPOD figures and a survey of other work (several of whose numbers are Turing / Volta). The CSDN page that returned 521 directly opens through the proxy.
  The Zhihu authors give no latencies and do not separate measured from whitepaper numbers.

## Not modeled

- TMA **gather4** (Blackwell only, not on Hopper), interleaved tensor layouts, `wgmma` with transposed 8-bit / tf32
  operands (the hardware does not allow it either).
- Switch-level hop latency, GPUDirect RDMA proxies, multi-node routing beyond two classes of link, NCCL's tuned algorithms.
- The axiomatic C++ / PTX memory model beyond the litmus patterns tested (IRIW, scoped causality); L2 compression and
  persistence controls; HBM refresh and bank-level timing.
- Nested guards; the hardware limit on reconvergence depth. (Dual issue does not exist on Volta and later.)
