"""H200 (GH100, SXM) machine description.

Structural numbers come from NVIDIA's Hopper whitepaper and tuning guide. Latencies and
throughputs are measurements from the literature (see README "Sources"). Most are from an
H800 PCIe (a GH100 part), and the core-pipeline details are from Ampere (RTX A6000), because
no public H200-specific microbenchmarks exist: treat them as calibration targets to refine
against your own profiles, not measurements of this chip.
"""

from dataclasses import dataclass


@dataclass
class H200:
    # --- structure ---
    num_sms: int = 132
    smsp_per_sm: int = 4
    warp_size: int = 32
    regs_per_sm: int = 65536          # 32-bit registers (256 KB)
    max_regs_per_thread: int = 255
    reg_alloc_warp_unit: int = 256    # registers are granted to a warp in chunks of 256
    max_warps_per_sm: int = 64
    max_ctas_per_sm: int = 32
    smem_per_sm: int = 228 * 1024
    smem_per_cta_max: int = 227 * 1024
    smem_banks: int = 32

    # --- clocks and bandwidth ---
    clock_ghz: float = 1.98           # H200 SXM boost; base 1.665, floor 0.345 under heavy throttling (power-capping study)
    clock_base_ghz: float = 1.665
    clock_min_ghz: float = 0.345
    hbm_tbs: float = 4.8
    hbm_efficiency: float = 0.885     # fraction of peak a streaming read reaches: 4.25 of 4.8 TB/s measured on the judge H200 (cold L2, 6-64 MB)
    l2_bytes: int = 50 * 1024 * 1024
    l2_bytes_per_cycle: float = 4000  # aggregate L2 -> SM fabric (measured 3942-4472 B/clk)
    sm_port_bytes_per_cycle: float = 64
    l1_smem_unified: int = 256 * 1024         # L1 + shared memory per SM
    l1_max_bytes: int = 208 * 1024            # largest L1 seen ("L1-preferred" setting)
    smem_carveouts: tuple = tuple(k * 1024 for k in (0, 8, 16, 32, 64, 100, 132, 164, 196, 228))
    l1_reserved: int = 1024                   # per-CTA shared-memory system reservation
    lsu_sectors_per_cycle: int = 4    # 128 B/cycle through the L1/LSU

    # --- latencies (cycles) ---
    lat_l1: int = 32                  # L1 hit, 32-bit load (+2 for 64-bit, +6 for 128-bit)
    lat_l2: int = 295                 # L2 hit in the SM's own NUMA node (H100 SXM5, DGNA)
    lat_hbm: int = 554                # DRAM, local NUMA node
    lat_smem: int = 24                # shared load RAW (+2 for 128-bit) before bank-conflict waves
    lat_l2_far_extra: int = 175       # remote NUMA node L2 hit: 470 - 295
    lat_hbm_far_extra: int = 174      # remote NUMA node DRAM: 728 - 554
    lat_subnuma_extra: int = 32       # other sub-NUMA node within the same L2 half (H100 only)
    dram_burst: int = 64              # DRAM -> L2 transaction size (default setting)
    lat_load_width_extra: tuple = ((4, 0), (8, 2), (16, 6))   # L1-hit RAW by access width in bytes

    # --- GPCs / clusters / async engines ---
    # SMs per GPC; sub-NUMA nodes hold 32 / 34 / 34 / 32 SMs = GPC pairs (Liu et al., H100 SXM5)
    gpc_sizes: tuple = (16, 16, 17, 17, 17, 17, 16, 16)
    lat_dsmem: int = 181              # SM-to-SM distributed shared memory access (cluster size 2)
    lat_dsmem_local: int = 33         # DSMEM access that resolves to the CTA's own shared memory
    # SM-to-SM bytes/cycle/SM on a ring pattern: 3.28 TB/s over 114 SMs at cluster size 2, 2.78 at 4;
    # 8 and 16 are extrapolated ("throughput falls as cluster size grows").
    dsmem_bytes_per_cycle: tuple = ((2, 16.4), (4, 13.9), (8, 11.0), (16, 8.0))
    tma_issue_lat: int = 170          # TMA access latency is ~170 cycles above a regular access
    tensor_flops_per_smsp_cycle: int = 1024   # dense bf16: 989 TFLOP/s = 132 SMs x 4096 FLOP/clk at 1.83 GHz (datasheet clock, not 1.98)
    wgmma_lat: int = 30
    smem_bytes_per_cycle: int = 128   # shared memory port: wgmma operand reads and TMA writes contend here
    fp64_tensor_flops_per_smsp_cycle: int = 64   # 67 TFLOPS FP64 tensor (whitepaper) = 2x the FP64 vector rate
    tensor_data_random: bool = True   # wgmma on non-zero random data runs at a lower, power-limited clock
    # random-data / zero-data wgmma throughput ratio by (input dtype, accumulator) at m64n256 (H800)
    tensor_derate: tuple = (("f16", "f16", 0.966), ("f16", "f32", 0.913), ("bf16", "f32", 0.913),
                            ("tf32", "f32", 0.980), ("e4m3", "f16", 0.994), ("e4m3", "f32", 0.979),
                            ("e5m2", "f16", 0.994), ("e5m2", "f32", 0.979), ("s8", "s32", 0.995),
                            ("u8", "s32", 0.995))               # tensor pipe drain after the last FLOP of a wgmma

    # --- L2 organisation ---
    l2_ways: int = 16
    l2_line: int = 128                # tag granularity; data is valid per 32 B sector
    l2_cross_bytes_per_cycle: float = 1600   # bandwidth between the two L2 partitions
    hbm_channels: int = 48            # 256 B interleave across channels

    # --- sub-core memory pipeline (Ampere core study; assumed unchanged on Hopper) ---
    mem_queue_depth: int = 5          # memory instructions a sub-core can have in flight before stalling
    smsp_mem_addr_cycles: int = 4     # per sub-core address calculation: one memory instruction / 4 cycles
    sm_mem_accept_cycles: int = 2     # shared structures accept one memory instruction / 2 cycles
    # cycles until a memory instruction's source registers are free again (WAR latency), regular regs
    war_ldg: int = 11
    war_stg: tuple = ((4, 14), (8, 16), (16, 20))
    war_lds: int = 9
    war_sts: tuple = ((4, 12), (8, 14), (16, 18))
    war_ldgsts: int = 13
    war_ldg_uniform: int = 9          # a uniform-register address is read sooner (and RAW is 3 cycles shorter)
    war_stg_uniform: tuple = ((4, 10), (8, 12), (16, 16))
    war_sts_uniform: tuple = ((4, 10), (8, 12), (16, 16))

    # --- register file (Ampere core study): 2 banks per sub-core, 1 read port each ---
    model_regfile: bool = True
    rf_banks: int = 2
    # --- constant caches: LDC hits an L0 constant cache, a miss costs 79 more cycles (L1.5) ---
    model_ccache: bool = True
    lat_ldc: int = 29
    lat_ldc_miss: int = 79
    # --- instruction fetch: 16 B instructions, per-sub-core L0 I-cache with a stream-buffer
    # prefetcher (16 lines), shared L1 I-cache. Sizes are assumptions (Volta-era figures); the
    # measured policy is "stream buffer of 16". The cold-code latency is real: the judge flushes L2.
    model_icache: bool = True
    instr_bytes: int = 16
    icache_line: int = 128
    l0i_bytes: int = 16 * 1024
    l1i_bytes: int = 128 * 1024
    lat_l1i: int = 35
    ifetch_stream_lines: int = 16
    ifetch_line_cycles: int = 4

    # --- address translation. Structure from the full TLB reverse-engineering of Ampere (Zhang et al., CCS 2023,
    # A100): a 16-entry fully-associative LRU L1 TLB shared by the two SMs of a TPC; an L2 TLB per GPC with 2048
    # entries (256 sets x 8 ways) of 16 sub-entries each, so one entry reaches 16 consecutive pages; and one L3
    # TLB shared by every SM (2 slices x 256 sets x 8 ways = 4096 entries x 16 sub-entries). A hit in a lower level
    # is reinserted into the levels above. Hopper's TLBs have not been published: the geometry is assumed to match.
    # Latencies are ASSUMPTIONS (a Turing TLB-miss latency of 472 cycles is the only measurement found).
    model_tlb: bool = True
    tlb_page_shift: int = 21            # 2 MB pages (the GPU also supports 4 KB and 64 KB)
    tlb_l1_entries: int = 16            # per TPC
    tlb_l2_entries: int = 2048          # per GPC
    tlb_l3_entries: int = 4096          # shared
    tlb_subs: int = 16
    tlb_l2_hit_cycles: int = 90
    tlb_l3_hit_cycles: int = 200
    tlb_walk_cycles: int = 472
    tlb_walkers: int = 16

    # --- memory ordering ---
    model_visibility: bool = True     # stores reach other SMs only when they land in L2 (so fences matter)
    model_l1_staleness: bool = True   # L1 is not coherent: ld.ca can return a stale snapshot of a sector
    store_drain_cycles: int = 100     # ASSUMPTION: a store waits in the SM's write path (store buffer) before it heads to L2
    model_load_read_time: bool = True # a global load reads memory where its request lands (about half its latency), not at issue
    strict_proxy_fence: bool = False  # raise (instead of record) async-proxy reads of un-fenced generic writes

    # --- NVLink to a peer GPU (memory mapped with Sim.map_peer). NVLink 4 is 900 GB/s bidirectional = 450 GB/s
    # per direction; DeepEP reaches 153-158 GB/s of the 200 GB/s per direction an H800 has (~80 %), the
    # efficiency applied here. The one-way latency is an ASSUMPTION (no credible published figure was found).
    nvlink_gbs_per_dir: float = 450.0   # 18 links x 25 GB/s each way (NVLink 4: 900 GB/s bidirectional)
    nvlink_efficiency: float = 0.8
    nvlink_latency_ns: float = 1000.0
    # InfiniBand between hosts: one ConnectX-7 400 Gb/s (50 GB/s) per GPU; DeepEP's normal kernels reach 43-58 GB/s.
    # The latency is an ASSUMPTION. NVSwitch is modeled as non-blocking (64 NVLink-4 ports, 13.6 Tbit/s per chip).
    ib_gbs: float = 50.0
    ib_efficiency: float = 0.9
    ib_latency_ns: float = 8350.0       # GPU-initiated put-with-signal: 16.7 us ping-pong round trip on H100 (NCCL GIN, EOS)
    # Switch fabric. NVSwitch gen 3: 64 NVLink-4 ports, 13.6 Tbit/s (1.7 TB/s) per chip, four chips per HGX node.
    # InfiniBand: a two-level fat tree whose first level is 2:1 oversubscribed (DGX SuperPOD, per the Hopper article).
    nvswitch_chips: int = 4
    nvswitch_tbps: float = 1.7
    ib_oversubscription: float = 2.0

    # --- ECC (HBM3e, on by default on H200). Behaviour: single-bit errors are corrected, double-bit errors
    # are detected and abort the kernel. The correction cost is an ASSUMPTION; there is no capacity overhead.
    ecc_enabled: bool = True
    ecc_correct_cycles: int = 150

    # --- power. Calibrated on the H200 power-capping study (arXiv 2604.11391): TDP 700 W; the memory clock is
    # fixed so HBM3e power does not scale with the SM clock and reaches ~220 W at full bandwidth; a streaming
    # kernel needs a cap of >= 550 W to keep full bandwidth, so SM + L2 + NoC dynamic power at boost is ~300 W
    # at 4.4 TB/s; under a 200 W cap the SM clock falls to its 345 MHz floor and the board still draws ~250 W
    # (static + memory + dynamic at (345/1980)^3). The issue and tensor coefficients are ASSUMPTIONS (tensor
    # power depends on the operand data: up to 38 % in the literature, and random data reaches the limit).
    power_cap_w: float = 700.0
    power_static_w: float = 30.0
    power_mem_w: float = 220.0          # HBM3e at full bandwidth, independent of the SM clock
    power_l2_w: float = 540.0           # SM/L2/NoC dynamic per unit of L2 utilisation, at the boost clock
    power_issue_w: float = 110.0        # all sub-cores issuing every cycle (assumed)
    power_tensor_zero_w: float = 150.0  # tensor pipes saturated on all-zero data (assumed)
    power_tensor_random_w: float = 520.0  # ... on random data (assumed)
    tensor_data_toggle: float = 1.0     # 0 = constant data, 1 = random data (bit-flip rate of the operands)

    # --- atomics: extra latency over an uncontended one by same-address lanes (Jia et al., V100, Table 4.2)
    atomic_shared_curve: tuple = ((1, 6), (2, 7), (4, 11), (8, 18), (16, 24), (32, 66))
    atomic_global_curve: tuple = ((1, 36), (2, 31), (4, 32), (8, 41), (16, 58), (32, 76))

    # --- kernel launch (Hazy Research, H100): 2.1 us between dependent kernels in a stream, 1.3 us in a
    # CUDA graph. With programmatic dependent launch the secondary grid is already staged; its gap is assumed.
    launch_gap_us: tuple = (("stream", 2.1), ("graph", 1.3))
    pdl_gap_us: float = 0.3

    # --- simulated memory ---
    mem_bytes: int = 1 << 28
    va_base: int = 0                  # device pointers start here; use >= 1<<32 with 64-bit (wide) ops

    @property
    def hbm_bytes_per_cycle(self) -> float:
        return self.hbm_tbs * 1e12 * self.hbm_efficiency / (self.clock_ghz * 1e9)
