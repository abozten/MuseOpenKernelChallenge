"""NVSHMEM / NCCL-style collectives over a Node. python3 -m h200sim.test_collectives"""

from . import H200, Node
from .collectives import all_gather, all_reduce_sum, ring_all_gather, ring_all_reduce_sum


def test_all_gather_is_correct_and_bandwidth_grows_with_size():
    small = all_gather(Node(2), 64 * 1024)
    large = all_gather(Node(2), 512 * 1024)
    assert small["correct"] and large["correct"]
    assert large["busbw_gbs"] > 2 * small["busbw_gbs"]                      # latency-bound when small
    cfg = H200()
    assert large["busbw_gbs"] <= cfg.nvlink_gbs_per_dir * cfg.nvlink_efficiency * 1.02     # never above the link


def test_all_gather_over_four_gpus_and_over_infiniband():
    nv = all_gather(Node(4), 128 * 1024)
    ib = all_gather(Node(4, hosts=[0, 0, 1, 1]), 128 * 1024)
    assert nv["correct"] and ib["correct"]
    assert ib["us"] > 1.5 * nv["us"]                                        # crossing hosts is slower


def test_all_reduce_sum_is_correct():
    r = all_reduce_sum(Node(2), 128 * 1024)
    assert r["correct"] and r["us"] > H200().nvlink_latency_ns / 1e3


def test_ring_algorithms_are_correct_and_scale_with_size():
    for fn in (ring_all_gather, ring_all_reduce_sum):
        small = fn(Node(4), 4 * 4 * 4096)
        large = fn(Node(4), 4 * 4 * 4096 * 8)
        assert small["correct"] and large["correct"], fn.__name__
        assert large["busbw_gbs"] > 2 * small["busbw_gbs"]                  # 2(n-1) dependent steps: latency-bound when small
        cfg = H200()
        assert large["busbw_gbs"] < cfg.nvlink_gbs_per_dir * cfg.nvlink_efficiency
    two = ring_all_reduce_sum(Node(2), 2 * 4 * 4096 * 2)
    assert two["correct"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
