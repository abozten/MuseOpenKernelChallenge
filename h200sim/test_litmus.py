"""python3 -m h200sim.test_litmus"""

from .litmus import ALLOWED, WITNESS, outcomes


def test_no_outcome_the_ptx_model_forbids():
    for (test, variant), allowed in ALLOWED.items():
        seen = outcomes(test, variant)
        assert set(seen) <= allowed, (test, variant, seen)


def test_unsynchronised_variants_show_their_weak_outcomes():
    for (test, variant), weak in WITNESS.items():
        assert weak in outcomes(test, variant, trials=48), (test, variant)


def test_synchronisation_removes_the_weak_outcomes():
    # the same randomised timing that exposes MP (1,0) and SB (0,0) never does once release/acquire or fence.sc is used
    assert (1, 0) not in outcomes("MP", "relacq", trials=48)
    assert (0, 0) not in outcomes("SB", "sc", trials=48)
    assert (0, 0) in outcomes("SB", "acq_rel", trials=48)       # fence.acq_rel orders but does not forbid store buffering


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
