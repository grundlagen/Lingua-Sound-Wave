"""Offline tests for the Beta-Bernoulli bandit router — no models, deterministic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fugu_swarm.bandit import BanditRouter, BetaArm


def test_arm_update_and_mean():
    arm = BetaArm()
    assert arm.mean == 0.5 and arm.trials == 0
    for _ in range(3):
        arm.update(True)
    arm.update(False)
    # alpha=4, beta=2 -> mean 0.666..., 4 trials
    assert abs(arm.mean - 4 / 6) < 1e-9
    assert arm.trials == 4


def test_bandit_concentrates_on_better_worker():
    """A worker that ACCEPTs 90% of the time should win the ranking and get the
    majority of selections after learning — verified deterministically."""
    rng_seed = 7
    router = BanditRouter(["good", "bad"], seed=rng_seed)
    sim = __import__("random").Random(99)
    # warm-up: play each arm against its true success rate
    truth = {"good": 0.9, "bad": 0.1}
    for _ in range(400):
        w = router.select()
        router.update(w, sim.random() < truth[w])

    assert router.ranking()[0][0] == "good"
    # post-learning selection should favour 'good' heavily
    picks = [router.select() for _ in range(200)]
    assert picks.count("good") > picks.count("bad") * 3


def test_lazy_candidate_registration_and_subset():
    router = BanditRouter(["a", "b"], seed=1)
    # a per-query subset including an unseen worker registers it lazily
    choice = router.select(candidates=["b", "c"])
    assert choice in {"b", "c"}
    assert "c" in router.state()


def test_empty_candidates_raises():
    router = BanditRouter(["a"], seed=1)
    try:
        router.select(candidates=[])
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all bandit tests passed")
