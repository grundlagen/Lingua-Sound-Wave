"""fugu_swarm.bandit — Beta-Bernoulli Thompson-sampling worker selector.

A small, dependency-free adaptive router. Each worker slot keeps a Beta(a, b)
posterior over its probability of producing a verifier-ACCEPTed answer. Per
query we Thompson-sample every candidate and dispatch the best draw, then update
the chosen slot's posterior from the verifier verdict. Over time, traffic
concentrates on the workers that actually pass verification for *this* workload.

This is the same "learn routing from verdicts" idea as BicaMindLabs/fugue's
Beta-Bernoulli bandit, reimplemented from scratch (original code, Apache-2.0).
It complements OpenFugu's TRINITY head: the head picks a role/slot from a single
hidden state; the bandit biases slot choice by observed success. Wire it in by
overriding agent selection in the serve loop (see README "Wiring the bandit").

Offline-testable: no network, no models, deterministic under a seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BetaArm:
    """Beta(alpha, beta) posterior for one worker's ACCEPT probability."""
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def trials(self) -> int:
        # subtract the (1, 1) uniform prior
        return int(round(self.alpha + self.beta - 2.0))

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class BanditRouter:
    """Thompson-sampling selection over a fixed (extensible) set of workers."""

    def __init__(self, workers, seed: int | None = None, prior=(1.0, 1.0)):
        a0, b0 = prior
        self.arms: dict[str, BetaArm] = {w: BetaArm(a0, b0) for w in workers}
        self.rng = random.Random(seed)

    def select(self, candidates=None) -> str:
        """Return the worker with the highest Thompson draw.

        `candidates` restricts the choice to a per-query subset (the adaptive
        k-of-n / access-list case); unknown candidates are registered lazily.
        """
        pool = list(self.arms) if candidates is None else list(candidates)
        for w in pool:
            self.arms.setdefault(w, BetaArm())
        if not pool:
            raise ValueError("no candidates to select from")
        return max(pool, key=lambda w: self.arms[w].sample(self.rng))

    def update(self, worker: str, success: bool) -> None:
        self.arms.setdefault(worker, BetaArm()).update(bool(success))

    def ranking(self):
        """Workers sorted by posterior mean, best first."""
        return sorted(self.arms.items(), key=lambda kv: kv[1].mean, reverse=True)

    def state(self) -> dict:
        return {w: (a.alpha, a.beta) for w, a in self.arms.items()}
