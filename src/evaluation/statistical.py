"""
Statistical significance tests for benchmark comparisons.

Primary method: paired bootstrap test (Berg-Kirkpatrick et al., 2012).
Resample paired predictions N times, compute accuracy delta distribution,
report two-sided p-value.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class BootstrapTestResult:
    """Result of a paired bootstrap significance test."""
    system_a: str
    system_b: str
    acc_a: float
    acc_b: float
    delta: float           # acc_b − acc_a
    p_value: float
    significant: bool      # True if p < alpha
    alpha: float
    n_bootstrap: int
    ci_lower: float        # 95 % CI on delta
    ci_upper: float

    @property
    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.system_b} vs {self.system_a}: "
            f"Δacc = {self.delta:+.4f}  "
            f"p = {self.p_value:.4f}  ({sig} at α={self.alpha})"
        )


def paired_bootstrap_test(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    name_a: str = "system_A",
    name_b: str = "system_B",
) -> BootstrapTestResult:
    """Paired bootstrap test for two systems on the *same* examples.

    Parameters
    ----------
    correct_a, correct_b : sequence of 0/1
        Per-example correctness indicators for system A and B.
        Must have the same length and be aligned by example.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (default 0.05 for 95 %).
    seed : int
        RNG seed for reproducibility.
    name_a, name_b : str
        Labels used in the result object.

    Returns
    -------
    BootstrapTestResult

    Raises
    ------
    ValueError
        If input lengths differ or are empty.

    References
    ----------
    Berg-Kirkpatrick, Burber, & Klein (2012).
    "An Empirical Investigation of Statistical Significance in NLP."
    """
    a = list(correct_a)
    b = list(correct_b)
    if len(a) != len(b):
        raise ValueError(
            f"Paired vectors must have equal length, got {len(a)} vs {len(b)}"
        )
    n = len(a)
    if n == 0:
        raise ValueError("Cannot run bootstrap on empty vectors")

    acc_a = sum(a) / n
    acc_b = sum(b) / n
    observed_delta = acc_b - acc_a

    rng = random.Random(seed)
    count_ge = 0  # times resampled |delta| >= |observed_delta|

    deltas: List[float] = []
    for _ in range(n_bootstrap):
        indices = rng.choices(range(n), k=n)
        sa = sum(a[i] for i in indices) / n
        sb = sum(b[i] for i in indices) / n
        d = sb - sa
        deltas.append(d)
        # Two-sided test: count how often |resampled delta| >= |observed|
        if abs(d) >= abs(observed_delta):
            count_ge += 1

    p_value = count_ge / n_bootstrap

    # Confidence interval on delta
    deltas.sort()
    ci_lo = deltas[int(alpha / 2 * n_bootstrap)]
    ci_hi = deltas[int((1 - alpha / 2) * n_bootstrap)]

    result = BootstrapTestResult(
        system_a=name_a,
        system_b=name_b,
        acc_a=acc_a,
        acc_b=acc_b,
        delta=observed_delta,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
    )

    logger.info(result.summary)
    return result
