"""State ensembles for the single-copy vs collective scaling study.

Both ensembles are depolarized low/high-rank states written in a *factored* form
``rho = (1 - q) * P + q * I / 2^n`` where ``P = G G^dag`` is a trace-1 component
(``G`` is ``2^n x R``).  Keeping ``G`` factored (rather than the dense ``2^n x
2^n`` matrix) lets shadow sampling scale to large ``n`` — the sampling only needs
``U G`` (``O(R n 2^n)``), never a ``2^n x 2^n`` rotation.

* ``noisy_pure(n, q, rng)``  — ``|psi>`` Haar-random pure (``R = 1``); a NISQ-like
  noisy pure state whose purity stays O(1).
* ``random_mixed(n, q, rng)`` — Ginibre high-rank component; the old highly-mixed
  ensemble whose purity collapses toward ``2^{-n}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoisyState:
    """``rho = (1 - q) * G G^dag + q * I / 2^n`` with ``G`` (``2^n x R``) trace-1."""

    components: np.ndarray  # (2^n, R), already normalized so Tr(G G^dag) = 1
    q: float
    n: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.q <= 1.0:
            raise ValueError(f"depolarizing weight q must be in [0, 1], got {self.q}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.components.shape != (2 ** self.n, self.components.shape[1]) or self.components.shape[1] < 1:
            raise ValueError(
                f"components must be (2^n, R>=1) = (2^{self.n}, R); got {self.components.shape}"
            )
        # Immutable data: block in-place mutation (frozen=True only stops rebinding).
        self.components.setflags(write=False)

    @property
    def dim(self) -> int:
        return 2 ** self.n

    def purity(self) -> float:
        """Exact ``Tr(rho^2) = (1-q)^2 Tr(P^2) + 2(1-q)q/2^n + q^2/2^n``."""
        gram = self.components.conj().T @ self.components  # (R, R) = G^dag G
        tr_p2 = float((np.abs(gram) ** 2).sum().real)  # Tr((GG^dag)^2) = ||G^dag G||_F^2
        d = self.dim
        return (1.0 - self.q) ** 2 * tr_p2 + 2.0 * (1.0 - self.q) * self.q / d + self.q ** 2 / d

    def density_matrix(self) -> np.ndarray:
        """Dense ``rho`` (``2^n x 2^n``) — for the collective channel signal."""
        d = self.dim
        p = self.components @ self.components.conj().T
        return (1.0 - self.q) * p + self.q * np.eye(d, dtype=np.complex128) / d


def _normalized(g: np.ndarray) -> np.ndarray:
    """Scale ``G`` so that ``Tr(G G^dag) = ||G||_F^2 = 1``."""
    return g / np.sqrt(float((np.abs(g) ** 2).sum()))


def noisy_pure(n: int, q: float, rng: np.random.Generator) -> NoisyState:
    """Noisy pure state ``(1-q)|psi><psi| + q I/2^n``, ``|psi>`` Haar-random."""
    dim = 2 ** n
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return NoisyState(_normalized(psi.reshape(dim, 1)), float(q), n)


def random_mixed(n: int, q: float, rng: np.random.Generator, rank: int | None = None) -> NoisyState:
    """Old ensemble: Ginibre high-rank component depolarized at rate ``q``.

    ``rank`` defaults to the full dimension ``2^n`` (highly mixed, purity ->
    ``~2^{-n}``); a smaller rank keeps sampling cheap at large ``n``.
    """
    dim = 2 ** n
    r = dim if rank is None else min(rank, dim)
    if r < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    g = rng.standard_normal((dim, r)) + 1j * rng.standard_normal((dim, r))
    return NoisyState(_normalized(g), float(q), n)
