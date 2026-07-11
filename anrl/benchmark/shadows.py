"""Single-copy classical-shadow purity estimation.

For each snapshot we draw an independent Haar-random single-qubit unitary per
qubit, rotate the state, and sample one computational-basis outcome.  The local
classical-shadow snapshot per qubit is

    rho_hat_q = 3 * U_q^dag |b_q><b_q| U_q - I,

whose expectation is the qubit's reduced-channel inverse acting on the true
state; tensored across qubits, ``E[rho_hat] = rho``.  Purity is the U-statistic
over distinct snapshot pairs,

    Tr(rho^2) ~= mean_{i != j} Tr(rho_hat_i @ rho_hat_j),

which is unbiased.  Because each snapshot is a tensor product, the pairwise trace
factorizes, ``Tr(rho_hat_i @ rho_hat_j) = prod_q Tr(rho_hat_i^q @ rho_hat_j^q)``,
so we store per-qubit 2x2 snapshots and never form the full 2^n matrix for the
pairing.  The estimator variance grows exponentially with n.
"""

from __future__ import annotations

import numpy as np

from anrl.physics import kron_all

_KET_BRA = (
    np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),  # |0><0|
    np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),  # |1><1|
)
_I2 = np.eye(2, dtype=np.complex128)


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a Haar-random ``dim x dim`` unitary (Mezzadri QR method)."""
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diag = np.diagonal(r)
    phases = diag / np.abs(diag)
    return q * phases  # q @ diag(phases), Haar distributed


def _snapshots(rho: np.ndarray, n: int, n_snapshots: int, rng: np.random.Generator) -> np.ndarray:
    """Generate ``(n_snapshots, n, 2, 2)`` per-qubit local-shadow snapshots."""
    dim = 2 ** n
    snaps = np.empty((n_snapshots, n, 2, 2), dtype=np.complex128)
    shifts = (n - 1 - np.arange(n))
    for s in range(n_snapshots):
        unitaries = [haar_unitary(2, rng) for _ in range(n)]
        u_full = kron_all(unitaries)
        rotated = u_full @ rho @ u_full.conj().T
        probs = np.clip(np.real(np.diag(rotated)), 0.0, None)
        probs /= probs.sum()
        outcome = int(rng.choice(dim, p=probs))
        bits = (outcome >> shifts) & 1
        for q in range(n):
            u_q = unitaries[q]
            rho_meas = u_q.conj().T @ _KET_BRA[bits[q]] @ u_q
            snaps[s, q] = 3.0 * rho_meas - _I2
    return snaps


def shadow_purity_estimate(
    rho: np.ndarray,
    n_snapshots: int,
    rng: np.random.Generator,
    n_pairs: int | None = None,
) -> float:
    """Single-copy classical-shadow estimate of ``Tr(rho^2)``.

    ``n_snapshots`` independent local shadows are drawn; the purity is the mean
    of ``Tr(rho_hat_i @ rho_hat_j)`` over ``n_pairs`` randomly subsampled
    distinct snapshot pairs (unbiased U-statistic).  ``n_pairs`` defaults to
    ``n_snapshots // 2`` — an O(n_snapshots) efficient pairing (each snapshot
    used about once) that is consistent as the snapshot count grows and reflects
    the realistic single-copy variance (many-more pairs would only lower the
    Monte-Carlo pairing noise, understating the estimator's true difficulty).
    """
    if n_snapshots < 2:
        raise ValueError(f"shadow purity needs >= 2 snapshots, got {n_snapshots}")
    rho = np.asarray(rho, dtype=np.complex128)
    n = int(round(np.log2(rho.shape[0])))

    snaps = _snapshots(rho, n, n_snapshots, rng)

    total_pairs = n_snapshots * (n_snapshots - 1) // 2
    default_pairs = max(1, n_snapshots // 2)
    k = min(total_pairs, default_pairs if n_pairs is None else n_pairs)

    # Sample k distinct ordered pairs (i, j != i); Tr is symmetric so ordered
    # sampling is unbiased for the unordered U-statistic mean.
    idx_i = rng.integers(0, n_snapshots, size=k)
    offset = rng.integers(1, n_snapshots, size=k)
    idx_j = (idx_i + offset) % n_snapshots

    a = snaps[idx_i]  # (k, n, 2, 2)
    b = snaps[idx_j]
    # Per-qubit Tr(A @ B) = sum_{ab} A_ab B_ba, then product over qubits.
    per_qubit_trace = np.einsum("knab,knba->kn", a, b).real
    pair_terms = per_qubit_trace.prod(axis=1)
    return float(pair_terms.mean())
