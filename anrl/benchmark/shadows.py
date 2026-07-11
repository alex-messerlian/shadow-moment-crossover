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


def full_purity_ustatistic(snaps: np.ndarray) -> float:
    """Exact full pairwise U-statistic of ``Tr(rho^2)`` over ALL distinct pairs.

    The copy-optimal single-copy purity estimate.  Because
    ``Tr(rho_hat_i @ rho_hat_j) = prod_q phi(rho_hat_i^q) . phi(rho_hat_j^q)``
    for the real per-qubit feature ``phi`` (with
    ``phi(A) = [A00, A11, sqrt2*Re A01, sqrt2*Im A01]``), the tensor feature
    ``Phi_i = (x)_q phi(rho_hat_i^q)`` satisfies ``Tr(G_i G_j) = Phi_i . Phi_j``,
    so the full U-statistic is ``(|sum_i Phi_i|^2 - sum_i |Phi_i|^2) / (M(M-1))``
    — exact in ``O(M * 4^n)``, no pair enumeration or subsampling.
    """
    m, n = snaps.shape[0], snaps.shape[1]
    phi = np.stack(
        [
            snaps[:, :, 0, 0].real,
            snaps[:, :, 1, 1].real,
            np.sqrt(2.0) * snaps[:, :, 0, 1].real,
            np.sqrt(2.0) * snaps[:, :, 0, 1].imag,
        ],
        axis=-1,
    )  # (M, n, 4)
    feat = phi[:, 0, :]
    for q in range(1, n):
        feat = (feat[:, :, None] * phi[:, q, None, :]).reshape(m, -1)  # row-wise kron -> (M, 4^n)
    total = feat.sum(axis=0)
    return float((total @ total - np.einsum("ma,ma->", feat, feat)) / (m * (m - 1)))


def shadow_purity_estimate(
    rho: np.ndarray,
    n_snapshots: int,
    rng: np.random.Generator,
    n_pairs: int | None = None,
) -> float:
    """Single-copy classical-shadow estimate of ``Tr(rho^2)``.

    ``n_snapshots`` independent local shadows are drawn; the purity is the
    U-statistic ``mean of Tr(rho_hat_i @ rho_hat_j)`` over distinct snapshot
    pairs (unbiased).

    Copy accounting: the copy budget is ``n_snapshots``.  Forming pairs from the
    already-collected snapshots is **classical post-processing that consumes no
    copies**, so the copy-fair estimator uses the FULL set of pairs (minimum
    variance).  Therefore:

    * ``n_pairs is None`` (default): the exact full pairwise U-statistic over all
      ``M(M-1)/2`` pairs — the correct, copy-optimal estimator
      (:func:`full_purity_ustatistic`).
    * ``n_pairs`` given: a random subsample of that many pairs.  Subsampling only
      inflates the variance (it saves no copies), so it is NOT copy-fair; it is
      kept only for comparison / to reproduce the old ``n_pairs = M // 2``
      sandbox convention.  Documented as variance-inflating.
    """
    if n_snapshots < 2:
        raise ValueError(f"shadow purity needs >= 2 snapshots, got {n_snapshots}")
    rho = np.asarray(rho, dtype=np.complex128)
    n = int(round(np.log2(rho.shape[0])))

    snaps = _snapshots(rho, n, n_snapshots, rng)
    if n_pairs is None:
        return full_purity_ustatistic(snaps)  # exact, copy-optimal

    # Variance-inflating subsample (kept for comparison only; not copy-fair).
    total_pairs = n_snapshots * (n_snapshots - 1) // 2
    k = min(total_pairs, n_pairs)
    idx_i = rng.integers(0, n_snapshots, size=k)
    offset = rng.integers(1, n_snapshots, size=k)
    idx_j = (idx_i + offset) % n_snapshots
    a = snaps[idx_i]
    b = snaps[idx_j]
    per_qubit_trace = np.einsum("knab,knba->kn", a, b).real
    return float(per_qubit_trace.prod(axis=1).mean())
