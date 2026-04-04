"""
NROY sampling pipeline inspired by the hmer R package.

Provides two methods for generating non-implausible parameter samples:
- 'lhs': Pure LHS rejection sampling (simple, can be slow at low acceptance)
- 'ray_resample': 4-stage pipeline (LHS → ray sampling → importance sampling
  → maximin thinning) that efficiently explores small NROY regions

Reference: Iskauskas (2024), "Emulation and History Matching using the hmer
Package", Journal of Statistical Software.
"""

import logging
import time as _time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .emulator_bank import EmulatorBank
from .observation_data import ObservationData
from .parameter_space import ParameterSpace
from .sampling import SamplingStrategy, SamplingStrategyFactory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NROYResult:
    """Container for NROY sampling results with stats."""
    def __init__(self, samples: pd.DataFrame, lhs_accepted: int, lhs_tested: int):
        self.samples = samples
        self.lhs_accepted = lhs_accepted
        self.lhs_tested = lhs_tested

    def __len__(self):
        return len(self.samples)


def generate_nroy_design(
    n_points: int,
    parameter_space: ParameterSpace,
    emulator_bank: EmulatorBank,
    observations: ObservationData,
    threshold: float = 3.5,
    sampling_strategy: Optional[SamplingStrategy] = None,
    method: str = 'lhs',
    seed: Optional[int] = None,
    # Stage 1: LHS rejection
    lhs_factor: int = 10,
    # Stage 2: ray sampling
    n_lines: int = 20,
    points_per_line: int = 50,
    # Stage 3: importance sampling
    imp_scale: float = 1.0,
    imp_target_rate: Tuple[float, float] = (0.10, 0.225),
    imp_batch_size: int = 1000,
    imp_max_batches: int = 200,
    # Auto-fallback to pure LHS when acceptance is above this rate
    lhs_fallback_rate: float = 0.10,
    # Stage 4: maximin thinning
    maximin_reps: int = 1000,
    # Safety
    max_candidates: int = 4_000_000,
) -> NROYResult:
    """Generate NROY parameter samples filtered through all emulators.

    Parameters
    ----------
    n_points : int
        Target number of NROY samples to return.
    parameter_space : ParameterSpace
        Bounds for each parameter.
    emulator_bank : EmulatorBank
        All trained emulators (across waves) used for filtering.
    observations : ObservationData
        Target values and uncertainties.
    threshold : float
        Implausibility threshold (default 3.5).
    sampling_strategy : SamplingStrategy, optional
        Strategy for generating initial LHS candidates. Defaults to LHS maximin.
    method : str
        ``'ray_resample'`` (default) for the 4-stage pipeline, or
        ``'lhs'`` for pure rejection sampling.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Up to *n_points* rows of NROY parameter samples.
    """
    if sampling_strategy is None:
        sampling_strategy = SamplingStrategyFactory.create('lhs')

    param_names = parameter_space.get_parameter_names()
    n_dims = len(param_names)
    rng = np.random.default_rng(seed)

    # Shared filter closure
    def filter_nroy(candidates_df: pd.DataFrame) -> pd.DataFrame:
        return _filter_nroy(candidates_df, emulator_bank, observations,
                            threshold, param_names)

    # ── Step 1: LHS rejection (always the primary method) ───────────
    t0 = _time.time()
    lhs_result = _lhs_reject_loop(
        n_points, parameter_space, sampling_strategy, filter_nroy,
        seed=seed, max_candidates=max_candidates,
    )
    elapsed = _time.time() - t0
    n_lhs_found = len(lhs_result)
    logger.info(f"  LHS rejection: {n_lhs_found}/{n_points} [{elapsed:.1f}s]")

    if n_lhs_found >= n_points:
        _log_summary(n_points, n_lhs_found, 0, 0, elapsed)
        return NROYResult(lhs_result.head(n_points), n_lhs_found, max_candidates)

    # ── Step 2: LHS fell short ────────────────────────────────────
    if method == 'lhs':
        # User explicitly requested LHS only — return what we have
        logger.warning(
            f"  LHS found only {n_lhs_found}/{n_points} NROY samples. "
            f"The NROY space may be near-empty.")
        return NROYResult(lhs_result, n_lhs_found, max_candidates)

    # ── Step 3: Escalate to ray_resample using LHS points as seed ─
    logger.info(f"  ESCALATING: LHS found {n_lhs_found}/{n_points} — switching to ray + importance sampling")
    pool = lhs_result.copy()
    n_from_lhs = len(pool)
    n_from_ray = 0
    n_from_imp = 0

    if len(pool) < 2:
        logger.warning(
            f"  Cannot run ray/importance with <2 seed points. "
            f"Returning {len(pool)} LHS points.")
        _log_summary(n_points, n_from_lhs, 0, 0, _time.time() - t0)
        return NROYResult(pool, n_lhs_found, max_candidates)

    rng = np.random.default_rng(seed)

    # Ray sampling — scale effort based on how far short we are
    t1 = _time.time()
    shortfall_ratio = n_points / max(len(pool), 1)  # how many x more points we need
    scaled_n_lines = min(int(n_lines * max(shortfall_ratio, 1)), 200)
    scaled_ppl = min(int(points_per_line * max(shortfall_ratio, 1)), 500)
    logger.info(f"  Ray sampling: {scaled_n_lines} rays × {scaled_ppl} pts "
                f"(scaled {shortfall_ratio:.0f}x for shortfall, "
                f"seeded with {len(pool)} points)...")
    ray_nroy = _ray_sample(
        pool, parameter_space, filter_nroy, rng,
        n_lines=scaled_n_lines, points_per_line=scaled_ppl,
    )
    n_from_ray = len(ray_nroy)
    if len(ray_nroy) > 0:
        pool = pd.concat([pool, ray_nroy], ignore_index=True)
    logger.info(f"  Ray sampling: +{n_from_ray}, pool={len(pool)}/{n_points} "
                f"[{_time.time()-t1:.1f}s]")

    if len(pool) >= n_points:
        result = _maximin_thin(pool, n_points, maximin_reps, rng)
        _log_summary(n_points, n_from_lhs, n_from_ray, 0, _time.time() - t0)
        return NROYResult(result, n_lhs_found, max_candidates)

    # Importance sampling
    if len(pool) >= 2:
        t2 = _time.time()
        remaining = n_points - len(pool)
        logger.info(f"  Importance sampling: need {remaining} more "
                    f"(proposals centered on {len(pool)} pool points)...")
        imp_nroy = _importance_sample(
            pool, parameter_space, filter_nroy, rng,
            n_target=remaining,
            scale=imp_scale,
            target_rate=imp_target_rate,
            batch_size=imp_batch_size,
            max_batches=imp_max_batches,
        )
        n_from_imp = len(imp_nroy)
        if len(imp_nroy) > 0:
            pool = pd.concat([pool, imp_nroy], ignore_index=True)
        elapsed = _time.time() - t2
        logger.info(f"  Importance sampling: +{n_from_imp}, "
                    f"pool={len(pool)}/{n_points} [{elapsed:.1f}s]")

    # Maximin thinning if we overshot
    if len(pool) > n_points:
        pool = _maximin_thin(pool, n_points, maximin_reps, rng)

    total = _time.time() - t0
    _log_summary(n_points, n_from_lhs, n_from_ray, n_from_imp, total)

    if len(pool) < n_points:
        logger.warning(
            f"  NROY INSUFFICIENT: {len(pool)}/{n_points} after all methods. "
            f"The NROY space is near-empty — the model may be over-constrained.")

    return NROYResult(pool, n_lhs_found, max_candidates)


# ---------------------------------------------------------------------------
# Stage 1: Pure LHS rejection loop (also used as fallback)
# ---------------------------------------------------------------------------

def _lhs_reject_loop(
    n_points: int,
    parameter_space: ParameterSpace,
    sampling_strategy: SamplingStrategy,
    filter_fn,
    seed: Optional[int] = None,
    max_candidates: int = 4_000_000,
    oversample_factor: float = 1.1,
    max_batch_size: int = 100_000,
) -> pd.DataFrame:
    """Pure LHS rejection sampling (existing behavior, extracted)."""
    plausible = pd.DataFrame()
    total_generated = 0
    batch_seed = seed
    batch_size = min(int(n_points * oversample_factor), max_batch_size)
    last_pct = -10
    t0 = _time.time()

    logger.info(f"  LHS rejection: 0/{n_points} (0%) — starting")

    while len(plausible) < n_points:
        candidates = sampling_strategy.generate_samples(
            parameter_space, batch_size, seed=batch_seed)
        if batch_seed is not None:
            batch_seed += 1

        batch_nroy = filter_fn(candidates)
        if len(batch_nroy) > 0:
            plausible = pd.concat([plausible, batch_nroy], ignore_index=True)

        total_generated += len(candidates)
        rate = len(plausible) / total_generated if total_generated > 0 else 0

        pct = int(100 * len(plausible) / n_points)
        if pct >= last_pct + 10:
            last_pct = pct - (pct % 10)
            elapsed = _time.time() - t0
            logger.info(
                f"  LHS rejection: {len(plausible)}/{n_points} ({pct}%) "
                f"| {total_generated:,} tested | rate={rate:.4%} [{elapsed:.0f}s]")

        if len(plausible) >= n_points:
            break

        if total_generated >= max_candidates:
            logger.warning(
                f"  LHS rejection: reached {max_candidates:,} candidates with only "
                f"{len(plausible)}/{n_points} NROY. Returning what we have.")
            break

        # Adaptive batch sizing
        remaining = n_points - len(plausible)
        if rate > 0:
            batch_size = min(int(oversample_factor * remaining / rate), max_batch_size)
        else:
            batch_size = min(batch_size * 2, max_batch_size)

    return plausible.head(n_points)


# ---------------------------------------------------------------------------
# Stage 2: Ray sampling
# ---------------------------------------------------------------------------

def _ray_sample(
    nroy_points: pd.DataFrame,
    parameter_space: ParameterSpace,
    filter_fn,
    rng: np.random.Generator,
    n_lines: int = 20,
    points_per_line: int = 50,
) -> pd.DataFrame:
    """Sample along rays connecting distant NROY point pairs."""
    X = nroy_points.values
    n_pts, n_dims = X.shape
    param_names = list(nroy_points.columns)

    # Get parameter bounds for clipping
    ps_df = parameter_space.to_dataframe()
    lo = ps_df['minimum'].values
    hi = ps_df['maximum'].values

    # Select pairs with largest pairwise distance
    if n_pts <= n_lines * 2:
        # Few points — use all pairs
        pairs = [(i, j) for i in range(n_pts) for j in range(i + 1, n_pts)]
    else:
        # Sample 10x candidate pairs, keep the n_lines most distant
        n_candidate_pairs = min(10 * n_lines, n_pts * (n_pts - 1) // 2)
        idx_a = rng.integers(0, n_pts, size=n_candidate_pairs)
        idx_b = rng.integers(0, n_pts, size=n_candidate_pairs)
        # Avoid self-pairs
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        # Compute distances
        dists = np.sqrt(np.sum((X[idx_a] - X[idx_b]) ** 2, axis=1))
        # Keep top n_lines by distance
        top_idx = np.argsort(dists)[-n_lines:]
        pairs = list(zip(idx_a[top_idx], idx_b[top_idx]))

    pairs = pairs[:n_lines]

    # Generate ray samples
    all_ray_pts = []
    for i, j in pairs:
        p1, p2 = X[i], X[j]
        midpoint = (p1 + p2) / 2
        direction = p2 - p1
        d = np.linalg.norm(direction)
        if d < 1e-12:
            continue

        # Sample t uniformly in [-d, d], giving points extending beyond both endpoints
        t_vals = rng.uniform(-d, d, size=points_per_line)
        ray_pts = midpoint + t_vals[:, np.newaxis] * direction

        # Clip to parameter bounds
        ray_pts = np.clip(ray_pts, lo, hi)
        all_ray_pts.append(ray_pts)

    if not all_ray_pts:
        return pd.DataFrame({c: pd.Series(dtype='float64') for c in param_names})

    ray_candidates = pd.DataFrame(
        np.vstack(all_ray_pts), columns=param_names)

    return filter_fn(ray_candidates)


# ---------------------------------------------------------------------------
# Stage 3: Importance sampling with PCA-oriented proposals
# ---------------------------------------------------------------------------

def _importance_sample(
    nroy_points: pd.DataFrame,
    parameter_space: ParameterSpace,
    filter_fn,
    rng: np.random.Generator,
    n_target: int,
    scale: float = 1.0,
    target_rate: Tuple[float, float] = (0.10, 0.225),
    batch_size: int = 1000,
    max_batches: int = 200,
) -> pd.DataFrame:
    """PCA-oriented multivariate normal proposals centered on existing NROY points."""
    X = nroy_points.values
    n_pts, n_dims = X.shape
    param_names = list(nroy_points.columns)

    # Get parameter bounds for clipping
    ps_df = parameter_space.to_dataframe()
    lo = ps_df['minimum'].values
    hi = ps_df['maximum'].values

    # PCA of existing NROY points
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    # Regularize in case of near-singular covariance
    cov += np.eye(n_dims) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Scale by eigenvalues for PCA-oriented proposals
    std_pca = np.sqrt(np.maximum(eigvals, 1e-12))

    collected = pd.DataFrame({c: pd.Series(dtype='float64') for c in param_names})
    total_proposed = 0

    for batch_num in range(max_batches):
        if len(collected) >= n_target:
            break

        # Pick random existing NROY points as centers
        center_idx = rng.integers(0, n_pts, size=batch_size)
        centers = X[center_idx]

        # Propose in PCA space: standard normal scaled by eigenvalues and adaptive scale
        z = rng.standard_normal(size=(batch_size, n_dims))
        proposals_pca = z * std_pca * scale
        # Transform back to original space
        proposals = centers + proposals_pca @ eigvecs.T

        # Clip to bounds
        proposals = np.clip(proposals, lo, hi)
        proposals_df = pd.DataFrame(proposals, columns=param_names)

        # Filter through emulators
        accepted = filter_fn(proposals_df)
        total_proposed += batch_size

        if len(accepted) > 0:
            collected = pd.concat([collected, accepted], ignore_index=True)

        # Adaptive scale adjustment
        rate = len(collected) / total_proposed if total_proposed > 0 else 0
        if rate > target_rate[1]:
            scale *= 1.1  # too many accepted → explore wider
        elif rate < target_rate[0] and rate > 0:
            scale *= 0.9  # too few → tighten

        # Progress logging every 10 batches
        if (batch_num + 1) % 10 == 0:
            pct = int(100 * len(collected) / n_target) if n_target > 0 else 100
            logger.info(
                f"    importance sampling: {len(collected)}/{n_target} ({pct}%) "
                f"| scale={scale:.3f} | rate={rate:.4%}")

    return collected.head(n_target)


# ---------------------------------------------------------------------------
# Stage 4: Maximin thinning
# ---------------------------------------------------------------------------

def _maximin_thin(
    pool: pd.DataFrame,
    n_target: int,
    reps: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Select a space-filling subset via randomized maximin criterion."""
    if len(pool) <= n_target:
        return pool

    if rng is None:
        rng = np.random.default_rng()

    X = pool.values
    n = len(X)
    best_subset = None
    best_min_dist = -1.0

    for _ in range(reps):
        idx = rng.choice(n, size=n_target, replace=False)
        subset = X[idx]
        min_dist = pdist(subset).min() if n_target > 1 else 0.0
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_subset = idx

    return pool.iloc[best_subset].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared NROY filter
# ---------------------------------------------------------------------------

def _filter_nroy(
    candidates: pd.DataFrame,
    emulator_bank: EmulatorBank,
    observations: ObservationData,
    threshold: float,
    param_cols: list,
) -> pd.DataFrame:
    """Filter candidates through all emulators with short-circuit evaluation."""
    mask = np.ones(len(candidates), dtype=bool)

    for iteration in reversed(emulator_bank.get_all_iterations()):
        emulators = emulator_bank.get_emulators_for_iteration(iteration)
        for feature_name, emulator in emulators.items():
            if mask.sum() == 0:
                break
            if not observations.has_feature(feature_name):
                continue
            try:
                active = candidates.loc[mask, param_cols]
                predictions = emulator.predict(active)
                pred_mean = predictions.get_mean()
                pred_var = predictions.get_variance()
                feature_impl = observations.calculate_implausibility(
                    feature_name, pred_mean, pred_var
                )
                failures = np.asarray(feature_impl > threshold, dtype=bool).ravel()
                mask[mask] &= ~failures
            except Exception as e:
                logger.warning(f"Filter failed for '{feature_name}': {e}")
                continue

    return candidates[mask]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total > 0 else "0%"


def _log_summary(n_target: int, n_lhs: int, n_ray: int, n_imp: int, elapsed: float):
    """Log a clear summary of where NROY points came from."""
    total = n_lhs + n_ray + n_imp
    parts = []
    if n_lhs > 0:
        parts.append(f"LHS={n_lhs}")
    if n_ray > 0:
        parts.append(f"ray={n_ray}")
    if n_imp > 0:
        parts.append(f"importance={n_imp}")
    breakdown = ", ".join(parts) if parts else "none"
    status = "OK" if total >= n_target else "INSUFFICIENT"
    logger.info(f"  NROY SUMMARY: {min(total, n_target)}/{n_target} [{status}] — "
                f"sources: {breakdown} [{elapsed:.1f}s]")
