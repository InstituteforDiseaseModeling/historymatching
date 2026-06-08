"""
IterationResult — the result of a single history matching wave.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sciris as sc

from .parameter_space import ParameterSpace
from .emulators.base import BaseEmulator
from .constants import NROY_COLOR, SAVE_KW


@dataclass(frozen=True)
class IterationResult:
    """
    Immutable result from one history matching wave (iteration).

    Holds the parameter samples run this wave, the simulator outputs, the
    outputs that were emulated, the trained emulators, and the fraction of
    parameter space still plausible (``nroy_fraction``).

    The NROY (Not Ruled Out Yet) set itself is not stored here — it is defined
    implicitly by the emulator bank.  Use :meth:`HistoryMatching.get_nroy_samples`
    to draw plausible samples.  The final wave's ``samples`` +
    ``simulation_results`` can be fed directly into trajectory selection.
    """

    iteration: int
    parameter_space: ParameterSpace
    samples: pd.DataFrame
    simulation_results: pd.DataFrame
    emulated_outputs: List[str]
    emulators: Dict[str, BaseEmulator]
    nroy_fraction: float
    execution_time_seconds: float

    def get_emulator(self, output: str) -> BaseEmulator:
        """
        Get the emulator trained for a specific output this wave.

        Args:
            output: Name of the emulated output.

        Returns:
            Emulator instance.

        Raises:
            KeyError: If no emulator was trained for that output this wave.
        """
        if output not in self.emulators:
            available = list(self.emulators.keys())
            raise KeyError(f"No emulator for output '{output}'. Emulated this wave: {available}")

        return self.emulators[output]

    def plot_emulator_diagnostics(self, output: str, **kwargs):
        """
        Plot diagnostics for a specific output's emulator.

        Args:
            output: Name of the emulated output.
            **kwargs: Additional arguments passed to the emulator's plot method.
        """
        emulator = self.get_emulator(output)

        if hasattr(emulator, 'plot_diagnostics'):
            emulator.plot_diagnostics(**kwargs)
        elif hasattr(emulator, 'plot'):
            emulator.plot(**kwargs)
        else:
            print(f"Emulator for output '{output}' does not support plotting")

    def get_emulator_quality_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Quality metrics for each emulated output.

        Returns:
            Dict mapping each output name to a metrics dict with keys
            ``r2`` (test R²), ``mse`` (test MSE), and ``n_train`` (training points).
            A key is absent if that metric could not be computed.
        """
        metrics = {}

        for output, emulator in self.emulators.items():
            m: Dict[str, float] = {}

            # Ensure the emulator has been tested (the engine only train()s).
            if hasattr(emulator, 'test') and not getattr(emulator, 'testing_complete', False):
                try:
                    emulator.test()
                except Exception:
                    pass  # Testing may fail; metrics will simply be absent.

            em_metrics = getattr(emulator, 'emulator_metrics', {})
            if 'R2' in em_metrics:
                m['r2'] = float(em_metrics['R2'])
            if 'MSE' in em_metrics:
                m['mse'] = float(em_metrics['MSE'])
            if getattr(emulator, 'X_train', None) is not None:
                m['n_train'] = len(emulator.X_train)

            metrics[output] = m

        return metrics

    def quality_table(self) -> pd.DataFrame:
        """Per-output emulator quality as a table (rows = outputs; columns
        ``r2``/``mse``/``n_train``). Renders nicely in notebooks."""
        metrics = self.get_emulator_quality_metrics()
        if not metrics:
            return pd.DataFrame()
        return pd.DataFrame(metrics).T

    def plot_emulator_quality(self, *, ax=None, **kwargs):
        """Bar chart of per-output emulator R² (delegates to
        :func:`historymatching.plotting.plot_emulator_quality`)."""
        from . import plotting
        return plotting.plot_emulator_quality(self.get_emulator_quality_metrics(), ax=ax, **kwargs)

    def plot_predicted_vs_actual(self, output: str, *, ax=None, **kwargs):
        """Predicted-vs-actual scatter for one output's emulator on its held-out
        test set (delegates to
        :func:`historymatching.plotting.plot_predicted_vs_actual`)."""
        from . import plotting
        em = self.get_emulator(output)
        if hasattr(em, "test") and not getattr(em, "testing_complete", False):
            em.test()
        if getattr(em, "y_test", None) is None or getattr(em, "y_test_pred", None) is None:
            raise ValueError(
                f"Emulator for '{output}' has no held-out test data to plot "
                "(it may have been trained with test_fraction=0). Re-run with a "
                "non-zero test fraction to enable predicted-vs-actual diagnostics."
            )
        m = getattr(em, "emulator_metrics", {})
        n_train = len(em.X_train) if getattr(em, "X_train", None) is not None else None
        return plotting.plot_predicted_vs_actual(
            em.y_test, em.y_test_pred,
            r2=m.get("R2"), mse=m.get("MSE"), n_train=n_train,
            title=f"{output} — predicted vs actual", ax=ax, **kwargs)

    def summary(self) -> Dict[str, Any]:
        """
        A summary of this wave as a plain dict (handy for logging/inspection).
        """
        emulator_metrics = self.get_emulator_quality_metrics()
        r2_scores = [m['r2'] for m in emulator_metrics.values() if 'r2' in m]
        avg_r2 = float(np.mean(r2_scores)) if r2_scores else None

        return {
            'iteration': self.iteration,
            'n_samples': len(self.samples),
            'n_outputs': len(self.emulated_outputs),
            'emulated_outputs': list(self.emulated_outputs),
            'nroy_fraction': self.nroy_fraction,
            'execution_time_seconds': self.execution_time_seconds,
            'parameter_count': len(self.parameter_space),
            'average_emulator_r2': avg_r2,
            'emulator_metrics': emulator_metrics,
        }

    def save(self, directory: str, all_results: Optional[list] = None) -> str:
        """
        Save everything about this wave to ``{directory}/wave{N}/``.

        Writes the parameter ``samples.csv`` and ``simulation_results.csv``, a
        pickle of each emulator under ``emulators/``, per-output diagnostic
        figures (predicted-vs-actual, and ARD lengthscales for GPR), a
        ``metrics.json``, and — when ``all_results`` is supplied — a
        ``convergence.png`` showing the plausible fraction across waves.

        Args:
            directory: Parent directory; a ``wave{N}/`` subfolder is created.
            all_results: Optional list of all waves so far, used for the
                convergence plot.

        Returns:
            The path to the ``wave{N}/`` directory that was written.
        """
        import matplotlib  # noqa: F401  (ensure a backend is selected)
        import matplotlib.pyplot as plt

        wave_dir = sc.path(directory) / f"wave{self.iteration}"
        wave_dir.mkdir(parents=True, exist_ok=True)

        # ── Raw data ──────────────────────────────────────────────────
        self.samples.to_csv(wave_dir / "samples.csv", index=False)
        self.simulation_results.to_csv(
            wave_dir / "simulation_results.csv", index=False)

        # ── Emulators (pickled) ───────────────────────────────────────
        em_dir = wave_dir / "emulators"
        for output, emulator in self.emulators.items():
            try:
                # sc.save auto-creates em_dir and compresses the pickle.
                sc.save(em_dir / f"{output}.pkl", emulator, die=True)
            except Exception:
                pass  # Some emulators may not pickle; skip rather than fail the save.

        all_metrics = self.get_emulator_quality_metrics()

        # ── Per-output diagnostic figures ─────────────────────────────
        for output, emulator in self.emulators.items():
            if not getattr(emulator, 'testing_complete', False):
                try:
                    emulator.test()
                except Exception:
                    continue

            ard_ls = None
            ard_names = None
            model = getattr(emulator, 'model', None)
            if model is not None and hasattr(model, 'kernel'):
                try:
                    ls = model.kernel.lengthscales.numpy()
                    if ls.ndim > 0:
                        ard_names = (list(emulator.X_train_df.columns)
                                     if hasattr(emulator, 'X_train_df') else
                                     [f"dim_{i}" for i in range(len(ls))])
                        ard_ls = ls
                        ls_dict = {n: float(v) for n, v in zip(ard_names, ls)}
                        all_metrics.setdefault(output, {})['ard_lengthscales'] = ls_dict
                except Exception:
                    pass

            has_ard = ard_ls is not None
            ncols = 2 if has_ard else 1
            fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
            if ncols == 1:
                axes = [axes]

            # Panel 1: predicted vs actual
            ax = axes[0]
            y_true = emulator.y_test.flatten()
            y_pred = emulator.y_test_pred.flatten()
            ax.scatter(y_true, y_pred, s=12, alpha=0.6, edgecolors='none')
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            margin = (hi - lo) * 0.05 or 1.0
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    '--', color='grey', linewidth=0.8, alpha=0.6)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)
            ax.set_xlabel('Simulation (true)', fontsize=9)
            ax.set_ylabel('Emulator (predicted)', fontsize=9)

            em = getattr(emulator, 'emulator_metrics', {})
            r2 = em.get('R2', float('nan'))
            mse = em.get('MSE', float('nan'))
            n_train = len(emulator.X_train) if emulator.X_train is not None else '?'
            ax.set_title(f"Predicted vs Actual\nR²={r2:.3f}  MSE={mse:.3g}  n={n_train}",
                         fontsize=9)
            ax.set_aspect('equal', adjustable='box')

            # Panel 2: ARD lengthscales (GPR only)
            if has_ard:
                ax2 = axes[1]
                order = np.argsort(ard_ls)
                sorted_names = [ard_names[i] for i in order]
                sorted_ls = ard_ls[order]
                colors = ['#d44d4d' if v < np.median(ard_ls) else '#888888'
                          for v in sorted_ls]
                ax2.barh(range(len(sorted_ls)), sorted_ls, color=colors, height=0.7)
                ax2.set_yticks(range(len(sorted_ls)))
                ax2.set_yticklabels([n.replace('_', '\n') for n in sorted_names], fontsize=6)
                ax2.set_xlabel('Lengthscale (shorter = more relevant)', fontsize=8)
                ax2.set_title('ARD Lengthscales', fontsize=9)

            for ax in axes:
                sc.boxoff(ax)

            fig.suptitle(f"Wave {self.iteration} — {output}",
                         fontsize=11, fontweight='bold', y=1.02)
            fig.tight_layout()
            sc.savefig(wave_dir / f"{output}_diagnostics.png", **SAVE_KW)
            plt.close(fig)

        # ── Convergence across waves ──────────────────────────────────
        if all_results:
            fig, ax = plt.subplots(figsize=(7, 4))
            waves = [r.iteration for r in all_results]
            fracs = [r.nroy_fraction for r in all_results]
            ax.bar(waves, fracs, color=NROY_COLOR, alpha=0.8, edgecolor='white')
            for w, frac in zip(waves, fracs):
                ax.annotate(f"{frac:.1%}", (w, frac), textcoords='offset points',
                            xytext=(0, 6), ha='center', fontsize=8)
            ax.set_xlabel('Wave', fontsize=10)
            ax.set_ylabel('Fraction of space remaining (NROY)', fontsize=10)
            ax.set_title('Convergence', fontsize=11, fontweight='bold')
            ax.set_ylim(0, min(1.0, max(fracs) * 1.3) if fracs else 1.0)
            ax.set_xticks(waves)
            sc.boxoff(ax)
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            sc.savefig(wave_dir / "convergence.png", **SAVE_KW)
            plt.close(fig)

        # ── Metrics ───────────────────────────────────────────────────
        sc.savejson(wave_dir / "metrics.json", all_metrics, indent=2)

        return wave_dir

    def __post_init__(self):
        """Validate the result after creation."""
        if self.iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {self.iteration}")

        if len(self.samples) != len(self.simulation_results):
            raise ValueError("Samples and simulation results must have same length")

        if not 0.0 <= self.nroy_fraction <= 1.0:
            raise ValueError(f"NROY fraction must be between 0 and 1, got {self.nroy_fraction}")

        for output in self.emulated_outputs:
            if output not in self.emulators:
                raise ValueError(f"Emulated output '{output}' missing from emulators")

        if self.execution_time_seconds < 0:
            raise ValueError(f"Execution time must be non-negative, got {self.execution_time_seconds}")

    def _precomputed_r2(self) -> List[float]:
        """Test-R² values already computed (no side effects), for repr/str."""
        out = []
        for em in self.emulators.values():
            r2 = getattr(em, 'emulator_metrics', {}).get('R2')
            if r2 is not None:
                out.append(float(r2))
        return out

    def __str__(self) -> str:
        return (f"Wave {self.iteration}: {len(self.samples)} samples, "
                f"outputs {self.emulated_outputs}, "
                f"plausible {self.nroy_fraction:.1%} of space remaining (NROY)")

    def __repr__(self) -> str:
        outs = list(self.emulated_outputs)
        out_repr = repr(outs[0]) if len(outs) == 1 else repr(outs)
        r2s = self._precomputed_r2()
        r2_str = f", emulator R²={min(r2s):.2f}" if r2s else ""
        return (f"IterationResult(wave {self.iteration}: {len(self.samples)} samples, "
                f"output {out_repr}, plausible {self.nroy_fraction:.1%} remaining (NROY){r2_str})")
