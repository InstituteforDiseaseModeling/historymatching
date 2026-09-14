"""End-to-end pytest coverage of checkpoint creation, resume, and serial NROY.

Previously this file only ran its assertions under ``if __name__ == '__main__'``,
so pytest never collected it and the checkpoint/resume path was untested in CI.
It is now a set of collected pytest tests sharing a module-scoped engine.
"""
import json
import os
import shutil
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'docs', 'tutorials'))
from model import SIR, generate_observed_data

import historymatching as hm

POPULATION = 5_000
SEED_INFECTIONS = 50

obs_incidence, _ = generate_observed_data(
    beta_true=0.7, gamma_true=0.4,
    population_size=POPULATION, n_seed_infections=SEED_INFECTIONS, seed=42,
)


def run_sir(samples):
    rows = []
    for _, row in samples.iterrows():
        m = SIR(beta=row['beta'], gamma=row['gamma'],
                s0=POPULATION - SEED_INFECTIONS, i0=SEED_INFECTIONS)
        inc = m.get_incidence()
        rows.append({'peak_incidence': float(inc.max()),
                     'attack_rate': float(inc.sum() / POPULATION)})
    return pd.DataFrame(rows)


def make_engine(output_dir):
    obs_peak = float(obs_incidence.max())
    obs_ar = float(obs_incidence.sum() / POPULATION)
    return hm.HistoryMatching(
        function=run_sir,
        bounds={'beta': (0.3, 2.0), 'gamma': (0.1, 0.8)},
        observations={'peak_incidence': (obs_peak, obs_peak * 0.10),
                      'attack_rate': (obs_ar, obs_ar * 0.05)},
        emulator_type='gpr',
        n_samples=100,
        max_iterations=2,
        output_dir=output_dir,
        run_name='test_run',
    )


@pytest.fixture(scope="module")
def output_dir():
    d = tempfile.mkdtemp(prefix="hm_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def first_wave(output_dir):
    """Run and commit wave 1, returning the engine and its run_dir."""
    engine = make_engine(output_dir)
    assert engine.run_dir is None, "output is created lazily, not at construction"
    engine.step()
    engine.commit_step()
    return engine, engine.run_dir


def test_checkpoint_creation(first_wave):
    """Wave 1 writes the checkpoint, config, and per-feature artifacts."""
    engine, run_dir = first_wave
    wave1_dir = run_dir / "wave1"

    assert wave1_dir.exists(), "wave1/ not created"
    assert (run_dir / "checkpoint.pkl").exists(), "checkpoint.pkl not created"
    assert (run_dir / "run_config.json").exists(), "run_config.json not created"
    assert (wave1_dir / "convergence.png").exists(), "convergence.png missing"
    assert (wave1_dir / "nroy_samples.csv").exists(), "nroy_samples.csv missing"

    feature_dirs = [d for d in wave1_dir.iterdir() if d.is_dir()]
    assert feature_dirs, "no per-feature directories created"
    for feat_dir in feature_dirs:
        assert (feat_dir / "emulator.pkl").exists(), f"emulator.pkl missing in {feat_dir}"
        assert (feat_dir / "metrics.json").exists(), f"metrics.json missing in {feat_dir}"
        with open(feat_dir / "metrics.json") as f:
            json.load(f)  # must be valid JSON


def test_resume_from_checkpoint(first_wave, output_dir):
    """A fresh engine resumes from the checkpoint and runs the remaining wave."""
    _, run_dir = first_wave

    engine2 = make_engine(output_dir)
    engine2.run(resume=True)

    assert engine2.current_iteration == 2, "resume did not continue to wave 2"
    assert (run_dir / "wave2").exists(), "wave2/ not created after resume"


def test_serial_nroy_sampling(first_wave, output_dir):
    """get_nroy_samples returns samples, and honors a larger explicit request."""
    _, run_dir = first_wave
    engine = make_engine(output_dir)
    engine.run(resume=True)

    nroy = engine.get_nroy_samples()
    assert len(nroy) > 0

    nroy200 = engine.get_nroy_samples(n=200)
    assert len(nroy200) >= 100
