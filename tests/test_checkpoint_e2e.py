"""Quick end-to-end test of checkpoint, resume, and parallel NROY."""
import sys, os, shutil, json, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'docs', 'tutorials'))
from model import SIR, generate_observed_data
import historymatching as hm

POPULATION = 5_000
SEED_INFECTIONS = 50

obs_incidence, _ = generate_observed_data(beta_true=0.7, gamma_true=0.4,
    population_size=POPULATION, n_seed_infections=SEED_INFECTIONS, seed=42)

def run_sir(samples):
    rows = []
    for _, row in samples.iterrows():
        m = SIR(beta=row['beta'], gamma=row['gamma'],
                s0=POPULATION-SEED_INFECTIONS, i0=SEED_INFECTIONS)
        inc = m.get_incidence()
        rows.append({'peak_incidence': float(inc.max()),
                     'attack_rate': float(inc.sum()/POPULATION)})
    return pd.DataFrame(rows)


def make_engine(output_dir):
    obs_peak = float(obs_incidence.max())
    obs_ar = float(obs_incidence.sum() / POPULATION)
    engine = hm.HistoryMatching(
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
    return engine


if __name__ == '__main__':
    output_dir = tempfile.mkdtemp(prefix="hm_test_")

    # ── Test 1: Checkpoint creation ──────────────────────────────────
    print("=" * 60)
    print("TEST 1: Checkpoint creation")
    print("=" * 60)

    engine = make_engine(output_dir)
    assert engine.run_dir is None, "output is created lazily, not at construction"

    result1 = engine.step()
    engine.commit_step()
    run_dir = engine.run_dir  # created on the first step()
    print(f"Run dir: {run_dir}")
    print(f"Wave 1 committed. Iteration: {engine.current_iteration}")

    wave1_dir = run_dir / "wave1"
    assert wave1_dir.exists(), "wave1/ not created"
    assert (run_dir / "checkpoint.pkl").exists(), "checkpoint.pkl not created"
    assert (run_dir / "run_config.json").exists(), "run_config.json not created"
    assert (wave1_dir / "convergence.png").exists(), "convergence.png missing"
    assert (wave1_dir / "nroy_samples.csv").exists(), "nroy_samples.csv missing"

    features_found = [d.name for d in wave1_dir.iterdir() if d.is_dir()]
    print(f"Feature dirs: {features_found}")
    for feat_dir in wave1_dir.iterdir():
        if feat_dir.is_dir():
            assert (feat_dir / "emulator.pkl").exists(), f"emulator.pkl missing in {feat_dir}"
            assert (feat_dir / "metrics.json").exists(), f"metrics.json missing in {feat_dir}"
            with open(feat_dir / "metrics.json") as f:
                metrics = json.load(f)
            print(f"  {feat_dir.name}: R2={metrics.get('r2', 'N/A')}")

    print("PASS\n")

    # ── Test 2: Resume ───────────────────────────────────────────────
    print("=" * 60)
    print("TEST 2: Resume from checkpoint")
    print("=" * 60)

    engine2 = make_engine(output_dir)
    results = engine2.run(resume=True)
    print(f"Resumed. Waves this call: {len(results)}, total: {engine2.current_iteration}")
    assert engine2.current_iteration == 2
    assert (run_dir / "wave2").exists(), "wave2/ not created"
    print("PASS\n")

    # ── Test 3: Serial NROY ──────────────────────────────────────────
    print("=" * 60)
    print("TEST 3: get_nroy_samples (serial)")
    print("=" * 60)

    nroy = engine2.get_nroy_samples()
    print(f"Default: {len(nroy)} samples")
    assert len(nroy) > 0

    nroy200 = engine2.get_nroy_samples(n=200)
    print(f"Requested 200: got {len(nroy200)}")
    assert len(nroy200) >= 100
    print("PASS\n")

    # ── Cleanup ──────────────────────────────────────────────────────
    shutil.rmtree(output_dir)
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
