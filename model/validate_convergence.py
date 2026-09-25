"""Reproduce the real-data convergence audit using parameters.json.

Run ../.venv/bin/python validate_convergence.py [--simulation] from this folder.
Writes a separate audit directory; does not replace the notebook's older results.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from bilinear_lasso import bilinear_lasso
from config import (PARAMS, DATA_DIR, P, N_SUBJECTS, MAX_ITER, STEP_SIZE, TOL,
                    NUM_CANDIDATES, INITIALIZATION, SPARSITY1, SPARSITY2, SNR)
from cross_validation import run_cv_over_lambdas, run_one_lambda_pair_cv
from data_utils import load_fmri_data, generate_simulation_data


def fold_diagnostics(results):
    keys = ('lambda1', 'lambda2', 'converged', 'stationarity', 'iterations',
            'selected_candidate', 'candidate_status', 'objective', 'dtype')
    return [dict({k: f[k] for k in keys}, fold=i + 1)
            for result in results for i, f in enumerate(result['folds'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation', action='store_true', help='Also validate five simulation folds at lambda=(1, 1).')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results' / 'convergence_audit')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'parameters_used.json').write_text(json.dumps(PARAMS, indent=2) + '\n')
    x1, x2, _, age = load_fmri_data(
        DATA_DIR, P, N_SUBJECTS, rs_file='z_fc.npy', emo_file='z_sc.npy',
        y_file='y.npy', age_file='age.npy',
    )
    start = time.monotonic()
    results, summary = run_cv_over_lambdas(x1, x2, age, seed=2026, require_convergence=True)
    summary.to_csv(out / 'real_age_cv_summary.csv', index=False)
    (out / 'real_age_cv_diagnostics.json').write_text(json.dumps({
        'elapsed_seconds': time.monotonic() - start, 'folds': fold_diagnostics(results),
    }, indent=2) + '\n')
    model = bilinear_lasso(x1 - x1.mean(2, keepdims=True), x2 - x2.mean(2, keepdims=True), age - age.mean(), 4, 4)
    model.initialize_beta(np.zeros((P, 1)), np.zeros((P, 1)))
    model.fit(num_candidates=NUM_CANDIDATES, max_iter=MAX_ITER, step_size=STEP_SIZE,
              tol=TOL, disturbance=2, seed=2026, initialization=INITIALIZATION,
              require_convergence=True)
    np.savez(out / 'final_age_fit.npz', beta1=np.asarray(model.beta1), beta2=np.asarray(model.beta2),
             y_pred=np.asarray(model.predict(model.beta1, model.beta2)))
    (out / 'final_age_fit_diagnostics.json').write_text(json.dumps(model.candidate_status, indent=2) + '\n')
    if args.simulation:
        x1, x2, y, _, _ = generate_simulation_data(
            DATA_DIR, P, N_SUBJECTS, SPARSITY1, SPARSITY2, SNR,
            rs_file='z_fc.npy', emo_file='z_sc.npy',
        )
        start = time.monotonic()
        result = run_one_lambda_pair_cv(1, 1, x1, x2, y, seed=2026)
        (out / 'simulation_cv_diagnostics.json').write_text(json.dumps({
            'lambda_pair': [1, 1], 'elapsed_seconds': time.monotonic() - start,
            'folds': fold_diagnostics([result]),
        }, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(f'Convergence audit saved to {out}')


if __name__ == '__main__':
    main()
