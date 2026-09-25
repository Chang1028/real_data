"""Save complete simulation experiments and reload them without refitting.

Pickle archives are intended only for locally generated, trusted results.
No live JAX model or external data paths are required to recreate saved plots.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil
import platform
from importlib.metadata import version
import uuid

import numpy as np
import pandas as pd

from bilinear_lasso import bilinear_lasso
from config import PARAMS
from cross_validation import run_cv_over_lambdas
from data_utils import generate_simulation_data

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results'


def betas_from_blocks(p, beta1_blocks, beta2_blocks):
    """Build betas from (start, stop, value) tuples, using 0-based [start, stop).

    Unspecified coefficients are zero. Overlapping blocks are rejected.
    """
    def build(blocks):
        beta = np.zeros((p, 1), dtype=np.float64)
        used = np.zeros(p, dtype=bool)
        for start, stop, value in blocks:
            if (not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer))
                    or not 0 <= start < stop <= p or not np.isfinite(value)):
                raise ValueError(f'Invalid block {(start, stop, value)} for p={p}.')
            if used[start:stop].any():
                raise ValueError('ROI blocks must not overlap within one beta vector.')
            beta[start:stop] = value
            used[start:stop] = True
        return beta
    return build(beta1_blocks), build(beta2_blocks)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=_json_value) + '\n')


def _numpy_state(value):
    if isinstance(value, dict):
        return {k: _numpy_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numpy_state(v) for v in value]
    if hasattr(value, 'shape'):
        return np.asarray(value)
    return value


def run_simulation_experiment(
    name, beta1_true=None, beta2_true=None, *, parameters=None, seed=2026,
    results_dir=RESULTS_DIR, final_lambda_pair=None, block_definitions=None,
    rs_file='z_fc.npy', emo_file='z_sc.npy', save_connectivity=True,
):
    """Generate, cross-validate, fit, and save one independently named experiment.

    parameters overrides entries from parameters.json for THIS run only.
    With final_lambda_pair=None, choose the pair with the lowest mean CV
    validation MSE, breaking ties by lambda1 then lambda2. Ground truth is
    never used to select penalties or initialize the fitted betas.

    Every invocation creates a new folder. Completed CV is saved before the
    final fit, and a failure leaves partial files plus an explicit failed status.
    save_connectivity=True makes the run self-contained for connectivity plots;
    False omits the large X arrays but retains coefficient/prediction diagnostics.
    """
    if final_lambda_pair is not None:
        final_lambda_pair = tuple(final_lambda_pair)
        if len(final_lambda_pair) != 2 or any(not np.isfinite(v) or v < 0 for v in final_lambda_pair):
            raise ValueError('final_lambda_pair must contain two finite, nonnegative penalties.')
    params = dict(PARAMS)
    params.update(parameters or {})
    x1, x2, y, true1, true2 = generate_simulation_data(
        data_dir=params['DATA_DIR'], p=params['P'], n=params['N_SUBJECTS'],
        sparsity1=params['SPARSITY1'], sparsity2=params['SPARSITY2'],
        snr=params['SNR'], seed=seed, rs_file=rs_file, emo_file=emo_file,
        beta1_true=beta1_true, beta2_true=beta2_true,
    )
    fingerprint = hashlib.sha256(true1.tobytes() + true2.tobytes()).hexdigest()[:12]
    slug = re.sub(r'[^A-Za-z0-9_-]+', '_', str(name)).strip('_') or 'simulation'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = Path(results_dir) / f'{slug}_{fingerprint}_{stamp}_{uuid.uuid4().hex[:6]}'
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'figures').mkdir()
    metadata = {
        'schema_version': 1, 'name': str(name), 'created_utc': stamp,
        'status': 'running', 'stage': 'saving_data', 'beta_fingerprint': fingerprint,
        'seed': int(seed), 'parameters': params, 'block_definitions': block_definitions,
        'versions': {'python': platform.python_version(), **{pkg: version(pkg) for pkg in ('numpy', 'scipy', 'jax', 'pandas', 'scikit-learn', 'joblib')}},
        'rs_file': rs_file, 'emo_file': emo_file, 'connectivity_saved': bool(save_connectivity),
        'actual_p': x1.shape[0], 'actual_n': x1.shape[2],
        'preprocessing': 'X2: log10(1+X2), normalize by transformed diagonal, zero diagonal; X1 unchanged.',
        'centering': 'CV uses training-fold means; final model uses full-sample means.',
        'final_selection': 'explicit lambda pair' if final_lambda_pair is not None else 'minimum mean CV validation MSE',
    }
    print(f'Saving simulation experiment to {run_dir}', flush=True)
    try:
        _write_json(run_dir / 'metadata.json', metadata)
        source_dir = run_dir / 'source'
        source_dir.mkdir()
        for filename in ('bilinear_lasso.py', 'cross_validation.py', 'data_utils.py',
                         'evaluation.py', 'plot_utils.py', 'config.py', 'simulation_results.py'):
            shutil.copy2(BASE_DIR / filename, source_dir / filename)
        # config.py expects the original single-object-in-list JSON format.
        _write_json(source_dir / 'parameters.json', [params])
        mean1, mean2, y_mean = x1.mean(2, keepdims=True), x2.mean(2, keepdims=True), float(y.mean())
        arrays = dict(beta1_true=true1, beta2_true=true2, y=y, y_centered=y-y_mean,
                      subject_index=np.arange(len(y)))
        if save_connectivity:
            arrays.update(X1=x1, X2=x2)
        np.savez_compressed(run_dir / 'simulation_data.npz', **arrays)
        metadata['stage'] = 'cross_validation'
        _write_json(run_dir / 'metadata.json', metadata)
        cv_results, cv_summary = run_cv_over_lambdas(
            x1, x2, y, seed=seed,
            lambda1_grid=params['LAMBDA1_GRID'], lambda2_grid=params['LAMBDA2_GRID'],
            n_jobs=params['N_JOBS'], n_splits=params['N_SPLITS'],
            max_iter=params['MAX_ITER'], step_size=params['STEP_SIZE'], tol=params['TOL'],
            num_candidates=params.get('NUM_CANDIDATES', 2),
            initialization=params.get('INITIALIZATION', 'scaled_random'),
            require_convergence=True,
        )
        with (run_dir / 'cv_results.pkl').open('wb') as stream:
            pickle.dump(_numpy_state(cv_results), stream, protocol=pickle.HIGHEST_PROTOCOL)
        cv_summary.to_csv(run_dir / 'cv_summary.csv', index=False)
        if final_lambda_pair is None:
            best = cv_summary.sort_values(['Test MSE', 'lambda1', 'lambda2']).iloc[0]
            final_lambda_pair = (float(best['lambda1']), float(best['lambda2']))
        lam1, lam2 = map(float, final_lambda_pair)
        metadata.update(stage='final_fit', final_lambda_pair=[lam1, lam2])
        _write_json(run_dir / 'metadata.json', metadata)
        model = bilinear_lasso(x1-mean1, x2-mean2, y-y_mean, lam1, lam2)
        model.initialize_beta(np.zeros_like(true1), np.zeros_like(true2))
        model.fit(num_candidates=params.get('NUM_CANDIDATES', 2), max_iter=params['MAX_ITER'],
                  step_size=params['STEP_SIZE'], tol=params['TOL'], disturbance=2, seed=seed,
                  initialization=params.get('INITIALIZATION', 'scaled_random'), require_convergence=True)
        predictions = np.asarray(model.predict(model.beta1, model.beta2))
        np.savez_compressed(
            run_dir / 'final_model.npz', beta1=np.asarray(model.beta1), beta2=np.asarray(model.beta2),
            y_true=y, y_pred=predictions+y_mean, y_true_centered=y-y_mean,
            y_pred_centered=predictions, X1_mean=mean1, X2_mean=mean2, y_mean=y_mean,
            lambda1=lam1, lambda2=lam2, selected_candidate=model.sel_idx,
            converged=model.converged_, stationarity=model.stationarity_, objective=model.l,
        )
        history = _numpy_state({
            'loss_history': model.loss_history, 'stationarity_history': model.stationarity_history,
            'trajectory': model.trajectory, 'candidate_status': model.candidate_status,
            'selected_candidate': model.sel_idx,
        })
        with (run_dir / 'model_history.pkl').open('wb') as stream:
            pickle.dump(history, stream, protocol=pickle.HIGHEST_PROTOCOL)
        _write_json(run_dir / 'model_diagnostics.json', model.candidate_status)
        metadata.update(status='complete', stage='complete')
        _write_json(run_dir / 'metadata.json', metadata)
    except Exception as error:
        metadata.update(status='failed', error_type=type(error).__name__, error=str(error))
        _write_json(run_dir / 'metadata.json', metadata)
        raise RuntimeError(f'Simulation failed during {metadata["stage"]}; partial results: {run_dir}') from error
    return run_dir


def load_simulation_experiment(run_dir, *, load_connectivity=False):
    """Load locally generated, trusted results for plotting; performs no fitting."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / 'metadata.json').read_text())
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported experiment schema version.')
    if metadata['status'] != 'complete':
        raise ValueError(f'Run is {metadata["status"]}, not complete; inspect metadata.json for its stage/error.')
    with np.load(run_dir / 'simulation_data.npz', allow_pickle=False) as archive:
        simulation = {k: archive[k] for k in archive.files
                      if load_connectivity or k not in ('X1', 'X2')}
    with np.load(run_dir / 'final_model.npz', allow_pickle=False) as archive:
        model = {k: archive[k] for k in archive.files}
    with (run_dir / 'cv_results.pkl').open('rb') as stream:
        cv_results = pickle.load(stream)
    with (run_dir / 'model_history.pkl').open('rb') as stream:
        history = pickle.load(stream)
    return dict(path=run_dir, metadata=metadata, simulation=simulation, model=model,
                history=history, cv_results=cv_results,
                cv_summary=pd.read_csv(run_dir / 'cv_summary.csv'))


def plot_saved_betas(saved, save_path=None):
    """Overlay ground truth with final estimates, sign-aligned for display only."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for block, ax in enumerate(axes, 1):
        true = saved['simulation'][f'beta{block}_true'].ravel()
        estimated = saved['model'][f'beta{block}'].ravel()
        aligned = estimated * (-1 if estimated @ true < 0 else 1)
        ax.plot(true, color='black', linewidth=2, drawstyle='steps-mid', label='True beta')
        ax.plot(aligned, color='tab:orange', linewidth=1, label='Estimated beta (sign-aligned)')
        ax.axhline(0, color='gray', linewidth=.7, linestyle=':')
        ax.set(title=rf'$\beta_{block}$', ylabel='Coefficient')
        ax.legend()
    axes[-1].set_xlabel('ROI index (0-based)')
    fig.suptitle(
        f"{saved['metadata']['name']} | "
        f"final $\\lambda_1$={float(saved['model']['lambda1']):g}, "
        f"$\\lambda_2$={float(saved['model']['lambda2']):g}"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig, axes
