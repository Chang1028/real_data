"""Pipeline checks, including the original (unsplit) KKT conditions."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from bilinear_lasso import bilinear_lasso
from cross_validation import run_one_cv_fold, run_cv_over_lambdas
from data_utils import preprocess_X2, load_connectivity_array, generate_simulation_data


def problem(p=5, n=100):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(2, p, p, n))
    x = (x + x.transpose(0, 2, 1, 3)) / 2
    beta1, beta2 = np.zeros(p), np.zeros(p)
    beta1[:2] = [1.2, -.8]
    beta2[:2] = [.7, 1.1]
    y = np.einsum('i,ijk,j->k', beta1, x[0], beta1) + np.einsum('i,ijk,j->k', beta2, x[1], beta2)
    return x[0], x[1], y


def independent_residual(x1, x2, y, b1, b2, lam1, lam2):
    b1, b2 = np.asarray(b1).ravel(), np.asarray(b2).ravel()
    error = (np.einsum('i,ijk,j->k', b1, x1, b1)
             + np.einsum('i,ijk,j->k', b2, x2, b2) - y)
    residuals = []
    for x, b, lam in ((x1, b1, lam1), (x2, b2, lam2)):
        gradient = np.einsum('k,ijk,j->i', error, x + x.transpose(1, 0, 2), b) / len(y)
        residuals.append(np.max(np.where(b != 0, abs(gradient + lam * np.sign(b)), np.maximum(abs(gradient) - lam, 0))))
    return max(residuals)


class PipelineTests(unittest.TestCase):
    def test_nonlinear_fit_meets_independent_kkt_and_predicts_signal(self):
        x1, x2, y = problem()
        model = bilinear_lasso(x1, x2, y, .001, .001)
        model.initialize_beta(np.zeros(5), np.zeros(5))
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(num_candidates=2, max_iter=1000, disturbance=1,
                      initialization='scaled_random', require_convergence=True)
        residual = independent_residual(x1, x2, y, model.beta1, model.beta2, .001, .001)
        self.assertLessEqual(residual, 1e-5)
        self.assertAlmostEqual(residual, model.stationarity_, places=10)
        self.assertLess(np.mean((np.asarray(model.predict(model.beta1, model.beta2)) - y)**2), .001)
        self.assertEqual(str(model.beta1.dtype), 'float64')
        trajectory = model.trajectory[model.sel_idx]
        np.testing.assert_array_equal(trajectory['beta1'][-1], model.beta1)
        self.assertEqual(trajectory['iterations'][-1], model.candidate_status[model.sel_idx]['iterations'])

    def test_scipy_success_cannot_override_stationarity(self):
        model = bilinear_lasso(np.ones((1, 1, 1)), np.zeros((1, 1, 1)), [1.], .01, .01)
        model.initialize_beta([.5], [0.])
        def fake_minimize(fun, x0, **kwargs):
            return OptimizeResult(x=x0, success=True, message='small objective change', nit=0)
        with patch('bilinear_lasso.minimize', side_effect=fake_minimize), contextlib.redirect_stdout(io.StringIO()):
            with self.assertWarnsRegex(RuntimeWarning, 'exactly zero'):
                with self.assertRaisesRegex(RuntimeError, 'No candidate satisfied stationarity'):
                    model.fit(max_iter=1, step_size=1e-6, disturbance=0, require_convergence=True)
        self.assertFalse(model.converged_)
        self.assertGreater(model.stationarity_, 1e-5)

    def test_float64_large_finite_start_is_not_rejected_as_float32_overflow(self):
        model = bilinear_lasso(np.ones((1, 1, 1)), np.zeros((1, 1, 1)), [1.], .01, .01)
        model.initialize_beta([1e20], [0.])
        with contextlib.redirect_stdout(io.StringIO()), self.assertWarnsRegex(RuntimeWarning, 'exactly zero'):
            model.fit(max_iter=500, disturbance=0, require_convergence=True)
        self.assertTrue(model.converged_)

    def test_parallel_cv_retains_precision_and_diagnostics(self):
        x1, x2, y = problem(p=3, n=40)
        with contextlib.redirect_stdout(io.StringIO()):
            results, summary = run_cv_over_lambdas(
                x1, x2, y, lambda1_grid=[.01, .02], lambda2_grid=[.01],
                n_splits=2, n_jobs=2, num_candidates=1, max_iter=1000,
            )
        self.assertTrue(summary['All Converged'].all())
        self.assertLessEqual(summary['Max Stationarity'].max(), 1e-5)
        for result in results:
            for fold in result['folds']:
                self.assertEqual(fold['dtype'], 'float64')
                self.assertTrue(fold['converged'])
                self.assertIn('candidate_status', fold)

    def test_heldout_values_do_not_affect_fit_or_centering(self):
        x1, x2, y = problem(p=3, n=40)
        train, val = np.arange(30), np.arange(30, 40)
        options = dict(train_idx=train, val_idx=val, lam1=.01, lam2=.01,
                       p=3, max_iter=1000, step_size=1e-6, tol=1e-5, num_candidates=1)
        with contextlib.redirect_stdout(io.StringIO()):
            a = run_one_cv_fold(x1, x2, y, **options)
            altered_x1, altered_y = x1.copy(), y.copy()
            altered_x1[:, :, val] += 1000
            altered_y[val] += 500
            b = run_one_cv_fold(altered_x1, x2, altered_y, **options)
        np.testing.assert_array_equal(a['beta1'], b['beta1'])
        np.testing.assert_array_equal(a['beta2'], b['beta2'])
        np.testing.assert_allclose(a['y_true'], y[val] - y[train].mean())

    def test_log10_normalization_and_zero_diagonal(self):
        x = np.array([[[9., 0.], [99., 1.]], [[99., 1.], [999., 9.]]])
        before = x.copy()
        expected = np.array([[[0., 0.], [2/np.sqrt(3), 0.]], [[2/np.sqrt(3), 0.], [0., 0.]]])
        np.testing.assert_allclose(preprocess_X2(x), expected)
        np.testing.assert_array_equal(x, before)
        for invalid in (np.full((2, 2, 1), np.nan), np.full((2, 2, 1), -2.), np.full((2, 2, 1), -.5)):
            with self.assertRaises(ValueError):
                preprocess_X2(invalid)

    def test_fixed_simulation_blocks_and_input_precision_are_preserved(self):
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as tmp:
            x1 = rng.normal(size=(6, 160, 160))
            x2 = rng.uniform(.1, 1., size=(6, 160, 160))
            np.save(Path(tmp) / 'x1.npy', x1)
            np.save(Path(tmp) / 'x2.npy', x2)
            loaded = load_connectivity_array(Path(tmp) / 'x1.npy', 160)
            self.assertEqual(loaded.dtype, np.float64)
            np.testing.assert_array_equal(loaded, x1.transpose(1, 2, 0))
            result = generate_simulation_data(tmp, 160, 6, .01, .99, rs_file='x1.npy', emo_file='x2.npy')
        beta1, beta2 = result[-2:]
        expected1, expected2 = np.zeros((160, 1)), np.zeros((160, 1))
        expected1[40:60], expected1[120:150] = 1.5, -1.
        expected2[40:60], expected2[120:150] = 1., -1.2
        np.testing.assert_array_equal(beta1, expected1)
        np.testing.assert_array_equal(beta2, expected2)
        self.assertTrue(np.isfinite(result[2]).all())


if __name__ == '__main__':
    unittest.main()
