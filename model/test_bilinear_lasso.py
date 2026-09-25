"""Regression checks: ../.venv/bin/python -m unittest test_bilinear_lasso."""
import contextlib
import io
import unittest
import warnings
from unittest.mock import patch

import jax.numpy as jnp
from jax import grad
import numpy as np

from bilinear_lasso import bilinear_lasso


class SolverTests(unittest.TestCase):
    def model(self, x=1., initial=.5):
        model = bilinear_lasso(np.array([[[x]]]), np.zeros((1, 1, 1)),
                               [1.], .01, .01)
        model.initialize_beta([initial], [0.])
        return model

    def fit(self, model, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return model.fit(max_iter=500, disturbance=0, **kwargs)

    def test_small_step_does_not_false_converge(self):
        model = self.fit(self.model(), step_size=1e-6)
        self.assertTrue(model.converged_)
        self.assertGreater(model.candidate_status[0]['iterations'], 1)
        self.assertLessEqual(model.stationarity_, 1e-5)
        self.assertLess(model.l, .011)

    def test_backtracking_prevents_overflow(self):
        model = self.fit(self.model(x=10., initial=1.), method="proximal")
        self.assertTrue(model.converged_)
        self.assertTrue(np.isfinite(model.l))
        self.assertLess(model.l, .004)
        losses = np.asarray(model.loss_history[0])
        self.assertTrue(np.all(np.diff(losses) <= 1e-7))

    def test_regular_start_and_final_trajectory(self):
        model = self.fit(self.model())
        self.assertTrue(model.converged_)
        np.testing.assert_array_equal(model.trajectory[0]['beta1'][-1], model.beta1)

    def test_zero_start_warns(self):
        with self.assertWarnsRegex(RuntimeWarning, 'exactly zero'), contextlib.redirect_stdout(io.StringIO()):
            model = self.model(initial=0.).fit(disturbance=0)
        self.assertEqual(model.l, .5)

    def test_iteration_limit_is_not_convergence(self):
        with self.assertWarnsRegex(RuntimeWarning, 'max_iter'), contextlib.redirect_stdout(io.StringIO()):
            model = self.model().fit(max_iter=1, step_size=1e-6, disturbance=.01)
        self.assertFalse(model.converged_)
        self.assertEqual(model.candidate_status[0]['status'], 'max_iter')

    def test_failed_candidate_cannot_win(self):
        starts = [jnp.full((1, 1), 1e100), jnp.zeros((1, 1)),
                  jnp.zeros((1, 1)), jnp.zeros((1, 1))]
        with patch('bilinear_lasso.random.normal', side_effect=starts), contextlib.redirect_stdout(io.StringIO()):
            model = self.model().fit(num_candidates=2, disturbance=1, max_iter=500)
        self.assertEqual(model.sel_idx, 1)
        self.assertEqual(model.candidate_status[0]['status'], 'nonfinite')
        self.assertTrue(np.isfinite(model.l))

    def test_all_failed_raises(self):
        with self.assertRaisesRegex(RuntimeError, 'All optimization candidates failed'):
            self.fit(self.model(initial=1e100))

    def test_repeated_fit_clears_diagnostics(self):
        model = self.fit(self.model(), num_candidates=2)
        self.fit(model)
        self.assertEqual(set(model.loss_history), {0})
        self.assertEqual(set(model.trajectory), {0})
        self.assertEqual(set(model.candidate_status), {0})

    def test_invalid_parameters(self):
        for kwargs in ({'num_candidates': 0}, {'step_size': 0}, {'tol': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.fit(self.model(), **kwargs)

    def test_gradient_and_zero_coordinate_stationarity(self):
        rng = np.random.default_rng(4)
        x1, x2 = rng.normal(size=(2, 3, 3, 5))
        y = rng.normal(size=5)
        b1, b2 = jnp.asarray(rng.normal(size=(2, 3, 1)))
        model = bilinear_lasso(x1, x2, y, .1, .2)
        residual = np.asarray(model.predict(b1, b2)) - y
        for x, b, g in zip((x1, x2), (b1, b2), grad(model.loss, (0, 1))(b1, b2)):
            expected = np.einsum('k,ijk,j->i', residual, x + x.transpose(1, 0, 2), np.asarray(b).ravel()) / 5
            np.testing.assert_allclose(np.asarray(g).ravel(), expected, atol=1e-6)
        zero = jnp.zeros((1, 1))
        self.assertAlmostEqual(model._stationarity(zero, zero, (zero + .3, zero), .1, .2), .2, places=6)


if __name__ == '__main__':
    unittest.main()
