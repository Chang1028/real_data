import time
import warnings

import jax.numpy as jnp
from jax import grad, random
import numpy as np


class bilinear_lasso:

    def __init__(self, input1, input2, y, lam1, lam2):
        """
        Bilinear regression model

        y_i =
            beta1^T X1_i beta1
            +
            beta2^T X2_i beta2
            +
            epsilon_i
        """

        self.X1 = jnp.asarray(input1)
        self.X2 = jnp.asarray(input2)
        self.y = jnp.asarray(y).ravel()

        self.lam1 = lam1
        self.lam2 = lam2

        self.p = input1.shape[0]
        self.n = input1.shape[2]

        self.beta1 = None
        self.beta2 = None

        self.l = None

        self.trajectory = {}
        self.loss_history = {}
        self.sel_idx = None


    # =========================================================
    # Initialization
    # =========================================================

    def initialize_beta(self, beta_1, beta_2):

        self.beta1 = jnp.asarray(beta_1).reshape(self.p, 1)
        self.beta2 = jnp.asarray(beta_2).reshape(self.p, 1)


    # =========================================================
    # Prediction
    # =========================================================

    def predict(self, beta_1, beta_2):

        b1 = beta_1.reshape(-1)
        b2 = beta_2.reshape(-1)

        pred1 = jnp.einsum(
            "i,ijk,j->k",
            b1,
            self.X1,
            b1
        )

        pred2 = jnp.einsum(
            "i,ijk,j->k",
            b2,
            self.X2,
            b2
        )

        return pred1 + pred2


    # =========================================================
    # Loss
    # =========================================================

    def loss(self, beta_1, beta_2):

        preds = self.predict(
            beta_1,
            beta_2
        )

        return 0.5 * jnp.mean(
            (preds - self.y) ** 2
        )


    def l1_loss(self, beta_1, beta_2):

        return (
            self.lam1 * jnp.sum(jnp.abs(beta_1))
            +
            self.lam2 * jnp.sum(jnp.abs(beta_2))
        )


    def loss_with_penalty(self, beta_1, beta_2):

        return (
            self.loss(beta_1, beta_2)
            +
            self.l1_loss(beta_1, beta_2)
        )


    # =========================================================
    # Proximal operator
    # =========================================================

    @staticmethod
    def prox(x, lam, step_size):
        """
        Soft-thresholding

        S_{eta*lambda}(x)
        """

        threshold = step_size * lam

        return (
            jnp.sign(x)
            *
            jnp.maximum(
                jnp.abs(x) - threshold,
                0.0
            )
        )


    # =========================================================
    # Fit
    # =========================================================

    @staticmethod
    def _stationarity(beta1, beta2, gradients, lam1, lam2):
        """Infinity norm of the minimum-norm L1 subgradient of F."""
        residuals = []
        for beta, gradient, lam in zip(
            (beta1, beta2), gradients, (lam1, lam2)
        ):
            residual = jnp.where(
                beta != 0,
                jnp.abs(gradient + lam * jnp.sign(beta)),
                jnp.maximum(jnp.abs(gradient) - lam, 0),
            )
            residuals.append(float(jnp.max(residual)))
        return max(residuals)

    def fit(
        self,
        num_candidates=1,
        max_iter=8000,
        step_size=0.1,
        tol=1e-5,
        disturbance=0.1,
        seed=2026
    ):
        """Fit by proximal gradient with sufficient-decrease backtracking.

        step_size is the initial trial step, adapted in both directions.
        tol bounds the infinity norm of the L1 stationarity residual, not
        coefficient movement. Stationarity does not imply a global minimum.
        candidate_status records convergence, iteration limits and numerical
        failures; converged_ and stationarity_ describe the selected candidate.
        """
        if self.beta1 is None or self.beta2 is None:
            raise ValueError("Call initialize_beta() before fit().")
        for name, value in (("num_candidates", num_candidates),
                            ("max_iter", max_iter)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in (("step_size", step_size), ("tol", tol)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (("disturbance", disturbance),
                            ("lam1", self.lam1), ("lam2", self.lam2)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if (self.X1.shape != (self.p, self.p, self.n)
                or self.X2.shape != self.X1.shape
                or self.y.shape != (self.n,) or self.n == 0 or self.p == 0):
            raise ValueError("Inputs must have shape (p, p, n), with y of length n.")
        if not all(np.all(np.isfinite(np.asarray(x))) for x in
                   (self.X1, self.X2, self.y, self.beta1, self.beta2)):
            raise ValueError("Data and initial coefficients must be finite.")
        if disturbance == 0 and any(
            np.all(np.asarray(b) == 0) for b in (self.beta1, self.beta2)
        ):
            warnings.warn(
                "An exactly zero coefficient block cannot leave zero under "
                "proximal gradient. Use nonzero initialization or disturbance > 0.",
                RuntimeWarning, stacklevel=2,
            )

        self.loss_history = {}
        self.trajectory = {}
        self.candidate_status = {}
        self.sel_idx = None
        self.l = None
        self.converged_ = False
        self.stationarity_ = np.inf
        results = {}
        master_key = random.PRNGKey(seed)
        gradient_fn = grad(self.loss, (0, 1))
        save_every = max(1, max_iter // 100)

        for rs in range(num_candidates):
            start_time = time.time()
            master_key, key1, key2 = random.split(master_key, 3)
            beta1 = self.beta1 + disturbance * random.normal(key1, (self.p, 1))
            beta2 = self.beta2 + disturbance * random.normal(key2, (self.p, 1))
            self.loss_history[rs] = []
            trajectory = {"beta1": [], "beta2": []}
            current_loss = float(self.loss_with_penalty(beta1, beta2))
            gradients = gradient_fn(beta1, beta2)
            step = step_size
            status = "max_iter"
            residual = np.inf
            updates = 0

            for iteration in range(max_iter):
                if not np.isfinite(current_loss) or not all(
                    np.all(np.isfinite(np.asarray(g))) for g in gradients
                ):
                    status = "nonfinite"
                    break
                residual = self._stationarity(
                    beta1, beta2, gradients, self.lam1, self.lam2
                )
                if residual <= tol:
                    status = "converged"
                    break

                # Proximal gradient without convex-only FISTA momentum.
                # Require F(new) <= F(old) - ||new-old||^2 / (4*step).
                # Allow objective roundoff, but never relax stationarity.
                # This also rejects overflowing trial points.
                roundoff = 4 * np.finfo(np.asarray(beta1).dtype).eps * max(
                    abs(current_loss), np.finfo(np.asarray(beta1).dtype).tiny
                )
                accepted = False
                for _ in range(80):
                    new1 = self.prox(beta1 - step * gradients[0], self.lam1, step)
                    new2 = self.prox(beta2 - step * gradients[1], self.lam2, step)
                    displacement = sum(
                        float(jnp.sum((new - old) ** 2))
                        for new, old in ((new1, beta1), (new2, beta2))
                    )
                    if displacement == 0:
                        status = "stalled"
                        break
                    trial_loss = float(self.loss_with_penalty(new1, new2))
                    if (np.isfinite(trial_loss) and np.isfinite(displacement)
                            and trial_loss <= current_loss - displacement / (4 * step) + roundoff):
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    if status != "stalled":
                        status = "line_search_failed"
                    break

                beta1, beta2 = new1, new2
                current_loss = trial_loss
                updates += 1
                self.loss_history[rs].append(current_loss)
                if iteration % save_every == 0:
                    trajectory["beta1"].append(beta1)
                    trajectory["beta2"].append(beta2)
                gradients = gradient_fn(beta1, beta2)
                # Allow recovery from an unnecessarily small initial step.
                if displacement / (4 * step) > roundoff:
                    step = min(step * 2, np.finfo(float).max / 4)

            finite = np.isfinite(current_loss) and all(
                np.all(np.isfinite(np.asarray(x)))
                for x in (beta1, beta2, *gradients)
            )
            if finite:
                residual = self._stationarity(
                    beta1, beta2, gradients, self.lam1, self.lam2
                )
                if residual <= tol:
                    status = "converged"
            else:
                status = "nonfinite"
            self.candidate_status[rs] = {
                "status": status, "iterations": updates,
                "stationarity": residual,
            }
            # Always save the returned endpoint, including a zero-update run.
            for name, beta in (("beta1", beta1), ("beta2", beta2)):
                trajectory[name].append(beta)
                trajectory[name] = jnp.stack(trajectory[name])
            self.trajectory[rs] = trajectory
            # A numerical failure must never beat a valid candidate.
            if finite and status in ("converged", "max_iter"):
                results[rs] = {"loss": current_loss, "beta1": beta1, "beta2": beta2}
            print(f"Initialization {rs + 1}/{num_candidates}: {status}; "
                  f"loss={current_loss:.6f}; stationarity={residual:.3g}; "
                  f"{time.time() - start_time:.2f} seconds")

        if not results:
            raise RuntimeError(
                "All optimization candidates failed. Inspect candidate_status; "
                "consider rescaling data or using higher precision."
            )
        self.sel_idx = min(results, key=lambda k: results[k]["loss"])
        selected = results[self.sel_idx]
        self.beta1, self.beta2 = selected["beta1"], selected["beta2"]
        self.l = selected["loss"]
        diagnostics = self.candidate_status[self.sel_idx]
        self.converged_ = diagnostics["status"] == "converged"
        self.stationarity_ = diagnostics["stationarity"]
        if not self.converged_:
            warnings.warn(
                "Selected candidate reached max_iter without satisfying the "
                "stationarity tolerance.", RuntimeWarning, stacklevel=2,
            )
        return self
