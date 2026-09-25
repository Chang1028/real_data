import time
import warnings

from jax import config, jit, random, value_and_grad
# This module is imported independently by joblib workers.
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
import numpy as np


def _smooth_loss(beta, X1, X2, y):
    p = X1.shape[1]
    prediction = (jnp.einsum("i,nij,j->n", beta[:p], X1, beta[:p])
                  + jnp.einsum("i,nij,j->n", beta[p:], X2, beta[p:]))
    return 0.5 * jnp.mean((prediction - y) ** 2)


# Compile once per input shape, rather than tracing each gradient on each step.
_smooth_value_grad = jit(value_and_grad(_smooth_loss))


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

        self.X1 = jnp.asarray(input1, dtype=jnp.float64)
        self.X2 = jnp.asarray(input2, dtype=jnp.float64)
        self.y = jnp.asarray(y, dtype=jnp.float64).ravel()

        self.lam1 = lam1
        self.lam2 = lam2

        if self.X1.ndim != 3:
            raise ValueError("Inputs must have shape (p, p, n).")
        self.p = self.X1.shape[0]
        self.n = self.X1.shape[2]

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

        self.beta1 = jnp.asarray(beta_1, dtype=jnp.float64).reshape(self.p, 1)
        self.beta2 = jnp.asarray(beta_2, dtype=jnp.float64).reshape(self.p, 1)


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
        self, num_candidates=1, max_iter=8000, step_size=0.1, tol=1e-5,
        disturbance=0.1, seed=2026, method="lbfgsb",
        initialization="provided", require_convergence=False,
    ):
        """Minimize the original quadratic prediction loss plus exact L1 penalty.

        L-BFGS-B optimizes beta = u-v, u,v >= 0, with penalty lambda*(u+v).
        Minimizing over u,v is equivalent to the original L1 problem. JIT-compiled
        float64 gradients are shared by this solver and the proximal fallback.
        A SciPy success flag or small objective change NEVER establishes
        convergence: the original L1 stationarity residual must be <= tol.

        initialization='provided' preserves initialize_beta + disturbance.
        'scaled_random' rescales each random block to the training outcome scale;
        it is intended for zero-initialized models. No validation data are used.
        method='proximal' retains proximal gradient for comparison. step_size
        controls that method and any proximal recovery after L-BFGS-B stops early.
        max_iter is the total update budget PER candidate, including recovery.
        require_convergence raises when no candidate meets tol. Even convergence
        only certifies stationarity, not a global minimum or coefficient recovery.
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

        if method not in ("lbfgsb", "proximal"):
            raise ValueError("method must be 'lbfgsb' or 'proximal'.")
        if initialization not in ("provided", "scaled_random"):
            raise ValueError("initialization must be 'provided' or 'scaled_random'.")
        self.loss_history = {}
        self.stationarity_history = {}
        self.trajectory = {}
        self.candidate_status = {}
        self.sel_idx = None
        self.l = None
        self.converged_ = False
        self.stationarity_ = np.inf
        results = {}
        master_key = random.PRNGKey(seed)
        save_every = max(1, max_iter // 100)
        X1 = jnp.transpose(self.X1, (2, 0, 1))
        X2 = jnp.transpose(self.X2, (2, 0, 1))
        penalties = np.repeat([self.lam1, self.lam2], self.p)

        def evaluate(beta):
            loss, gradient = _smooth_value_grad(jnp.asarray(beta), X1, X2, self.y)
            return float(loss) + float(penalties @ np.abs(beta)), np.asarray(gradient)

        def stationarity(beta, gradient):
            residual = np.where(
                beta != 0, np.abs(gradient + penalties * np.sign(beta)),
                np.maximum(np.abs(gradient) - penalties, 0),
            )
            return float(np.max(residual))

        for rs in range(num_candidates):
            start_time = time.monotonic()
            master_key, key1, key2 = random.split(master_key, 3)
            blocks = [np.asarray(base + disturbance * random.normal(key, (self.p, 1))).ravel()
                      for base, key in ((self.beta1, key1), (self.beta2, key2))]
            if initialization == "scaled_random":
                target_sd = float(np.std(np.asarray(self.y))) / np.sqrt(2)
                for k, X in enumerate((self.X1, self.X2)):
                    prediction = np.asarray(jnp.einsum("i,ijk,j->k", blocks[k], X, blocks[k]))
                    sd = float(np.std(prediction))
                    if sd > 0 and target_sd > 0:
                        blocks[k] = blocks[k] * np.sqrt(target_sd / sd)
            beta = np.concatenate(blocks)
            loss, gradient = evaluate(beta)
            initial_loss = loss
            updates = 0
            history, residual_history = [], []
            trajectory = {"beta1": [], "beta2": [], "iterations": []}
            messages = []
            step = step_size

            def record(b, f, g):
                history.append(float(f))
                residual_history.append(stationarity(b, g))
                if updates % save_every == 0:
                    trajectory["beta1"].append(b[:self.p].reshape(self.p, 1).copy())
                    trajectory["beta2"].append(b[self.p:].reshape(self.p, 1).copy())
                    trajectory["iterations"].append(updates)

            record(beta, loss, gradient)
            status = "max_iter"
            # Recanonicalize u,v and reset curvature after premature termination.
            # At most three L-BFGS-B runs share the SAME max_iter budget.
            if method == "lbfgsb":
                for restart in range(3):
                    if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                        status = "nonfinite"
                        break
                    if stationarity(beta, gradient) <= tol or updates >= max_iter:
                        break
                    size = beta.size
                    z0 = np.r_[np.maximum(beta, 0), np.maximum(-beta, 0)]
                    cache = {}

                    def objective(z):
                        b = z[:size] - z[size:]
                        f, g = evaluate(b)
                        # Account for simultaneous positive and negative parts.
                        split_f = f + penalties @ (z[:size] + z[size:] - np.abs(b))
                        cache.update(z=z.copy(), beta=b, loss=f, gradient=g)
                        return split_f, np.r_[g + penalties, -g + penalties]

                    def callback(z):
                        nonlocal updates
                        if not np.array_equal(z, cache.get("z")):
                            objective(z)
                        updates += 1
                        record(cache["beta"], cache["loss"], cache["gradient"])
                        if residual_history[-1] <= tol:
                            raise StopIteration

                    result = minimize(
                        objective, z0, jac=True, method="L-BFGS-B",
                        bounds=[(0, None)] * z0.size, callback=callback,
                        options={"maxiter": max_iter - updates, "maxfun": 50 * (max_iter - updates) + 1,
                                 "gtol": tol * 0.1, "ftol": 0.0, "maxls": 50, "maxcor": 30},
                    )
                    proposed = result.x[:size] - result.x[size:]
                    proposed_loss, proposed_gradient = evaluate(proposed)
                    messages.append(
                        "Original L1 stationarity tolerance reached"
                        if stationarity(proposed, proposed_gradient) <= tol
                        else str(result.message)
                    )
                    if not np.isfinite(proposed_loss) or not np.all(np.isfinite(proposed_gradient)):
                        status = "nonfinite"
                        break
                    # Never replace a finite starting point with a worse endpoint.
                    if proposed_loss > loss + 16 * np.finfo(float).eps * max(1, abs(loss)):
                        break
                    beta, loss, gradient = proposed, proposed_loss, proposed_gradient
                    if stationarity(beta, gradient) <= tol:
                        break
                    # Take a proximal step to refresh zero coordinates before retrying.
                    if updates < max_iter:
                        proposal = self._proximal_step(beta, loss, gradient, penalties, evaluate, step)
                        if proposal is None:
                            status = "stalled"
                            break
                        beta, loss, gradient, step = proposal
                        updates += 1
                        record(beta, loss, gradient)

            # Finish remaining budget with proximal gradient if quasi-Newton
            # terminated early. Never call a small movement 'convergence'.
            while updates < max_iter and status != "nonfinite":
                if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                    status = "nonfinite"
                    break
                if stationarity(beta, gradient) <= tol:
                    break
                proposal = self._proximal_step(beta, loss, gradient, penalties, evaluate, step)
                if proposal is None:
                    status = "stalled"
                    break
                beta, loss, gradient, step = proposal
                updates += 1
                record(beta, loss, gradient)

            finite = np.isfinite(loss) and np.all(np.isfinite(beta)) and np.all(np.isfinite(gradient))
            residual = stationarity(beta, gradient) if finite else np.inf
            if not finite:
                status = "nonfinite"
            elif residual <= tol:
                status = "converged"
            elif updates >= max_iter:
                status = "max_iter"
            else:
                status = "stalled"
            self.candidate_status[rs] = {
                "status": status, "iterations": updates, "stationarity": residual,
                "loss": loss, "initial_loss": initial_loss,
                "elapsed_seconds": time.monotonic() - start_time,
                "solver_messages": messages,
            }
            # Include the exact endpoint even if no updates were possible.
            if (trajectory["iterations"][-1] != updates
                    or not np.array_equal(trajectory["beta1"][-1].ravel(), beta[:self.p])
                    or not np.array_equal(trajectory["beta2"][-1].ravel(), beta[self.p:])):
                trajectory["beta1"].append(beta[:self.p].reshape(self.p, 1).copy())
                trajectory["beta2"].append(beta[self.p:].reshape(self.p, 1).copy())
                trajectory["iterations"].append(updates)
            self.trajectory[rs] = {key: np.asarray(value) for key, value in trajectory.items()}
            self.loss_history[rs] = history
            self.stationarity_history[rs] = residual_history
            if finite and status in ("converged", "max_iter"):
                results[rs] = {"loss": loss, "beta": beta.copy()}
            print(f"Initialization {rs + 1}/{num_candidates}: {status}; loss={loss:.6f}; "
                  f"stationarity={residual:.3g}; iterations={updates}; "
                  f"{time.monotonic() - start_time:.2f} seconds", flush=True)

        if not results:
            raise RuntimeError(f"All optimization candidates failed: {self.candidate_status}")
        converged = [k for k in results if self.candidate_status[k]["status"] == "converged"]
        pool = converged or list(results)
        self.sel_idx = min(pool, key=lambda k: results[k]["loss"])
        selected = results[self.sel_idx]
        self.beta1 = jnp.asarray(selected["beta"][:self.p].reshape(self.p, 1))
        self.beta2 = jnp.asarray(selected["beta"][self.p:].reshape(self.p, 1))
        self.l = selected["loss"]
        self.converged_ = bool(converged)
        self.stationarity_ = self.candidate_status[self.sel_idx]["stationarity"]
        if not self.converged_:
            message = (f"No candidate satisfied stationarity <= {tol:g}; "
                       f"selected candidate reached max_iter: {self.candidate_status}")
            if require_convergence:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return self

    @staticmethod
    def _proximal_step(beta, loss, gradient, penalties, evaluate, step):
        """Backtracking step; recover from an initial step too small to move."""
        roundoff = 4 * np.finfo(float).eps * max(abs(loss), np.finfo(float).tiny)
        for _ in range(80):
            trial = beta - step * gradient
            new = np.sign(trial) * np.maximum(np.abs(trial) - step * penalties, 0)
            displacement = float(np.sum((new - beta) ** 2))
            if displacement == 0:
                # An unrepresentably small trial is not evidence of stationarity.
                step *= 2
                continue
            new_loss, new_gradient = evaluate(new)
            if (np.isfinite(new_loss) and np.all(np.isfinite(new_gradient))
                    and new_loss <= loss - displacement / (4 * step) + roundoff):
                return new, new_loss, new_gradient, min(step * 2, 1e12)
            step *= 0.5
        return None
