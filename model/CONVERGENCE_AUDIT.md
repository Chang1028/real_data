The active `bilinear_regression` pipeline was inspected and updated on 2026-09-25. The checks below used the data and settings used by `simulation copy 2.ipynb`: 514 subjects, 200 ROIs, age as the outcome, seed 2026, `MAX_ITER=6000`, and the original absolute `TOL=1e-5`.

The original penalized objective is unchanged:

`F(beta1, beta2) = 0.5 * mean((prediction - y)**2) + lambda1 * ||beta1||_1 + lambda2 * ||beta2||_1`.

For each coefficient, the stationarity violation is `abs(gradient + lambda * sign(beta))` when nonzero, and `max(abs(gradient) - lambda, 0)` when zero. Convergence requires the maximum violation across BOTH blocks to be at most `1e-5`. No stopping tolerance was relaxed.

| Check | Result |
| --- | --- |
| Real-age CV: lambda pairs (1,1), (1,2), (2,1), (2,2), five folds each | All 20 selected fits converged; all 40 individual starts also converged |
| Maximum selected CV stationarity | 9.979730540399245e-6 |
| Full-grid elapsed time | 301.84 seconds with two workers |
| Individual CV candidates | 400–2,800 iterations; median 10.45 seconds, observed range 5.42–36.01 seconds |
| Full-data final fit at (4,4) | Both candidates converged; selected objective 545.4275027420829, stationarity 9.544045483700359e-6 |
| Independent final-fit verification | Original objective and KKT residual recomputed with NumPy, independently of JAX and the optimizer; passed |
| Fixed-block simulation with real connectivity, SNR=2, lambda=(1,1), five folds | All five selected fits and all ten starts converged; maximum selected stationarity 9.96265267505514e-6 |
| Regression and pipeline tests | 17 passed |
| Notebook execution | Every code cell executed on a reduced synthetic fixture, including CV and all diagnostic plots |

Timings describe this local run, not a runtime guarantee. The simulation check covered one lambda pair; the real-age check covered the entire configured grid. The CLI entry point was repaired and compile-checked; the full-data runs called its shared loaders/CV/model functions directly.

Changes and findings:

- `bilinear_lasso.py` now uses float64 in every process and JIT-compiles its smooth loss/gradient. The default optimizer is L-BFGS-B with `beta = u-v`, `u,v >= 0`, and penalty `lambda*(u+v)`. Minimizing this bounded smooth objective is equivalent to minimizing the original exact L1 objective. It remains nonconvex. The previous proximal-gradient algorithm is available with `method="proximal"` and is also used for recovery if L-BFGS-B stops early. The total update budget per candidate remains `max_iter`, including restarts and recovery.
- A SciPy success message, small coefficient movement, or small objective change cannot mark a fit converged. The original L1 stationarity residual is checked independently. `ftol=0` avoids an ordinary relative-objective tolerance as the main stopping rule, but even an equal-objective termination is checked and retried. See the [SciPy L-BFGS-B stopping-rule documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
- Two training-data-scaled random starts are used by CV and the final notebook fit. Each block's initial quadratic prediction standard deviation is scaled to `std(y_train)/sqrt(2)`. This is an initialization heuristic; it changes neither the data nor the fitted penalty. Calling `fit(initialization="provided")` retains the previous supplied-beta-plus-disturbance initialization. Candidate selection uses the lowest original objective among converged candidates.
- Loss, stationarity, iteration counts, solver messages, timings, and coefficient snapshots are retained. The reported loss is the original objective. During split-variable optimization it need not decrease at every iteration, because overlapping positive/negative parts can carry additional penalty in the optimizer's internal objective. In the original audit logs, a successful callback stop appears as `` `callback` raised `StopIteration`. ``; the checked stationarity established convergence. The current code labels this message more clearly.
- CV now defaults to `require_convergence=True`, includes the diagnostics in fold results and summary columns, and identifies the fold/lambda pair if a fit fails. It cannot silently rank unfinished fits as converged. `require_convergence=False` allows explicit exploratory runs, with warnings and false convergence flags retained.
- `N_JOBS` was reduced from five to two and nested BLAS threads are limited. This reduces competing memory and CPU usage. `NUM_CANDIDATES=2` and `INITIALIZATION="scaled_random"` are explicit in `parameters.json`. The user's lambda grid, iteration limit, and tolerance are preserved.
- Loaders preserve float64 before any JAX conversion. X2 still follows exactly the requested order: `log10(1+X2)`, normalize by the transformed diagonal square roots, then zero the diagonal. Zero denominators keep the existing zero-entry convention. Invalid log/square-root inputs now raise a clear error. X1 receives no additional transformation.
- CV centers predictors and outcome using training-fold means only. An automated test changes held-out matrices/outcomes and verifies that the estimated betas remain identical.
- The fixed simulation blocks are preserved as requested. SPARSITY1/SPARSITY2 remain compatibility parameters and do not control these fixed supports; the docstring and notebook now say so.
- `main.py` had stale sparsity, CV, and plotting calls. These now use the current APIs and the same data filenames as the active notebook. Importing `main` no longer launches a full experiment.
- The notebook's final penalty pair remains (4,4), as before. It is a separately specified full-data fit, not the pair selected by the current (1,2) CV grid. Its prediction plots are in-sample. The CV summary retains the existing EBIC sort and information-criterion calculation.

Results are in `results/convergence_audit/`: the real-age CV summary, fold/candidate diagnostics, simulation diagnostics, selected final betas/predictions, parameters, and an independent final-fit check. No earlier result files were overwritten.

To repeat the automated tests from the repository root:

```sh
.venv/bin/python -m unittest discover -s bilinear_regression -p 'test_*.py'
```

To repeat the full real-data audit and five-fold simulation check:

```sh
.venv/bin/python bilinear_regression/validate_convergence.py --simulation
```

Restart the notebook kernel and run all cells to load the revised modules. The notebook explicitly loads the module in `bilinear_regression`; the separate repository-root `bilinear_lasso.py` is a legacy implementation and was not changed in this audit.

These checks establish numerical stationarity for the tested estimates. They do not guarantee convergence for every future dataset or starting point, a global minimum, or recovery of the true coefficients. In particular, each quadratic block has an unavoidable global sign ambiguity (`beta` and `-beta` give identical predictions), and the all-zero solution is also stationary. Prediction and scientific validity must be assessed separately using held-out results.
