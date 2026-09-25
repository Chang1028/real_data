# Bilinear lasso simulation workflow

This repository contains a reproducible workflow for fitting a sparse two-block quadratic regression model,

\[
\hat y_i = \beta_1^T X_{1,i}\beta_1 + \beta_2^T X_{2,i}\beta_2,
\]

selecting \(\lambda_1,\lambda_2\) by cross-validation, and comparing estimated coefficients with known simulation coefficients.

## Repository layout

| Location | Purpose |
| --- | --- |
| [`model/`](model/) | Solver, preprocessing, cross-validation, evaluation, plotting, notebooks, tests, and run configuration |
| [`model/simulation_experiments.ipynb`](model/simulation_experiments.ipynb) | Define true beta blocks, run simulations and CV, fit a final model, and save each experiment |
| [`model/plot_saved_simulation.ipynb`](model/plot_saved_simulation.ipynb) | Load a saved experiment and reproduce plots without refitting |
| [`model/real_y_experiment.ipynb`](model/real_y_experiment.ipynb) | Fit the observed outcome in `y.npy`, select penalties by CV, and save real-outcome diagnostics |
| [`simulation output/`](simulation%20output/) | Three published true-versus-estimated beta overlays |
| [`real data output/`](real%20data%20output/) | Published diagnostics from the observed `y.npy` run |

## Requirements

The workflow was tested with Python 3.12. Create an environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy jax pandas scikit-learn joblib matplotlib jupyter
```

## Data required to run simulations

The connectivity arrays are not included in this repository. Obtain the following files separately:

```text
z_fc.npy
z_sc.npy
```

Put them in a local data folder. Then edit [`model/parameters.json`](model/parameters.json) and change `DATA_DIR` to that folder, for example:

```json
"DATA_DIR": "/Users/your_name/project/data"
```

For the simulation workflow, only `z_fc.npy` and `z_sc.npy` are required. Both arrays must contain the same subjects and be reshapeable to `(n_subjects, 200, 200)` under the default configuration.

## Reproduce a simulation

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open [`model/simulation_experiments.ipynb`](model/simulation_experiments.ipynb) and edit the `EXPERIMENTS` cell. A beta block is written as `(start, stop, coefficient)`, uses zero-based ROI indices, and excludes the stop index.

```python
EXPERIMENTS = [
    {
        "name": "original_blocks",
        "beta1_blocks": [(40, 60, 1.5), (120, 150, -1.0)],
        "beta2_blocks": [(40, 60, 1.0), (120, 150, -1.2)],
    },
]
```

For example, `(40, 60, 1.5)` sets ROIs 40 through 59 to `1.5`. Duplicate the experiment dictionary and alter its blocks to evaluate another true-beta structure.

Run all notebook cells. The workflow:

1. Generates the simulation using the specified true betas.
2. Evaluates every configured \((\lambda_1,\lambda_2)\) pair using five-fold cross-validation.
3. Selects the pair with lowest mean validation MSE unless `FINAL_LAMBDA_PAIR` is explicitly set.
4. Refits the selected model on all simulated subjects.
5. Saves the run under `model/results/<experiment_name>_<fingerprint>_<timestamp>/`.

Each saved experiment includes true betas, generated outcomes, CV folds and predictions, final betas, convergence diagnostics, optimization histories, and generated figures.

## Recreate diagnostic figures without refitting

Open [`model/plot_saved_simulation.ipynb`](model/plot_saved_simulation.ipynb). Its first selection cell prints completed saved runs. Set:

```python
RUN_INDEX = -1
```

to load the latest run, or choose one of the printed indices. Run all cells to generate:

- True-versus-estimated beta overlays
- Cross-validation heatmaps and fold-level performance plots
- Fitted-versus-simulated outcome and residual plots
- Objective and stationarity histories
- Coefficient trajectories

This plotting notebook loads saved files; it does not rerun fitting.

## Fit an observed outcome

Open [`model/real_y_experiment.ipynb`](model/real_y_experiment.ipynb) to fit the observed outcome in `y.npy` rather than a simulated outcome. Change `Y_FILE` in its configuration cell to fit another outcome file such as `age.npy`.

The notebook uses the same cross-validation and convergence checks, selects the final pair by mean validation MSE unless an explicit pair is supplied, refits using all available subjects, and saves each run below `model/results/real_y_<name>_<timestamp>/`.

## Published beta overlays

The three figures in [`simulation output/`](simulation%20output/) use the fixed `original_blocks` true beta structure. Their final full-data fits used lambda pairs selected from cross-validation:

| Figure | Final \((\lambda_1,\lambda_2)\) |
| --- | --- |
| [`original_blocks_lambda1_5_lambda2_5.png`](simulation%20output/original_blocks_lambda1_5_lambda2_5.png) | `(5, 5)` |
| [`original_blocks_lambda1_20_lambda2_10.png`](simulation%20output/original_blocks_lambda1_20_lambda2_10.png) | `(20, 10)` |
| [`original_blocks_lambda1_20_lambda2_20.png`](simulation%20output/original_blocks_lambda1_20_lambda2_20.png) | `(20, 20)` |

The overlays are based on final fits using all simulation subjects after cross-validation selected the penalties. Estimated beta signs are aligned only for display because \(\beta\) and \(-\beta\) produce the same quadratic prediction.

## Published observed-outcome diagnostics

[`real data output/`](real%20data%20output/) contains four figures from the completed observed `y.npy` experiment using 514 subjects. It evaluated the 25 configured lambda pairs with five-fold CV and selected final \((\lambda_1,\lambda_2)=(5,5)\) by minimum mean validation MSE. The folder includes a validation-R² heatmap, fold-level validation R² plot, final beta/prediction/residual diagnostics, and optimizer objective/stationarity diagnostics.

## Verification

Run the model tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s model -p 'test_*.py'
```

The test suite checks the solver's stationarity criterion, preprocessing, cross-validation isolation, saved-result reconstruction, and plot regeneration.
