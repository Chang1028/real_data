# Bilinear lasso workflow

This repository contains one reproducible workflow for sparse two-block quadratic regression:

\[
\hat y_i = \beta_1^T X_{1,i}\beta_1 + \beta_2^T X_{2,i}\beta_2.
\]

## Use these notebooks

| Notebook | Use |
| --- | --- |
| `model/simulation_experiments.ipynb` | Define true beta blocks, simulate outcomes, cross-validate penalties, and fit the final simulation model. |
| `model/plot_saved_simulation.ipynb` | Load a completed simulation run and make plots without refitting. |
| `model/real_y_experiment.ipynb` | Fit the model to the observed outcome in `y.npy`. |

All generated runs are saved locally under `model/results/`. Each run has its own folder containing cross-validation summaries, fitted coefficients, diagnostics, and figures. The large raw arrays in these runs are intentionally excluded from GitHub. Small published figures are available in `model/results/published_figures/`.

## Setup

Create an environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy jax pandas scikit-learn joblib matplotlib jupyter
```

Set `DATA_DIR` in `model/parameters.json` to a folder containing `z_fc.npy` and `z_sc.npy`. The observed-outcome notebook also requires the selected outcome file, such as `y.npy`, in that folder.

Start Jupyter at the repository root and open one of the three notebooks above:

```bash
jupyter notebook
```
