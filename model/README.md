# Model workflow

Use only these notebooks:

- `simulation_experiments.ipynb` — create and fit simulated experiments.
- `plot_saved_simulation.ipynb` — plot a completed simulation run.
- `real_y_experiment.ipynb` — fit an observed outcome.

The shared modules in this folder provide fitting, data loading, cross-validation, result saving, and plotting. Set the data location in `parameters.json` before running a notebook.

Generated output is saved locally in `results/`. Each experiment receives a separate subfolder.
