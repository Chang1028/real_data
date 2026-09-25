# Bilinear simulation model

Run `simulation_experiments.ipynb` to define true beta blocks, run cross-validation, and save each experiment under `model/results/`. Run `plot_saved_simulation.ipynb` to load a completed experiment and reproduce diagnostics without refitting.

The notebooks require the connectivity files specified by `DATA_DIR` in `parameters.json`. Update that path before running on another computer.

Run all verification checks from the repository root with:

```sh
.venv/bin/python -m unittest discover -s model -p 'test_*.py'
```
