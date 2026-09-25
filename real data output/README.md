# Observed `y.npy` output

These figures come from the completed observed-outcome run:

- Outcome: `y.npy`
- Subjects: 514
- ROI count: 200
- Cross-validation: five folds over 25 \((\lambda_1,\lambda_2)\) pairs
- Final pair: \((5, 5)\), selected by minimum mean validation MSE
- Solver tolerance: `1e-5`; every selected CV fit converged

| Figure | Contents |
| --- | --- |
| `cv_test_r2_heatmap.png` | Mean validation R² for every lambda pair |
| `cv_fold_r2.png` | Individual and mean validation R² values across folds |
| `final_fit_diagnostics.png` | Estimated beta profiles, observed-versus-fitted outcome, and residuals from the full-data refit |
| `optimizer_diagnostics.png` | Penalized objective and L1-stationarity histories for final-fit starts |

The beta and prediction panels use a full-data refit after CV selected the penalties. They are not held-out test predictions. The CV figures contain the held-out validation performance.
