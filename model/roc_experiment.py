"""Repeated beta-support ROC experiments for the bilinear lasso model."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import uuid

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from bilinear_lasso import bilinear_lasso
from data_utils import generate_simulation_data, simulation_signal_noise_diagnostics


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def beta_support_rates(beta_true, beta_estimated, threshold=1e-6):
    """Return TPR and FPR for a coefficient vector's nonzero support."""
    truth = np.abs(np.asarray(beta_true).ravel()) > 0
    selected = np.abs(np.asarray(beta_estimated).ravel()) > threshold
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    true_positives = int(np.logical_and(truth, selected).sum())
    false_positives = int(np.logical_and(~truth, selected).sum())
    return {
        "tpr": true_positives / positives if positives else np.nan,
        "fpr": false_positives / negatives if negatives else np.nan,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "n_true_support": positives,
        "n_null": negatives,
    }


def _center_full_sample(X1, X2, y):
    return (
        X1 - X1.mean(axis=2, keepdims=True),
        X2 - X2.mean(axis=2, keepdims=True),
        y - y.mean(),
    )


def _one_replication(
    *, data_dir, p, n, snr, replicate, seed, beta1_true, beta2_true,
    lambda1, lambda2_values, max_iter, step_size, tol, num_candidates,
    initialization, support_threshold, rs_file, emo_file, lambda1_values=None,
    lambda2=None,
):
    """Fit one penalty path and measure support recovery for both betas."""
    x1, x2, y, true1, true2 = generate_simulation_data(
        data_dir=data_dir, p=p, n=n, sparsity1=0, sparsity2=0, snr=snr,
        seed=seed, rs_file=rs_file, emo_file=emo_file,
        beta1_true=beta1_true, beta2_true=beta2_true,
    )
    x1, x2, y = _center_full_sample(x1, x2, y)
    rows = []
    if lambda1_values is None:
        lambda_pairs = [(lambda1, value) for value in lambda2_values]
    else:
        lambda_pairs = [(value, lambda2) for value in lambda1_values]
    for offset, (lambda1_value, lambda2_value) in enumerate(lambda_pairs):
        model = bilinear_lasso(x1, x2, y, lambda1_value, lambda2_value)
        model.initialize_beta(np.zeros_like(true1), np.zeros_like(true2))
        # Retain a diagnostic row even if this penalty reaches max_iter.  The
        # averaged ROC uses only fits that satisfy the stationarity criterion.
        model.fit(
            num_candidates=num_candidates, max_iter=max_iter, step_size=step_size,
            tol=tol, disturbance=2, seed=seed + offset + 1,
            initialization=initialization, require_convergence=False,
        )
        beta1_rates = beta_support_rates(true1, model.beta1, threshold=support_threshold)
        beta2_rates = beta_support_rates(true2, model.beta2, threshold=support_threshold)
        rows.append({
            "snr": float(snr), "replicate": int(replicate), "seed": int(seed),
            "lambda1": float(lambda1_value), "lambda2": float(lambda2_value),
            "converged": bool(model.converged_),
            "stationarity": float(model.stationarity_),
            "iterations": int(model.candidate_status[model.sel_idx]["iterations"]),
            **{f"beta1_{key}": value for key, value in beta1_rates.items()},
            **beta2_rates,
        })
    return rows


def _replicate_curve(points, fpr_grid, fpr_column="fpr", tpr_column="tpr"):
    """Create a monotone ROC path, then interpolate it on ``fpr_grid``."""
    points = points.loc[points["converged"] & points[fpr_column].notna()].copy()
    if points.empty:
        return None
    points = pd.concat([
        pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}),
        points[[fpr_column, tpr_column]].rename(
            columns={fpr_column: "fpr", tpr_column: "tpr"}
        ),
    ], ignore_index=True).sort_values("fpr")
    points = points.groupby("fpr", as_index=False)["tpr"].max()
    points["tpr"] = points["tpr"].cummax()
    return np.interp(fpr_grid, points["fpr"], points["tpr"])


def _average_support_roc(points, *, fpr_column, tpr_column, fpr_grid=None):
    """Average regularization-path ROC-like curves within each SNR."""
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 101)
    output = []
    for snr, snr_points in points.groupby("snr", sort=True):
        curves = []
        for _, replicate_points in snr_points.groupby("replicate", sort=True):
            curve = _replicate_curve(
                replicate_points, fpr_grid, fpr_column=fpr_column, tpr_column=tpr_column
            )
            if curve is not None:
                curves.append(curve)
        if not curves:
            continue
        curves = np.asarray(curves)
        mean_tpr = curves.mean(axis=0)
        for fpr, tpr in zip(fpr_grid, mean_tpr):
            output.append({
                "snr": float(snr), "fpr": float(fpr), "mean_tpr": float(tpr),
                "n_replicates": int(len(curves)),
                "auc": float(np.trapezoid(mean_tpr, fpr_grid)),
            })
    return pd.DataFrame(output)


def average_beta2_roc(points, fpr_grid=None):
    """Average beta2 regularization-path ROC curves within each SNR."""
    return _average_support_roc(points, fpr_column="fpr", tpr_column="tpr", fpr_grid=fpr_grid)


def average_beta1_roc(points, fpr_grid=None):
    """Average beta1 support recovery as lambda2 varies at fixed lambda1."""
    return _average_support_roc(
        points, fpr_column="beta1_fpr", tpr_column="beta1_tpr", fpr_grid=fpr_grid
    )


def average_beta2_lambda_points(points):
    """Average the raw beta2 support-recovery point for every lambda2 and SNR."""
    points = points.copy()
    usable = points["converged"] & points["fpr"].notna() & points["tpr"].notna()
    points["valid_fpr"] = points["fpr"].where(usable)
    points["valid_tpr"] = points["tpr"].where(usable)
    return points.groupby(["snr", "lambda2"], as_index=False).agg(
        mean_fpr=("valid_fpr", "mean"), mean_tpr=("valid_tpr", "mean"),
        n_replicates=("valid_fpr", "count"), n_attempted=("replicate", "nunique"),
    )


def average_beta1_lambda_points(points):
    """Average raw beta1 support-recovery points for every lambda2 and SNR."""
    usable = points.loc[points["converged"] & points["beta1_fpr"].notna()].copy()
    return usable.groupby(["snr", "lambda2"], as_index=False).agg(
        mean_fpr=("beta1_fpr", "mean"), mean_tpr=("beta1_tpr", "mean"),
        n_replicates=("replicate", "nunique"),
    )


def average_beta1_lambda1_points(points):
    """Average raw beta1 support-recovery points for every lambda1 and SNR."""
    points = points.copy()
    usable = points["converged"] & points["beta1_fpr"].notna() & points["beta1_tpr"].notna()
    points["valid_fpr"] = points["beta1_fpr"].where(usable)
    points["valid_tpr"] = points["beta1_tpr"].where(usable)
    return points.groupby(["snr", "lambda1"], as_index=False).agg(
        mean_fpr=("valid_fpr", "mean"), mean_tpr=("valid_tpr", "mean"),
        n_replicates=("valid_fpr", "count"), n_attempted=("replicate", "nunique"),
    )


def _plot_centroid_path(
    centers, *, beta, path_lambda, lambda_values=None, save_path=None,
    label_column="snr", label_prefix="SNR"
):
    """Connect mean raw (FPR, TPR) points in penalty order, without added endpoints."""
    import matplotlib.pyplot as plt

    if label_column not in centers:
        raise ValueError(f"Missing legend column: {label_column}")
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for snr, group in centers.groupby("snr", sort=True):
        group = group.copy()
        if lambda_values is not None:
            order = {float(value): i for i, value in enumerate(lambda_values)}
            group["path_order"] = group[path_lambda].map(order)
            group = group.sort_values("path_order")
        else:
            group = group.sort_values(path_lambda)
        label = group[label_column].iloc[0]
        line, = ax.plot(group["mean_fpr"], group["mean_tpr"], marker="o", linewidth=2,
                        label=f"{label_prefix}={label:g}")
        for _, point in group.dropna(subset=["mean_fpr", "mean_tpr"]).iterrows():
            ax.annotate(f"$\\lambda_{path_lambda[-1]}$={point[path_lambda]:g} (n={int(point['n_replicates'])})",
                        (point["mean_fpr"], point["mean_tpr"]), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color=line.get_color())
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1, label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"False positive rate ({beta})",
           ylabel=f"True positive rate ({beta})",
           title=f"{beta.capitalize()} path through mean replication points")
    ax.legend(loc="lower right")
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, ax


def plot_beta1_centroid_path(
    centers, *, lambda1_values=None, save_path=None, label_column="snr", label_prefix="SNR"
):
    return _plot_centroid_path(
        centers, beta="beta1", path_lambda="lambda1", lambda_values=lambda1_values,
        save_path=save_path, label_column=label_column, label_prefix=label_prefix,
    )


def plot_beta2_centroid_path(
    centers, *, lambda2_values=None, save_path=None, label_column="snr", label_prefix="SNR"
):
    return _plot_centroid_path(
        centers, beta="beta2", path_lambda="lambda2", lambda_values=lambda2_values,
        save_path=save_path, label_column=label_column, label_prefix=label_prefix,
    )


def _plot_average_roc(
    mean_roc, *, beta_label, save_path=None, label_column="snr", label_prefix="SNR",
    lambda_points=None, path_lambda="lambda2",
):
    """Plot averaged regularization-path ROC curves for one beta vector."""
    import matplotlib.pyplot as plt

    if label_column not in mean_roc.columns:
        raise ValueError(f"Missing ROC legend column: {label_column}")
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for snr, curve in mean_roc.groupby("snr", sort=True):
        auc = curve["auc"].iloc[0]
        n_reps = curve["n_replicates"].iloc[0]
        label_value = curve[label_column].iloc[0]
        line, = ax.plot(curve["fpr"], curve["mean_tpr"], linewidth=2,
                        label=f"{label_prefix}={label_value:g} (AUC={auc:.3f}, n={n_reps})")
        if lambda_points is not None:
            raw = lambda_points.loc[lambda_points["snr"] == snr]
            ax.scatter(raw["mean_fpr"], raw["mean_tpr"], color=line.get_color(),
                       edgecolor="black", linewidth=.4, s=45, zorder=3)
            for _, point in raw.iterrows():
                ax.annotate(f"$\\lambda_{path_lambda[-1]}$={point[path_lambda]:g}",
                            (point["mean_fpr"], point["mean_tpr"]),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=8, color=line.get_color())
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1,
            label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1),
           xlabel=f"False positive rate ({beta_label})",
           ylabel=f"True positive rate ({beta_label})",
           title=f"Average {beta_label} regularization-path ROC (markers show {path_lambda})")
    ax.legend(loc="lower right")
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, ax


def plot_average_beta2_roc(
    mean_roc, save_path=None, label_column="snr", label_prefix="SNR",
    lambda_points=None,
):
    """Plot averaged beta2 ROC paths, optionally labeling raw lambda2 points."""
    return _plot_average_roc(
        mean_roc, beta_label="beta2", save_path=save_path,
        label_column=label_column, label_prefix=label_prefix,
        lambda_points=lambda_points,
    )


def plot_average_beta1_roc(
    mean_roc, save_path=None, label_column="snr", label_prefix="SNR",
    lambda_points=None, path_lambda="lambda2",
):
    """Plot averaged beta1 ROC paths as lambda2 varies at fixed lambda1."""
    return _plot_average_roc(
        mean_roc, beta_label="beta1", save_path=save_path,
        label_column=label_column, label_prefix=label_prefix,
        lambda_points=lambda_points, path_lambda=path_lambda,
    )


def fit_beta2_roc_detail(
    *, data_dir, p, n, beta1_true, beta2_true, snr, seed, lambda1, lambda2,
    max_iter=12000, step_size=1e-6, tol=3e-5, num_candidates=2,
    initialization="scaled_random", optimizer_seed=None, rs_file="z_fc.npy", emo_file="z_sc.npy",
):
    """Refit one ROC-path point and return its true and estimated betas."""
    x1, x2, y, true1, true2 = generate_simulation_data(
        data_dir=data_dir, p=p, n=n, sparsity1=0, sparsity2=0, snr=snr,
        seed=seed, rs_file=rs_file, emo_file=emo_file,
        beta1_true=beta1_true, beta2_true=beta2_true,
    )
    x1, x2, y = _center_full_sample(x1, x2, y)
    model = bilinear_lasso(x1, x2, y, lambda1, lambda2)
    model.initialize_beta(np.zeros_like(true1), np.zeros_like(true2))
    if optimizer_seed is None:
        optimizer_seed = seed + 1
    model.fit(
        num_candidates=num_candidates, max_iter=max_iter, step_size=step_size,
        tol=tol, disturbance=2, seed=optimizer_seed,
        initialization=initialization, require_convergence=False,
    )
    return {
        "beta1_true": true1.ravel(), "beta2_true": true2.ravel(),
        "beta1_estimated": np.asarray(model.beta1).ravel(),
        "beta2_estimated": np.asarray(model.beta2).ravel(),
        "converged": model.converged_, "stationarity": model.stationarity_,
        "lambda1": float(lambda1), "lambda2": float(lambda2),
        "snr": float(snr), "seed": int(seed),
    }


def plot_beta2_roc_detail(detail, save_path=None):
    """Overlay true and estimated beta vectors for one ROC-path point."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for index, ax in enumerate(axes, start=1):
        truth = np.asarray(detail[f"beta{index}_true"])
        estimated = np.asarray(detail[f"beta{index}_estimated"])
        # A beta vector and its negative give the same quadratic prediction.
        if estimated @ truth < 0:
            estimated = -estimated
        ax.plot(truth, color="black", linewidth=2, drawstyle="steps-mid", label="True beta")
        ax.plot(estimated, color="tab:orange", linewidth=1, label="Estimated beta (sign-aligned)")
        ax.axhline(0, color="0.5", linewidth=.7, linestyle=":")
        ax.set(title=rf"$\beta_{index}$", ylabel="Coefficient")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("ROI index (0-based)")
    status = "converged" if detail["converged"] else "not certified"
    fig.suptitle(
        f"ROC detail | SNR={detail['snr']:.4g}, "
        f"lambda1={detail['lambda1']:g}, lambda2={detail['lambda2']:g} | {status}"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, axes


def run_beta2_roc_experiment(
    *, name, data_dir, p, n, beta1_true, beta2_true, snr_values,
    lambda1=100, lambda2_values=range(100, 201, 10), n_replicates=10,
    seed=2026, max_iter=12000, step_size=1e-6, tol=3e-5,
    num_candidates=2, initialization="scaled_random", support_threshold=1e-6,
    n_jobs=1, rs_file="z_fc.npy", emo_file="z_sc.npy", results_dir=RESULTS_DIR,
    centroid_path=False,
):
    """Run and save repeated beta1 and beta2 ROC curves at fixed ``lambda1``.

    ``snr_values`` are passed directly to the simulator, where
    SNR = sd(signal) / sd(epsilon).  beta1 remains estimated with lambda1
    fixed; each fit records support rates for both beta vectors as lambda2 varies.
    """
    beta1_true = np.asarray(beta1_true, dtype=np.float64).reshape(p, 1)
    beta2_true = np.asarray(beta2_true, dtype=np.float64).reshape(p, 1)
    lambda2_values = [float(value) for value in lambda2_values]
    if not lambda2_values or any(value < 0 or not np.isfinite(value) for value in lambda2_values):
        raise ValueError("lambda2_values must be nonempty, finite, and nonnegative.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_") or "beta2_roc"
    run_dir = Path(results_dir) / f"{slug}_{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    metadata = {
        "name": str(name), "created_utc": stamp, "status": "running",
        "stage": "replications", "replications_completed": 0,
        "lambda1": float(lambda1), "lambda2_values": lambda2_values,
        "snr_values": [float(value) for value in snr_values],
        "n_replicates": int(n_replicates), "seed": int(seed),
        "support_targets": ["beta1", "beta2"],
        "beta1_treatment": "estimated with fixed lambda1",
        "parameters": {"p": int(p), "n": int(n), "max_iter": int(max_iter),
                       "step_size": float(step_size), "tol": float(tol),
                       "num_candidates": int(num_candidates), "n_jobs": int(n_jobs)},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tasks = []
    for snr_index, snr in enumerate(snr_values):
        for replicate in range(n_replicates):
            tasks.append(dict(
                data_dir=data_dir, p=p, n=n, snr=float(snr), replicate=replicate,
                seed=seed + 10_000 * snr_index + replicate, beta1_true=beta1_true,
                beta2_true=beta2_true, lambda1=lambda1, lambda2_values=lambda2_values,
                max_iter=max_iter, step_size=step_size, tol=tol,
                num_candidates=num_candidates, initialization=initialization,
                support_threshold=support_threshold, rs_file=rs_file, emo_file=emo_file,
            ))
    print(f"Running {len(tasks)} replications: {len(snr_values)} SNR values x "
          f"{n_replicates} replications; beta1/beta2 path has {len(lambda2_values)} lambda values.")
    started = time.monotonic()
    nested_rows = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed_rows = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_one_replication)(**task) for task in tasks
        )
        for completed, rows in enumerate(completed_rows, start=1):
            nested_rows.append(rows)
            points_so_far = pd.DataFrame([row for result in nested_rows for row in result])
            points_so_far.to_csv(run_dir / "beta2_roc_points.csv", index=False)
            metadata.update(
                replications_completed=completed,
                n_fit_rows=int(len(points_so_far)),
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            example = rows[0]
            print(f"Completed {completed}/{len(tasks)} replications "
                  f"(SNR={example['snr']:g}, replicate={example['replicate'] + 1}); "
                  f"elapsed {time.monotonic() - started:.1f} seconds", flush=True)
    points = pd.DataFrame([row for rows in nested_rows for row in rows])
    if centroid_path:
        centers = average_beta2_lambda_points(points)
        points.to_csv(run_dir / "beta2_roc_points.csv", index=False)
        centers.to_csv(run_dir / "beta2_centroid_points.csv", index=False)
        plot_beta2_centroid_path(
            centers, lambda2_values=lambda2_values,
            save_path=run_dir / "figures" / "beta2_centroid_path.png",
        )
        incomplete = centers.loc[centers["n_replicates"] < n_replicates]
        if not incomplete.empty:
            print("Some centers use fewer than the requested replications because fits "
                  "did not converge or rates were undefined; see beta2_centroid_points.csv.",
                  flush=True)
        metadata.update(status="complete", n_fit_rows=int(len(points)),
                        n_converged_fit_rows=int(points["converged"].sum()),
                        averaging_method="mean raw FPR and TPR by lambda2",
                        n_incomplete_centers=int(len(incomplete)))
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return run_dir, points, centers
    mean_roc = average_beta2_roc(points)
    mean_beta1_roc = average_beta1_roc(points)
    points.to_csv(run_dir / "beta2_roc_points.csv", index=False)
    mean_roc.to_csv(run_dir / "mean_beta2_roc.csv", index=False)
    mean_beta1_roc.to_csv(run_dir / "mean_beta1_roc.csv", index=False)
    plot_average_beta2_roc(mean_roc, save_path=run_dir / "figures" / "mean_beta2_roc.png")
    plot_average_beta1_roc(mean_beta1_roc, save_path=run_dir / "figures" / "mean_beta1_roc.png")
    metadata.update(status="complete", n_fit_rows=int(len(points)),
                    n_converged_fit_rows=int(points["converged"].sum()))
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir, points, mean_roc


def run_beta1_roc_experiment(
    *, name, data_dir, p, n, beta1_true, beta2_true, snr_values,
    lambda2=100, lambda1_values=range(100, 201, 10), n_replicates=10,
    seed=2026, max_iter=12000, step_size=1e-6, tol=3e-5,
    num_candidates=2, initialization="scaled_random", support_threshold=1e-6,
    n_jobs=1, rs_file="z_fc.npy", emo_file="z_sc.npy", results_dir=RESULTS_DIR,
):
    """Run beta1 ROC curves while varying lambda1 at fixed ``lambda2``."""
    beta1_true = np.asarray(beta1_true, dtype=np.float64).reshape(p, 1)
    beta2_true = np.asarray(beta2_true, dtype=np.float64).reshape(p, 1)
    lambda1_values = [float(value) for value in lambda1_values]
    if not lambda1_values or any(value < 0 or not np.isfinite(value) for value in lambda1_values):
        raise ValueError("lambda1_values must be nonempty, finite, and nonnegative.")
    if lambda2 < 0 or not np.isfinite(lambda2):
        raise ValueError("lambda2 must be finite and nonnegative.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_") or "beta1_roc"
    run_dir = Path(results_dir) / f"{slug}_{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    metadata = {
        "name": str(name), "created_utc": stamp, "status": "running",
        "stage": "replications", "replications_completed": 0,
        "lambda1_values": lambda1_values, "lambda2": float(lambda2),
        "snr_values": [float(value) for value in snr_values],
        "n_replicates": int(n_replicates), "seed": int(seed),
        "support_targets": ["beta1", "beta2"],
        "path_parameter": "lambda1",
        "parameters": {"p": int(p), "n": int(n), "max_iter": int(max_iter),
                       "step_size": float(step_size), "tol": float(tol),
                       "num_candidates": int(num_candidates), "n_jobs": int(n_jobs)},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tasks = []
    for snr_index, snr in enumerate(snr_values):
        for replicate in range(n_replicates):
            tasks.append(dict(
                data_dir=data_dir, p=p, n=n, snr=float(snr), replicate=replicate,
                seed=seed + 10_000 * snr_index + replicate, beta1_true=beta1_true,
                beta2_true=beta2_true, lambda1=None, lambda2_values=None,
                lambda1_values=lambda1_values, lambda2=float(lambda2),
                max_iter=max_iter, step_size=step_size, tol=tol,
                num_candidates=num_candidates, initialization=initialization,
                support_threshold=support_threshold, rs_file=rs_file, emo_file=emo_file,
            ))
    print(f"Running {len(tasks)} replications: {len(snr_values)} SNR values x "
          f"{n_replicates} replications; beta1 path has {len(lambda1_values)} lambda values.")
    started = time.monotonic()
    nested_rows = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed_rows = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_one_replication)(**task) for task in tasks
        )
        for completed, rows in enumerate(completed_rows, start=1):
            nested_rows.append(rows)
            points_so_far = pd.DataFrame([row for result in nested_rows for row in result])
            points_so_far.to_csv(run_dir / "beta1_roc_points.csv", index=False)
            metadata.update(
                replications_completed=completed,
                n_fit_rows=int(len(points_so_far)),
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            example = rows[0]
            print(f"Completed {completed}/{len(tasks)} replications "
                  f"(SNR={example['snr']:g}, replicate={example['replicate'] + 1}); "
                  f"elapsed {time.monotonic() - started:.1f} seconds", flush=True)
    points = pd.DataFrame([row for rows in nested_rows for row in rows])
    centers = average_beta1_lambda1_points(points)
    points.to_csv(run_dir / "beta1_roc_points.csv", index=False)
    centers.to_csv(run_dir / "beta1_centroid_points.csv", index=False)
    plot_beta1_centroid_path(
        centers, lambda1_values=lambda1_values,
        save_path=run_dir / "figures" / "beta1_centroid_path.png",
    )
    incomplete = centers.loc[centers["n_replicates"] < n_replicates]
    if not incomplete.empty:
        print("Some centers use fewer than the requested replications because fits "
              "did not converge or rates were undefined; see beta1_centroid_points.csv.", flush=True)
    metadata.update(status="complete", n_fit_rows=int(len(points)),
                    n_converged_fit_rows=int(points["converged"].sum()),
                    averaging_method="mean raw FPR and TPR by lambda1",
                    n_incomplete_centers=int(len(incomplete)))
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir, points, centers
