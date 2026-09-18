import numpy as np
import pandas as pd
import jax.numpy as jnp

from itertools import product
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

from bilinear_lasso import bilinear_lasso

from config import (
    MAX_ITER,
    STEP_SIZE,
    TOL,
    P,
    N_SPLITS,
    N_JOBS,
    LAMBDA1_GRID,
    LAMBDA2_GRID
)

from evaluation import (
    predict_bilinear,
    evaluate_prediction,
    calculate_ebic
)


# ============================================================
# One CV fold
# ============================================================

def run_one_cv_fold(
    X1,
    X2,
    y,
    train_idx,
    val_idx,
    lam1,
    lam2,
    p,
    max_iter,
    step_size,
    tol,
    support_threshold=1e-6,
    optimizer_seed=2026
):
    """
    Run one train/validation fold for a fixed pair
    (lambda1, lambda2).
    """

    # ========================================================
    # 1. Split data
    # ========================================================

    X1_tr = X1[:, :, train_idx]
    X1_val = X1[:, :, val_idx]

    X2_tr = X2[:, :, train_idx]
    X2_val = X2[:, :, val_idx]

    y_tr = np.asarray(
        y[train_idx]
    ).ravel()

    y_val = np.asarray(
        y[val_idx]
    ).ravel()


    # ========================================================
    # 2. Center y using TRAINING mean only
    # ========================================================

    y_mean = np.mean(y_tr)

    y_tr_c = y_tr - y_mean
    y_val_c = y_val - y_mean


    # ========================================================
    # 3. Center X using TRAINING means only
    # ========================================================

    X1_mean = jnp.mean(
        X1_tr,
        axis=2,
        keepdims=True
    )

    X2_mean = jnp.mean(
        X2_tr,
        axis=2,
        keepdims=True
    )

    X1_tr_c = X1_tr - X1_mean
    X1_val_c = X1_val - X1_mean

    X2_tr_c = X2_tr - X2_mean
    X2_val_c = X2_val - X2_mean


    # ========================================================
    # 4. Fit bilinear model
    # ========================================================

    model = bilinear_lasso(
        X1_tr_c,
        X2_tr_c,
        jnp.asarray(y_tr_c),
        lam1,
        lam2
    )

    model.initialize_beta(
        jnp.zeros((p, 1)),
        jnp.zeros((p, 1))
    )

    model.fit(
        num_candidates=1,
        max_iter=max_iter,
        step_size=step_size,
        tol=tol,
        disturbance=2,
        seed=optimizer_seed
    )


    # ========================================================
    # 5. Extract beta estimates
    # ========================================================

    beta1 = np.asarray(
        model.beta1
    ).ravel()

    beta2 = np.asarray(
        model.beta2
    ).ravel()


    # ========================================================
    # 6. Sparse support
    # ========================================================

    supp1 = np.where(
        np.abs(beta1) > support_threshold
    )[0]

    supp2 = np.where(
        np.abs(beta2) > support_threshold
    )[0]

    n_features = (
        len(supp1)
        +
        len(supp2)
    )


    # ========================================================
    # 7. Predictions
    # ========================================================

    y_pred_tr = predict_bilinear(
        X1_tr_c,
        X2_tr_c,
        beta1,
        beta2
    )

    y_pred_val = predict_bilinear(
        X1_val_c,
        X2_val_c,
        beta1,
        beta2
    )


    # ========================================================
    # 8. Metrics
    # ========================================================

    train_metrics = evaluate_prediction(
        y_tr_c,
        y_pred_tr
    )

    val_metrics = evaluate_prediction(
        y_val_c,
        y_pred_val
    )


    # ========================================================
    # 9. BIC / EBIC
    # ========================================================

    bic, ebic = calculate_ebic(
        mse=train_metrics["mse"],
        n_features=n_features,
        n_samples=len(train_idx),
        total_p=2 * p
    )


    # ========================================================
    # 10. Return fold results
    # ========================================================

    return {

        "lambda1":
            float(lam1),

        "lambda2":
            float(lam2),

        "lambda_pair":
            (
                float(lam1),
                float(lam2)
            ),

        # Training metrics
        "train_mse":
            train_metrics["mse"],

        "train_r2":
            train_metrics["r2"],

        "train_corr":
            train_metrics["corr"],

        # Validation metrics
        "val_mse":
            val_metrics["mse"],

        "val_r2":
            val_metrics["r2"],

        "val_corr":
            val_metrics["corr"],

        # Model selection
        "bic":
            float(bic),

        "ebic":
            float(ebic),

        "n_features":
            n_features,

        # Supports
        "supp1":
            supp1,

        "supp2":
            supp2,

        # Coefficients
        "beta1":
            beta1,

        "beta2":
            beta2,

        # Values for plots
        "y_train_true":
            np.asarray(y_tr_c),

        "y_train_pred":
            np.asarray(y_pred_tr),

        "y_true":
            np.asarray(y_val_c),

        "y_pred":
            np.asarray(y_pred_val)
    }


# ============================================================
# One lambda pair
# ============================================================

def run_one_lambda_pair_cv(
    lam1,
    lam2,
    X1,
    X2,
    y,
    seed=1999
):
    """
    Run K-fold CV for one lambda pair.
    """

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    folds = []

    for fold_id, (
        train_idx,
        val_idx
    ) in enumerate(
        kf.split(
            np.arange(len(y))
        )
    ):

        fold_result = run_one_cv_fold(
            X1=X1,
            X2=X2,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            lam1=lam1,
            lam2=lam2,
            p=P,
            max_iter=MAX_ITER,
            step_size=STEP_SIZE,
            tol=TOL,
            optimizer_seed=(
                seed
                +
                fold_id
            )
        )

        folds.append(
            fold_result
        )


    # ========================================================
    # Aggregate metrics
    # ========================================================

    val_mses = np.asarray(
        [
            f["val_mse"]
            for f in folds
        ]
    )

    mean_mse = np.mean(
        val_mses
    )

    sd_mse = np.std(
        val_mses
    )


    return {

        "lambda1":
            float(lam1),

        "lambda2":
            float(lam2),

        "lambda_pair":
            (
                float(lam1),
                float(lam2)
            ),

        # Training
        "mean_train_r2":
            np.nanmean(
                [
                    f["train_r2"]
                    for f in folds
                ]
            ),

        "mean_train_corr":
            np.nanmean(
                [
                    f["train_corr"]
                    for f in folds
                ]
            ),

        "mean_train_mse":
            np.mean(
                [
                    f["train_mse"]
                    for f in folds
                ]
            ),

        # Validation
        "mean_r2":
            np.nanmean(
                [
                    f["val_r2"]
                    for f in folds
                ]
            ),

        "mean_corr":
            np.nanmean(
                [
                    f["val_corr"]
                    for f in folds
                ]
            ),

        "mean_mse":
            mean_mse,

        "sd_mse":
            sd_mse,

        # Information criteria
        "mean_bic":
            np.mean(
                [
                    f["bic"]
                    for f in folds
                ]
            ),

        "mean_ebic":
            np.mean(
                [
                    f["ebic"]
                    for f in folds
                ]
            ),

        # Sparsity
        "mean_features":
            np.mean(
                [
                    f["n_features"]
                    for f in folds
                ]
            ),

        # CV stability
        "cv_stability":
            (
                sd_mse / mean_mse
                if mean_mse > 1e-12
                else np.nan
            ),

        # Fold-level results
        "folds":
            folds
    }


# ============================================================
# Full 2D lambda grid
# ============================================================

def run_cv_over_lambdas(
    X1,
    X2,
    y,
    seed=1999,
    lambda1_grid=None,
    lambda2_grid=None
):
    """
    Run CV across all combinations of lambda1 and lambda2.

    Example:

        lambda1_grid = [0.1, 0.5, 1]
        lambda2_grid = [0.1, 0.5, 1]

    gives 3 x 3 = 9 lambda pairs.
    """

    # ========================================================
    # Use defaults from config
    # ========================================================

    if lambda1_grid is None:
        lambda1_grid = LAMBDA1_GRID

    if lambda2_grid is None:
        lambda2_grid = LAMBDA2_GRID


    # ========================================================
    # Generate Cartesian product
    # ========================================================

    lambda_pairs = list(
        product(
            lambda1_grid,
            lambda2_grid
        )
    )

    print(
        f"\nRunning {len(lambda_pairs)} "
        "lambda combinations"
    )

    print(
        f"lambda1 grid: "
        f"{lambda1_grid}"
    )

    print(
        f"lambda2 grid: "
        f"{lambda2_grid}"
    )


    # ========================================================
    # Run lambda pairs in parallel
    # ========================================================

    results = Parallel(
        n_jobs=N_JOBS
    )(
        delayed(
            run_one_lambda_pair_cv
        )(
            lam1=lam1,
            lam2=lam2,
            X1=X1,
            X2=X2,
            y=y,
            seed=seed
        )

        for (
            lam1,
            lam2
        )
        in lambda_pairs
    )


    # ========================================================
    # Summary dataframe
    # ========================================================

    summary_df = pd.DataFrame(
        [
            {

                "lambda1":
                    r["lambda1"],

                "lambda2":
                    r["lambda2"],

                "Train R2":
                    r["mean_train_r2"],

                "Test R2":
                    r["mean_r2"],

                "Train Corr":
                    r["mean_train_corr"],

                "Test Corr":
                    r["mean_corr"],

                "Train MSE":
                    r["mean_train_mse"],

                "Test MSE":
                    r["mean_mse"],

                "Avg Features":
                    r["mean_features"],

                "BIC":
                    r["mean_bic"],

                "EBIC":
                    r["mean_ebic"],

                "Stability":
                    r["cv_stability"]
            }

            for r
            in results
        ]
    )


    # ========================================================
    # Train-test gap
    # ========================================================

    summary_df["Gap"] = (
        summary_df["Train R2"]
        -
        summary_df["Test R2"]
    )


    # ========================================================
    # Sort by EBIC
    # ========================================================

    summary_df = (
        summary_df
        .sort_values("EBIC")
        .reset_index(drop=True)
    )


    return (
        results,
        summary_df
    )