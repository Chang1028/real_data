import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# X1 / X2 heatmaps
# ============================================================

def plot_X1_X2_heatmaps(
    X1,
    X2,
    subject=0,
    cmap="RdBu_r",
    save_path=None
):

    X1_mat = np.asarray(
        X1[:, :, subject]
    )

    X2_mat = np.asarray(
        X2[:, :, subject]
    )

    vmax = max(
        np.abs(X1_mat).max(),
        np.abs(X2_mat).max()
    )

    vmin = -vmax

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 3.5),
        constrained_layout=True
    )

    matrices = [
        X1_mat,
        X2_mat
    ]

    titles = [
        r"$X_1$",
        r"$X_2$"
    ]

    for ax, matrix, title in zip(
        axes,
        matrices,
        titles
    ):

        im = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        ax.set_title(
            title,
            fontsize=13
        )

        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")

        divider = make_axes_locatable(ax)

        cax = divider.append_axes(
            "right",
            size="4%",
            pad=0.15
        )

        plt.colorbar(
            im,
            cax=cax
        )

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()



def plot_average_X1_X2_heatmaps(
    X1,
    X2,
    cmap="RdBu_r",
    save_path=None
):
    """
    Plot the element-wise average connectivity matrices
    across all subjects for X1 and X2.

    Expected shapes:
        X1: (p, p, n_subjects)
        X2: (p, p, n_subjects)

    Returns
    -------
    X1_mean : np.ndarray
        p x p average X1 matrix.

    X2_mean : np.ndarray
        p x p average X2 matrix.
    """

    # ========================================================
    # Convert to NumPy
    # ========================================================

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)


    # ========================================================
    # Average across subjects
    # ========================================================

    X1_mean = np.mean(
        X1,
        axis=2
    )

    X2_mean = np.mean(
        X2,
        axis=2
    )


    # ========================================================
    # Use same color scale for X1 and X2
    # ========================================================

    vmax = max(
        np.abs(X1_mean).max(),
        np.abs(X2_mean).max()
    )

    vmin = -vmax


    # ========================================================
    # Plot
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 3.5),
        constrained_layout=True
    )

    matrices = [
        X1_mean,
        X2_mean
    ]

    titles = [
        r"Mean $X_1$ across subjects",
        r"Mean $X_2$ across subjects"
    ]


    for ax, matrix, title in zip(
        axes,
        matrices,
        titles
    ):

        im = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        ax.set_title(
            title,
            fontsize=13
        )

        ax.set_xlabel(
            "ROI"
        )

        ax.set_ylabel(
            "ROI"
        )


        # Separate colorbar for each panel
        divider = make_axes_locatable(
            ax
        )

        cax = divider.append_axes(
            "right",
            size="4%",
            pad=0.15
        )

        plt.colorbar(
            im,
            cax=cax
        )


    # ========================================================
    # Save
    # ========================================================

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()


    # Also return the average matrices
    return X1_mean, X2_mean

# ============================================================
# Full train/test grid for all lambdas
# ============================================================

def plot_train_test_grid_all_lambdas(
    results,
    save_path=None
):
    """
    Creates:
        2 * number_of_lambdas rows
        x
        number_of_folds columns

    For each lambda:
        first row  = training predictions
        second row = test predictions
    """

    n_lambdas = len(results)
    n_folds = len(
        results[0]["folds"]
    )

    fig, axes = plt.subplots(
        2 * n_lambdas,
        n_folds,
        figsize=(
            3.0 * n_folds,
            2.3 * 2 * n_lambdas
        ),
        squeeze=False
    )

    for lambda_idx, result in enumerate(results):

        lam = result["lambda"]
        folds = result["folds"]

        train_row = 2 * lambda_idx
        test_row = train_row + 1

        # ----------------------------------------------------
        # Shared range for train + test plots of this lambda
        # ----------------------------------------------------

        all_values = []

        for fold in folds:

            for key in [
                "y_train_true",
                "y_train_pred",
                "y_true",
                "y_pred"
            ]:

                all_values.extend(
                    np.asarray(
                        fold[key]
                    ).ravel()
                )

        all_values = np.asarray(
            all_values
        )

        mn = np.min(all_values)
        mx = np.max(all_values)

        padding = (
            0.05
            *
            (mx - mn)
        )

        if padding == 0:
            padding = 1

        mn -= padding
        mx += padding

        # ----------------------------------------------------
        # Five folds
        # ----------------------------------------------------

        for fold_idx, fold in enumerate(folds):

            # ================================================
            # Training
            # ================================================

            ax = axes[
                train_row,
                fold_idx
            ]

            y_true_train = np.asarray(
                fold["y_train_true"]
            ).ravel()

            y_pred_train = np.asarray(
                fold["y_train_pred"]
            ).ravel()

            ax.scatter(
                y_true_train,
                y_pred_train,
                s=8,
                alpha=0.45
            )

            ax.plot(
                [mn, mx],
                [mn, mx],
                "--",
                linewidth=1
            )

            ax.set_xlim(
                mn,
                mx
            )

            ax.set_ylim(
                mn,
                mx
            )

            ax.set_title(
                f"Fold {fold_idx + 1}\n"
                f"Train $R^2$={fold['train_r2']:.2f}\n"
                f"r={fold['train_corr']:.2f}",
                fontsize=8
            )

            if fold_idx == 0:

                ax.set_ylabel(
                    f"$\\lambda$={lam}\n"
                    "Train\nPredicted y",
                    fontsize=8
                )

            ax.tick_params(
                labelsize=7
            )

            # ================================================
            # Test
            # ================================================

            ax = axes[
                test_row,
                fold_idx
            ]

            y_true_test = np.asarray(
                fold["y_true"]
            ).ravel()

            y_pred_test = np.asarray(
                fold["y_pred"]
            ).ravel()

            ax.scatter(
                y_true_test,
                y_pred_test,
                s=12,
                alpha=0.6
            )

            ax.plot(
                [mn, mx],
                [mn, mx],
                "--",
                linewidth=1
            )

            ax.set_xlim(
                mn,
                mx
            )

            ax.set_ylim(
                mn,
                mx
            )

            ax.set_title(
                f"Fold {fold_idx + 1}\n"
                f"Test $R^2$={fold['val_r2']:.2f}\n"
                f"r={fold['val_corr']:.2f}",
                fontsize=8
            )

            if fold_idx == 0:

                ax.set_ylabel(
                    f"$\\lambda$={lam}\n"
                    "Test\nPredicted y",
                    fontsize=8
                )

            ax.set_xlabel(
                "True y",
                fontsize=8
            )

            ax.tick_params(
                labelsize=7
            )

    fig.suptitle(
        "Training and validation predictions across lambdas",
        fontsize=14
    )

    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# Compact plot:
# mean train/test R2 versus lambda
# ============================================================

def plot_r2_over_lambda(
    results,
    save_path=None
):
    """
    Compact summary of train and test R2
    across lambda values.
    """

    lambdas = np.array(
        [
            r["lambda"]
            for r in results
        ],
        dtype=float
    )

    train_r2 = np.array(
        [
            r["mean_train_r2"]
            for r in results
        ]
    )

    test_r2 = np.array(
        [
            r["mean_r2"]
            for r in results
        ]
    )

    order = np.argsort(
        lambdas
    )

    lambdas = lambdas[order]
    train_r2 = train_r2[order]
    test_r2 = test_r2[order]

    plt.figure(
        figsize=(6, 4)
    )

    plt.plot(
        lambdas,
        train_r2,
        marker="o",
        linewidth=1.5,
        label="Train"
    )

    plt.plot(
        lambdas,
        test_r2,
        marker="o",
        linewidth=1.5,
        label="Test"
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    # Use log scale only when all lambdas are positive
    if np.all(
        lambdas > 0
    ):
        plt.xscale(
            "log"
        )

    plt.xlabel(
        r"$\lambda$"
    )

    plt.ylabel(
        r"$R^2$"
    )

    plt.title(
        r"Prediction performance across $\lambda$"
    )

    plt.legend()

    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# Compact plot:
# fold-specific test R2 for each lambda
# ============================================================

def plot_fold_r2_by_lambda(
    results,
    save_path=None
):
    """
    Each point = one CV fold.
    Diamond = mean test R2 for that lambda.
    """

    plt.figure(
        figsize=(7, 4)
    )

    for i, result in enumerate(results):

        fold_r2 = np.array(
            [
                fold["val_r2"]
                for fold in result["folds"]
            ]
        )

        x = np.repeat(
            i,
            len(fold_r2)
        )

        # Small jitter so overlapping points are visible
        jitter = np.linspace(
            -0.08,
            0.08,
            len(fold_r2)
        )

        plt.scatter(
            x + jitter,
            fold_r2,
            s=35,
            alpha=0.7
        )

        plt.scatter(
            i,
            np.mean(fold_r2),
            marker="D",
            s=65
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xticks(
        range(
            len(results)
        ),
        [
            f"{r['lambda']:.3g}"
            for r in results
        ]
    )

    plt.xlabel(
        r"$\lambda$"
    )

    plt.ylabel(
        r"Test $R^2$"
    )

    plt.title(
        "Fold-specific test performance"
    )

    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# Correlation map
# ============================================================

def correlation_map(
    y,
    X
):

    y = np.asarray(
        y,
        dtype=float
    ).ravel()

    X = np.asarray(
        X,
        dtype=float
    )

    if X.ndim != 3:

        raise ValueError(
            "X must have shape "
            "(ROI, ROI, subjects)."
        )

    if X.shape[2] != len(y):

        raise ValueError(
            f"Subject mismatch: "
            f"X has {X.shape[2]} subjects, "
            f"but y has {len(y)} values."
        )

    y_c = (
        y
        -
        np.mean(y)
    )

    X_c = (
        X
        -
        np.mean(
            X,
            axis=2,
            keepdims=True
        )
    )

    numerator = np.sum(
        X_c
        *
        y_c[
            None,
            None,
            :
        ],
        axis=2
    )

    denominator = np.sqrt(
        np.sum(
            X_c ** 2,
            axis=2
        )
        *
        np.sum(
            y_c ** 2
        )
    )

    r_map = np.divide(
        numerator,
        denominator,
        out=np.full_like(
            numerator,
            np.nan,
            dtype=float
        ),
        where=denominator != 0
    )

    return r_map



def plot_correlation_distributions(
    y,
    X1,
    X2,
    y_name="Age",
    n_bins=50,
    save_path=None
):
    """
    Plot distributions of edge-wise Pearson correlations between
    y and X1 / X2.

    Parameters
    ----------
    y : array-like, shape (n_subjects,)
        Subject-level variable, e.g. age.

    X1 : array, shape (n_roi, n_roi, n_subjects)
        First connectivity array.

    X2 : array, shape (n_roi, n_roi, n_subjects)
        Second connectivity array.

    y_name : str, default="Age"
        Label used in plot titles.

    n_bins : int, default=50
        Number of histogram bins.

    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    r_X1 : array, shape (n_roi, n_roi)
        Correlation map for X1.

    r_X2 : array, shape (n_roi, n_roi)
        Correlation map for X2.
    """

    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------
    if X1.ndim != 3 or X2.ndim != 3:
        raise ValueError(
            "X1 and X2 must have shape "
            "(ROI, ROI, subjects)."
        )

    if X1.shape[0] != X1.shape[1]:
        raise ValueError(
            "X1 must contain square ROI x ROI matrices."
        )

    if X2.shape[0] != X2.shape[1]:
        raise ValueError(
            "X2 must contain square ROI x ROI matrices."
        )

    if X1.shape[:2] != X2.shape[:2]:
        raise ValueError(
            "X1 and X2 must have the same ROI dimensions."
        )

    # --------------------------------------------------------
    # Correlation maps
    # --------------------------------------------------------
    r_X1 = correlation_map(y, X1)
    r_X2 = correlation_map(y, X2)

    # --------------------------------------------------------
    # Extract upper triangular edges only
    # Excludes diagonal
    # --------------------------------------------------------
    p = X1.shape[0]

    upper_idx = np.triu_indices(
        p,
        k=1
    )

    r1 = r_X1[upper_idx]
    r2 = r_X2[upper_idx]

    # --------------------------------------------------------
    # Remove NaN / infinite values
    # --------------------------------------------------------
    r1 = r1[np.isfinite(r1)]
    r2 = r2[np.isfinite(r2)]

    if len(r1) == 0 or len(r2) == 0:
        raise ValueError(
            "No finite correlation values are available to plot."
        )

    # --------------------------------------------------------
    # Shared histogram bins
    # --------------------------------------------------------
    lower = min(
        r1.min(),
        r2.min()
    )

    upper = max(
        r1.max(),
        r2.max()
    )

    bins = np.linspace(
        lower,
        upper,
        n_bins + 1
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
        sharex=True,
        sharey=True
    )

    axes[0].hist(
        r1,
        bins=bins,
        edgecolor="black"
    )

    axes[0].axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    axes[0].set_title(
        f"X1 correlations with {y_name}"
    )

    axes[0].set_xlabel(
        "Pearson correlation"
    )

    axes[0].set_ylabel(
        "Number of ROI pairs"
    )

    axes[1].hist(
        r2,
        bins=bins,
        edgecolor="black"
    )

    axes[1].axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    axes[1].set_title(
        f"X2 correlations with {y_name}"
    )

    axes[1].set_xlabel(
        "Pearson correlation"
    )

    fig.suptitle(
        f"Distributions of edge-wise correlations with {y_name}"
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return r_X1, r_X2


# ============================================================
# Full train/test scatter grid for 2D lambdas
# ============================================================

def plot_train_test_grid_all_lambdas(
    results,
    save_path=None
):
    """
    Plot train/test predictions for every (lambda1, lambda2) pair.

    Each lambda pair uses two rows:
        row 1 = training predictions across folds
        row 2 = test predictions across folds
    """

    n_pairs = len(results)
    n_folds = len(results[0]["folds"])

    fig, axes = plt.subplots(
        2 * n_pairs,
        n_folds,
        figsize=(
            3.0 * n_folds,
            4.2 * n_pairs
        ),
        squeeze=False
    )

    for pair_idx, result in enumerate(results):

        lam1 = result["lambda1"]
        lam2 = result["lambda2"]

        folds = result["folds"]

        train_row = 2 * pair_idx
        test_row = train_row + 1

        # ----------------------------------------------------
        # Shared plot limits for this lambda pair
        # ----------------------------------------------------

        all_values = []

        for fold in folds:

            for key in [
                "y_train_true",
                "y_train_pred",
                "y_true",
                "y_pred"
            ]:

                all_values.extend(
                    np.asarray(
                        fold[key]
                    ).ravel()
                )

        all_values = np.asarray(all_values)

        mn = np.min(all_values)
        mx = np.max(all_values)

        padding = 0.05 * (mx - mn)

        if padding == 0:
            padding = 1

        mn -= padding
        mx += padding

        # ----------------------------------------------------
        # Individual folds
        # ----------------------------------------------------

        for fold_idx, fold in enumerate(folds):

            # =================================================
            # Training
            # =================================================

            ax = axes[
                train_row,
                fold_idx
            ]

            y_true_train = np.asarray(
                fold["y_train_true"]
            ).ravel()

            y_pred_train = np.asarray(
                fold["y_train_pred"]
            ).ravel()

            ax.scatter(
                y_true_train,
                y_pred_train,
                s=8,
                alpha=0.45
            )

            ax.plot(
                [mn, mx],
                [mn, mx],
                "--",
                linewidth=1
            )

            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)

            ax.set_title(
                f"Fold {fold_idx + 1}\n"
                f"Train $R^2$={fold['train_r2']:.2f}\n"
                f"r={fold['train_corr']:.2f}",
                fontsize=8
            )

            if fold_idx == 0:

                ax.set_ylabel(
                    rf"$\lambda_1={lam1:g}$"
                    "\n"
                    rf"$\lambda_2={lam2:g}$"
                    "\nTrain predicted y",
                    fontsize=8
                )

            ax.tick_params(
                labelsize=7
            )

            # =================================================
            # Test
            # =================================================

            ax = axes[
                test_row,
                fold_idx
            ]

            y_true_test = np.asarray(
                fold["y_true"]
            ).ravel()

            y_pred_test = np.asarray(
                fold["y_pred"]
            ).ravel()

            ax.scatter(
                y_true_test,
                y_pred_test,
                s=12,
                alpha=0.60
            )

            ax.plot(
                [mn, mx],
                [mn, mx],
                "--",
                linewidth=1
            )

            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)

            ax.set_title(
                f"Fold {fold_idx + 1}\n"
                f"Test $R^2$={fold['val_r2']:.2f}\n"
                f"r={fold['val_corr']:.2f}",
                fontsize=8
            )

            if fold_idx == 0:

                ax.set_ylabel(
                    rf"$\lambda_1={lam1:g}$"
                    "\n"
                    rf"$\lambda_2={lam2:g}$"
                    "\nTest predicted y",
                    fontsize=8
                )

            ax.set_xlabel(
                "True y",
                fontsize=8
            )

            ax.tick_params(
                labelsize=7
            )

    fig.suptitle(
        "Training and validation predictions across "
        r"$(\lambda_1,\lambda_2)$",
        fontsize=14
    )

    plt.tight_layout()

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# 2D heatmap of lambda performance
# ============================================================

def plot_lambda_heatmap(
    summary_df,
    metric="Test R2",
    save_path=None,
    annotate=True
):
    """
    Plot a heatmap over lambda1 x lambda2.

    Examples
    --------
    metric="Test R2"
    metric="Test Corr"
    metric="Test MSE"
    metric="EBIC"
    metric="Avg Features"
    """

    table = summary_df.pivot(
        index="lambda1",
        columns="lambda2",
        values=metric
    )

    # Make sure lambda values are ordered numerically
    table = table.sort_index(
        axis=0
    ).sort_index(
        axis=1
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        table.values,
        aspect="auto",
        origin="lower"
    )

    ax.set_xticks(
        np.arange(
            len(table.columns)
        )
    )

    ax.set_xticklabels(
        [
            f"{x:g}"
            for x in table.columns
        ]
    )

    ax.set_yticks(
        np.arange(
            len(table.index)
        )
    )

    ax.set_yticklabels(
        [
            f"{x:g}"
            for x in table.index
        ]
    )

    ax.set_xlabel(
        r"$\lambda_2$"
    )

    ax.set_ylabel(
        r"$\lambda_1$"
    )

    ax.set_title(
        metric
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        metric
    )

    # --------------------------------------------------------
    # Put metric values inside cells
    # --------------------------------------------------------

    if annotate:

        for i in range(
            table.shape[0]
        ):

            for j in range(
                table.shape[1]
            ):

                value = table.iloc[
                    i,
                    j
                ]

                if np.isfinite(value):

                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8
                    )

    plt.tight_layout()

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# Heatmaps for several important metrics at once
# ============================================================

def plot_lambda_summary_heatmaps(
    summary_df
):
    """
    Convenience function for examining the 2D lambda grid.
    """

    plot_lambda_heatmap(
        summary_df,
        metric="Test R2"
    )

    plot_lambda_heatmap(
        summary_df,
        metric="Test Corr"
    )

    plot_lambda_heatmap(
        summary_df,
        metric="EBIC"
    )

    plot_lambda_heatmap(
        summary_df,
        metric="Avg Features"
    )


# ============================================================
# Fold-specific test R2 for lambda pairs
# ============================================================

def plot_fold_r2_by_lambda_pair(
    results,
    save_path=None
):
    """
    Each group corresponds to one (lambda1, lambda2) pair.

    Small points = individual CV folds.
    Diamond = mean test R2.
    """

    fig, ax = plt.subplots(
        figsize=(
            max(
                8,
                len(results) * 0.7
            ),
            4.5
        )
    )

    labels = []

    for i, result in enumerate(results):

        lam1 = result["lambda1"]
        lam2 = result["lambda2"]

        fold_r2 = np.asarray(
            [
                fold["val_r2"]
                for fold in result["folds"]
            ]
        )

        jitter = np.linspace(
            -0.08,
            0.08,
            len(fold_r2)
        )

        ax.scatter(
            i + jitter,
            fold_r2,
            s=30,
            alpha=0.7
        )

        ax.scatter(
            i,
            np.nanmean(fold_r2),
            marker="D",
            s=60
        )

        labels.append(
            f"({lam1:g}, {lam2:g})"
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xticks(
        range(
            len(results)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_xlabel(
        r"$(\lambda_1,\lambda_2)$"
    )

    ax.set_ylabel(
        r"Test $R^2$"
    )

    ax.set_title(
        "Fold-specific test performance"
    )

    plt.tight_layout()

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# Train/test R2 by lambda pair
# ============================================================

def plot_r2_by_lambda_pair(
    results,
    save_path=None
):
    """
    Compact comparison of mean train and test R2
    for all lambda pairs.
    """

    labels = [
        (
            f"({r['lambda1']:g}, "
            f"{r['lambda2']:g})"
        )
        for r in results
    ]

    train_r2 = np.asarray(
        [
            r["mean_train_r2"]
            for r in results
        ]
    )

    test_r2 = np.asarray(
        [
            r["mean_r2"]
            for r in results
        ]
    )

    x = np.arange(
        len(results)
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                8,
                0.7 * len(results)
            ),
            4.5
        )
    )

    ax.plot(
        x,
        train_r2,
        marker="o",
        label="Train"
    )

    ax.plot(
        x,
        test_r2,
        marker="o",
        label="Test"
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_xlabel(
        r"$(\lambda_1,\lambda_2)$"
    )

    ax.set_ylabel(
        r"$R^2$"
    )

    ax.set_title(
        "Prediction performance across lambda pairs"
    )

    ax.legend()

    plt.tight_layout()

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()