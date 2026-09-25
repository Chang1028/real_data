from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config import (
    DATA_DIR,
    P,
    N_SUBJECTS,
    SPARSITY1,
    SPARSITY2,
    SNR
)

from data_utils import (
    generate_simulation_data,
    load_fmri_data
)

from cross_validation import (
    run_cv_over_lambdas
)

from plot_utils import (
    plot_X1_X2_heatmaps,
    plot_train_test_grid_all_lambdas,
    plot_r2_by_lambda_pair,
    plot_fold_r2_by_lambda_pair
)


def main():
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(exist_ok=True)
    # ============================================================
    # 1. Simulation data
    # ============================================================

    X1_sim, X2_sim, y_sim, beta1_true, beta2_true = (
        generate_simulation_data(
            data_dir=DATA_DIR,
            p=P,
            n=N_SUBJECTS,
            sparsity1=SPARSITY1,
            sparsity2=SPARSITY2,
            snr=SNR,
            rs_file="z_fc.npy",
            emo_file="z_sc.npy",
        )
    )


    # ============================================================
    # 2. Real data
    # ============================================================

    X1, X2, y, age = load_fmri_data(
        data_dir=DATA_DIR,
        p=P,
        n_subjects=N_SUBJECTS,

        # Change these for PMAT / WRAT / sex-specific analyses
        rs_file="z_fc.npy",
        emo_file="z_sc.npy",
        y_file="y.npy",
        age_file="age.npy"
    )


    # ============================================================
    # 3. Inspect outcome distributions
    # ============================================================

    y_real_c = (
        np.asarray(y)
        -
        np.mean(y)
    )

    y_sim_c = (
        np.asarray(y_sim)
        -
        np.mean(y_sim)
    )

    age_c = (
        np.asarray(age)
        -
        np.mean(age)
    )

    plt.figure(
        figsize=(7, 4)
    )

    plt.hist(
        age_c,
        bins=30,
        alpha=0.5,
        density=True,
        label="Real y"
    )

    plt.hist(
        y_sim_c,
        bins=30,
        alpha=0.5,
        density=True,
        label="Simulated y"
    )

    plt.xlabel(
        "Centered outcome"
    )

    plt.ylabel(
        "Density"
    )

    plt.legend()

    plt.title(
        "Real and simulated outcomes"
    )

    plt.show()


    # ============================================================
    # 4. Inspect connectivity matrices
    # ============================================================

    plot_X1_X2_heatmaps(
        X1,
        X2,
        subject=20
    )


    # ============================================================
    # 5. Simulation CV
    # ============================================================

    simulation_results, simulation_summary = (
        run_cv_over_lambdas(
            X1=X1_sim,
            X2=X2_sim,
            y=y_sim,

            seed=1999
        )
    )

    print(
        "\nSimulation CV results"
    )

    print(
        simulation_summary
    )


    # ============================================================
    # 6. Real-data CV
    # ============================================================

    real_results, real_summary = (
        run_cv_over_lambdas(
            X1=X1,
            X2=X2,
            y=age_c,

            seed=1999
        )
    )

    print(
        "\nReal-data CV results"
    )

    print(
        real_summary
    )


    # ============================================================
    # 7. CV plots
    # ============================================================
    plot_train_test_grid_all_lambdas(
        real_results,
        save_path=figure_dir / "train_test_grid_all_lambdas.pdf"
    )


    plot_r2_by_lambda_pair(
        real_results,
        save_path=figure_dir / "r2_vs_lambda.pdf"
    )


    plot_fold_r2_by_lambda_pair(
        real_results,
        save_path=figure_dir / "fold_r2_vs_lambda.pdf"
    )




if __name__ == "__main__":
    main()
