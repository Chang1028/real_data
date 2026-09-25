import os

import numpy as np


# ============================================================
# Connectivity loader
# ============================================================

def load_connectivity_array(
    file_path,
    p
):

    arr = np.load(file_path).reshape((-1, p, p))

    # Preserve disk precision even when loaded before the solver/JAX config.
    X = np.asarray(np.transpose(arr, (1, 2, 0)), dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Connectivity contains nonfinite values: {file_path}")
    return X


def preprocess_X2(X2):
    """Log10-transform, normalize by the log10-diagonal, then zero the diagonal.

    X2 has shape (p, p, n). Entries with a zero normalization denominator
    are set to zero to avoid division by zero.
    """
    X2 = np.asarray(X2, dtype=np.float64)
    if X2.ndim != 3 or X2.shape[0] != X2.shape[1]:
        raise ValueError("X2 must have shape (p, p, n).")
    if not np.all(np.isfinite(X2)) or np.any(X2 <= -1):
        raise ValueError("X2 must be finite and greater than -1 before log10(1 + X2).")
    if np.any(np.diagonal(X2, axis1=0, axis2=1) < 0):
        raise ValueError("X2 diagonal must be nonnegative for square-root normalization.")
    X2 = np.log10(1 + X2)
    sqrt_diag = np.sqrt(np.diagonal(X2, axis1=0, axis2=1).T)
    denominator = sqrt_diag[:, None, :] * sqrt_diag[None, :, :]
    X2 = np.divide(
        X2, denominator, out=np.zeros_like(X2), where=denominator != 0
    )
    idx = np.arange(X2.shape[0])
    X2[idx, idx, :] = 0.0
    return X2


# ============================================================
# Simulation
# ============================================================

def generate_simulation_data(
    data_dir,
    p,
    n,
    sparsity1,
    sparsity2=None,
    snr=1.0,
    rs_file="z_rs_raw_ea.npy",
    emo_file="z_emo_raw_ea.npy",
    seed=2026,
    beta1_true=None,
    beta2_true=None,
):
    """
    Semi-synthetic simulation using real connectivity matrices.

    Parameters
    ----------
    sparsity1 : float
        Retained for call compatibility. The fixed ROI blocks below determine
        the current simulation support; this value does not change them.
    sparsity2 : float, optional
        Also retained for compatibility with the fixed-block experiment.
    beta1_true, beta2_true : array-like, optional
        Supply both to define another simulation. If omitted, the original
        fixed ROI blocks are used. Custom arrays are copied, not modified.
    """

    # Fallback if sparsity2 isn't explicitly passed
    if sparsity2 is None:
        sparsity2 = sparsity1

    rng = np.random.default_rng(seed)

    # --- Loading & Normalization Steps ---
    X1 = load_connectivity_array(os.path.join(data_dir, rs_file), p)
    X2 = load_connectivity_array(os.path.join(data_dir, emo_file), p)

    X2 = preprocess_X2(X2)

    available_n = min(X1.shape[2], X2.shape[2])
    if n is None:
        n = available_n
    if n > available_n:
        raise ValueError(f"Requested n={n}, but only {available_n} subjects exist.")

    X1 = X1[:, :, :n]
    X2 = X2[:, :, :n]

    # --- Centering ---
    X1_mean = np.mean(X1, axis=2, keepdims=True)
    X2_mean = np.mean(X2, axis=2, keepdims=True)

    X1_centered = X1 - X1_mean
    X2_centered = X2 - X2_mean

    # Preserve the original fixed blocks unless both custom vectors are supplied.
    if (beta1_true is None) != (beta2_true is None):
        raise ValueError("Supply both beta1_true and beta2_true, or neither.")
    if beta1_true is None:
        beta1_true = np.zeros((p, 1))
        beta2_true = np.zeros((p, 1))
        beta1_true[40:60] = 1.5
        beta1_true[120:150] = -1.0
        beta2_true[40:60] = 1.0
        beta2_true[120:150] = -1.2
    else:
        betas = []
        for name, beta in (("beta1_true", beta1_true), ("beta2_true", beta2_true)):
            beta = np.asarray(beta, dtype=np.float64)
            if beta.size != p or not np.all(np.isfinite(beta)):
                raise ValueError(f"{name} must contain {p} finite coefficients.")
            betas.append(beta.reshape(p, 1).copy())
        beta1_true, beta2_true = betas

    # --- Generate Signal & Noise ---
    def get_signal(X, beta):
        b = np.asarray(beta).ravel()
        return np.einsum("i,ijk,j->k", b, np.asarray(X), b)

    signal = get_signal(X1_centered, beta1_true) + get_signal(X2_centered, beta2_true)

    if snr <= 0:
        raise ValueError("snr must be positive.")

    noise_sd = np.std(signal) / snr
    noise = rng.normal(0, noise_sd, size=n)
    y = signal + noise

    return (
        np.asarray(X1, dtype=np.float64),
        np.asarray(X2, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        beta1_true,
        beta2_true
    )


# ============================================================
# Real data
# ============================================================

def load_fmri_data(
    data_dir,
    p,
    n_subjects=None,
    rs_file="z_rs_raw_ea.npy",
    emo_file="z_emo_raw_ea.npy",
    y_file="y_ea_pmat.npy",
    age_file="age_ea.npy"
):

    X1 = load_connectivity_array(
        os.path.join(
            data_dir,
            rs_file
        ),
        p
    )

    X2 = load_connectivity_array(
        os.path.join(
            data_dir,
            emo_file
        ),
        p
    )

    X2 = preprocess_X2(X2)


    y = np.load(
        os.path.join(
            data_dir,
            y_file
        )
    ).reshape(-1)

    age = np.load(
        os.path.join(
            data_dir,
            age_file
        )
    ).reshape(-1)

    n = min(
        X1.shape[2],
        X2.shape[2],
        len(y),
        len(age)
    )

    if n_subjects is not None:

        n = min(
            n,
            n_subjects
        )

    X1 = X1[:, :, :n]
    X2 = X2[:, :, :n]

    y = y[:n]
    age = age[:n]

    print(
        "Loaded data:"
        f"\nX1 = {X1.shape}"
        f"\nX2 = {X2.shape}"
        f"\ny = {y.shape}"
        f"\nage = {age.shape}"
    )

    return (
        np.asarray(X1, dtype=np.float64),
        np.asarray(X2, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(age, dtype=np.float64)
    )
