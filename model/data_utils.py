import os

import numpy as np
import jax.numpy as jnp


# ============================================================
# Connectivity loader
# ============================================================

def load_connectivity_array(
    file_path,
    p
):

    arr = np.load(file_path).reshape((-1, p, p))

    X = jnp.array(np.transpose(arr, (1, 2, 0)))
    
    return jnp.asarray(X)


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
    seed=2026
):
    """
    Semi-synthetic simulation using real connectivity matrices.

    Parameters
    ----------
    sparsity1 : float
        Sparsity for beta1_true (proportion of non-zero ROIs).
    sparsity2 : float, optional
        Sparsity for beta2_true. If None, defaults to sparsity1.
    """
    
    # Fallback if sparsity2 isn't explicitly passed
    if sparsity2 is None:
        sparsity2 = sparsity1

    rng = np.random.default_rng(seed)

    # --- Loading & Normalization Steps ---
    X1 = load_connectivity_array(os.path.join(data_dir, rs_file), p)
    X2 = load_connectivity_array(os.path.join(data_dir, emo_file), p)

    X2 = np.log10(1 + X2)
    diag = np.diagonal(X2, axis1=0, axis2=1).T
    sqrt_diag = np.sqrt(diag)

    denominator = sqrt_diag[:, None, :] * sqrt_diag[None, :, :]
    threshold = 1e-3
    valid_mask = denominator > threshold

    X2 = np.where(valid_mask, X2 / np.where(valid_mask, denominator, 1.0), 0.0)

    for i in range(X2.shape[2]):
        np.fill_diagonal(X2[:, :, i], 0)

    available_n = min(X1.shape[2], X2.shape[2])
    if n is None:
        n = available_n
    if n > available_n:
        raise ValueError(f"Requested n={n}, but only {available_n} subjects exist.")

    X1 = X1[:, :, :n]
    X2 = X2[:, :, :n]

    # --- Centering ---
    X1_mean = jnp.mean(X1, axis=2, keepdims=True)
    X2_mean = jnp.mean(X2, axis=2, keepdims=True)

    X1_centered = X1 - X1_mean
    X2_centered = X2 - X2_mean

    # --- Sparse Ground Truth Generation ---
    beta1_true = np.zeros((p, 1))
    beta2_true = np.zeros((p, 1))

    # # Calculate independent number of non-zero entries
    # n_nonzero1 = max(1, int(round(p * sparsity1)))
    # n_nonzero2 = max(1, int(round(p * sparsity2)))

    # # ROI variability / strength
    # strength1 = np.sum(np.abs(np.asarray(X1_centered)), axis=(1, 2))
    # strength2 = np.sum(np.abs(np.asarray(X2_centered)), axis=(1, 2))

    # # Select top ROIs independently according to their respective sparsity
    # idx1 = np.argsort(strength1)[-n_nonzero1:]
    # idx2 = np.argsort(strength2)[-n_nonzero2:]

    # signs1 = rng.choice([-1, 1], size=(n_nonzero1, 1))
    # signs2 = rng.choice([-1, 1], size=(n_nonzero2, 1))

    # beta1_true[idx1] = rng.uniform(0.5, 2.0, size=(n_nonzero1, 1)) * signs1
    # beta2_true[idx2] = rng.uniform(0.5, 2.0, size=(n_nonzero2, 1)) * signs2

    beta1_true[40:60] = 1.5
    beta1_true[120:150] = -1.0

    beta2_true[40:60] = 1
    beta2_true[120:150] = -1.2

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
        jnp.asarray(X1),
        jnp.asarray(X2),
        jnp.asarray(y),
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

    X2 = np.log10(1 + X2)
    for i in range(X2.shape[2]):
        np.fill_diagonal(X2[:, :, i], 0)


    y = np.load(
        os.path.join(
            data_dir,
            y_file
        )
    ).squeeze()

    age = np.load(
        os.path.join(
            data_dir,
            age_file
        )
    ).squeeze()

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
        jnp.asarray(X1),
        jnp.asarray(X2),
        jnp.asarray(y),
        jnp.asarray(age)
    )