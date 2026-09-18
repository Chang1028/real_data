import numpy as np


def predict_bilinear(
    X1,
    X2,
    beta1,
    beta2
):

    b1 = np.asarray(
        beta1
    ).ravel()

    b2 = np.asarray(
        beta2
    ).ravel()

    pred1 = np.einsum(
        "i,ijk,j->k",
        b1,
        np.asarray(X1),
        b1
    )

    pred2 = np.einsum(
        "i,ijk,j->k",
        b2,
        np.asarray(X2),
        b2
    )

    return pred1 + pred2


def compute_r2(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    ).ravel()

    y_pred = np.asarray(
        y_pred
    ).ravel()

    rss = np.sum(
        (y_true - y_pred) ** 2
    )

    tss = np.sum(
        (
            y_true
            -
            np.mean(y_true)
        ) ** 2
    )

    if tss < 1e-12:
        return np.nan

    return 1 - rss / tss


def compute_corr(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    ).ravel()

    y_pred = np.asarray(
        y_pred
    ).ravel()

    if (
        np.std(y_true) < 1e-12
        or
        np.std(y_pred) < 1e-12
    ):
        return np.nan

    return np.corrcoef(
        y_true,
        y_pred
    )[0, 1]


def evaluate_prediction(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    ).ravel()

    y_pred = np.asarray(
        y_pred
    ).ravel()

    mse = np.mean(
        (y_true - y_pred) ** 2
    )

    return {
        "mse": float(mse),
        "r2": float(
            compute_r2(
                y_true,
                y_pred
            )
        ),
        "corr": float(
            compute_corr(
                y_true,
                y_pred
            )
        )
    }


def calculate_ebic(
    mse,
    n_features,
    n_samples,
    total_p,
    gamma=0.5
):

    if mse <= 1e-12:
        return -np.inf, -np.inf

    n = n_samples
    k = n_features

    bic = (
        n * np.log(mse)
        +
        k * np.log(n)
    )

    ebic = (
        bic
        +
        2
        * gamma
        * k
        * np.log(total_p)
    )

    return bic, ebic