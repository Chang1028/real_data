"""Helpers for assigning simulated effects in the Schaefer 2018 atlas.

The bundled label file follows the standard
``Schaefer2018_200Parcels_7Networks_order`` ordering: row 1 corresponds to
ROI index 0 in the connectivity arrays, row 2 to ROI index 1, and so on.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ATLAS_FILE = Path(__file__).with_name(
    "schaefer2018_200parcels_7networks_order.txt"
)

NETWORK_DISPLAY_NAMES = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience / ventral attention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default mode",
}


def load_schaefer200_7networks(atlas_file=ATLAS_FILE):
    """Return the standard 200-row Schaefer 7-network parcel table.

    ``roi_index`` is zero-based and is the index to use for a NumPy beta
    vector.  This table is valid only when the connectivity matrices use the
    standard Schaefer ``..._order`` parcel ordering.
    """
    atlas = pd.read_csv(
        atlas_file,
        sep="\t",
        header=None,
        names=["atlas_index", "parcel", "red", "green", "blue", "alpha"],
    )
    if len(atlas) != 200 or not np.array_equal(atlas["atlas_index"].to_numpy(), np.arange(1, 201)):
        raise ValueError("Expected the standard Schaefer 200-parcel order file.")

    pieces = atlas["parcel"].str.split("_", expand=True)
    atlas["hemisphere"] = pieces[1]
    atlas["network"] = pieces[2]
    unexpected = set(atlas["network"]) - set(NETWORK_DISPLAY_NAMES)
    if unexpected:
        raise ValueError(f"Unrecognized Schaefer network labels: {sorted(unexpected)}")
    atlas["network_name"] = atlas["network"].map(NETWORK_DISPLAY_NAMES)
    atlas["roi_index"] = atlas["atlas_index"] - 1
    return atlas[
        ["roi_index", "atlas_index", "parcel", "hemisphere", "network", "network_name"]
    ].copy()


def betas_from_schaefer_networks(network_coefficients, *, p=200, atlas=None):
    """Create beta vectors by giving each Schaefer network one coefficient pair.

    Parameters
    ----------
    network_coefficients : mapping
        Mapping such as ``{"SomMot": {"beta1": 6, "beta2": -5}}``.
        Unlisted networks receive coefficient zero for both betas.
    p : int
        Must be 200, since this helper is specific to Schaefer 200.
    atlas : pandas.DataFrame, optional
        A previously loaded parcel table.

    Returns
    -------
    beta1, beta2, assignment_table
        The two length-200 vectors and a parcel-level table containing the
        assigned coefficients.
    """
    if p != 200:
        raise ValueError("Schaefer 2018 7-network helper requires p=200.")
    atlas = load_schaefer200_7networks() if atlas is None else atlas.copy()
    unknown = set(network_coefficients) - set(NETWORK_DISPLAY_NAMES)
    if unknown:
        allowed = ", ".join(NETWORK_DISPLAY_NAMES)
        raise ValueError(f"Unknown network(s): {sorted(unknown)}. Use: {allowed}.")

    beta1 = np.zeros(p, dtype=float)
    beta2 = np.zeros(p, dtype=float)
    for network, values in network_coefficients.items():
        if not isinstance(values, dict) or set(values) - {"beta1", "beta2"}:
            raise ValueError(
                f"{network!r} must map to a dictionary containing beta1 and/or beta2."
            )
        mask = atlas["network"].eq(network).to_numpy()
        beta1[mask] = float(values.get("beta1", 0.0))
        beta2[mask] = float(values.get("beta2", 0.0))

    assignment_table = atlas.copy()
    assignment_table["beta1"] = beta1
    assignment_table["beta2"] = beta2
    return beta1, beta2, assignment_table


def betas_from_schaefer_parcel_selections(selections, *, p=200, atlas=None):
    """Create sparse beta vectors from selected parcels in named 7-networks.

    ``selections`` is a list such as::

        [{"network": "SomMot", "roi_indices": [14, 15, 16],
          "beta1": 6, "beta2": -5}]

    ROI indices are zero-based.  The function validates that each selected ROI
    belongs to its declared network, which prevents an accidental mismatch
    between a manually chosen index and the Schaefer ordering.
    """
    if p != 200:
        raise ValueError("Schaefer 2018 7-network helper requires p=200.")
    atlas = load_schaefer200_7networks() if atlas is None else atlas.copy()
    network_by_roi = atlas.set_index("roi_index")["network"]
    beta1 = np.zeros(p, dtype=float)
    beta2 = np.zeros(p, dtype=float)

    for item in selections:
        required = {"network", "roi_indices"}
        missing = required - set(item)
        if missing:
            raise ValueError(f"Each selection needs {sorted(required)}; missing {sorted(missing)}.")
        network = item["network"]
        if network not in NETWORK_DISPLAY_NAMES:
            raise ValueError(f"Unknown network {network!r}.")
        roi_indices = np.asarray(item["roi_indices"], dtype=int)
        if roi_indices.ndim != 1 or len(roi_indices) == 0:
            raise ValueError("roi_indices must be a non-empty one-dimensional list.")
        if np.any((roi_indices < 0) | (roi_indices >= p)):
            raise ValueError("ROI indices must be between 0 and 199.")
        wrong_network = [
            int(roi) for roi in roi_indices if network_by_roi.loc[int(roi)] != network
        ]
        if wrong_network:
            actual = {roi: network_by_roi.loc[roi] for roi in wrong_network}
            raise ValueError(
                f"Selected ROI(s) do not belong to {network}: {actual}."
            )
        beta1[roi_indices] = float(item.get("beta1", 0.0))
        beta2[roi_indices] = float(item.get("beta2", 0.0))

    assignment_table = atlas.copy()
    assignment_table["selected"] = False
    assignment_table.loc[assignment_table["roi_index"].isin(np.flatnonzero((beta1 != 0) | (beta2 != 0))), "selected"] = True
    assignment_table["beta1"] = beta1
    assignment_table["beta2"] = beta2
    return beta1, beta2, assignment_table
