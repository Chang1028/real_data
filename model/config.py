import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PARAMETER_FILE = BASE_DIR / "parameters.json"


def load_parameters(
    json_file=PARAMETER_FILE
):

    with open(
        json_file,
        "r"
    ) as f:

        params = json.load(f)

    if isinstance(
        params,
        list
    ):
        params = params[0]

    return params


PARAMS = load_parameters()


MAX_ITER = PARAMS["MAX_ITER"]
STEP_SIZE = PARAMS["STEP_SIZE"]
TOL = PARAMS["TOL"]

P = PARAMS["P"]
N_SUBJECTS = PARAMS["N_SUBJECTS"]

SPARSITY1 = PARAMS["SPARSITY1"]
SPARSITY2 = PARAMS["SPARSITY2"]
SNR = PARAMS["SNR"]

N_SPLITS = PARAMS["N_SPLITS"]
N_JOBS = PARAMS["N_JOBS"]

DATA_DIR = PARAMS["DATA_DIR"]


LAMBDA1_GRID = PARAMS["LAMBDA1_GRID"]
LAMBDA2_GRID = PARAMS["LAMBDA2_GRID"]

NUM_CANDIDATES = PARAMS.get("NUM_CANDIDATES", 2)
INITIALIZATION = PARAMS.get("INITIALIZATION", "scaled_random")
