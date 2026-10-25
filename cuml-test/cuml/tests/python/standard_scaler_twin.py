"""Twin of cuml/tests/standard-scaler-twin-test.rkt: sklearn's StandardScaler
fits and transforms the rows the Racket side used."""

import json
import sys

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

BITS = {"float32": np.uint32, "float64": np.uint64}


def rows(case, key, dtype):
    return np.asarray(case[key], dtype=dtype).reshape(-1, case["d"])


def scaler(case):
    return StandardScaler(with_mean=case["with_mean"], with_std=case["with_std"])


def vector(v):
    return None if v is None else np.asarray(v, dtype=np.float64).tolist()


def fit(case):
    fitted = scaler(case).fit(rows(case, "X", case["dtype"]))
    return {
        "mean": vector(fitted.mean_),
        "var": vector(fitted.var_),
        "scale": vector(fitted.scale_),
        "n_samples": int(fitted.n_samples_seen_),
    }


def transform(case):
    fitted = scaler(case).fit(rows(case, "X", case["dtype"]))
    Y = rows(case, "Y", case["y_dtype"])
    t = fitted.transform(Y)
    back = fitted.inverse_transform(t)
    bits = BITS[case["y_dtype"]]
    return {
        "dtype": str(t.dtype),
        "transform": t.astype(np.float64).ravel().tolist(),
        "inverse": back.astype(np.float64).ravel().tolist(),
        "transform_bits": [int(b) for b in t.ravel().view(bits)],
        "inverse_bits": [int(b) for b in back.ravel().view(bits)],
    }


def params(case):
    return StandardScaler().get_params()


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {"fit": fit, "transform": transform, "params": params}
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": sklearn.__version__}, f)


if __name__ == "__main__":
    main()
