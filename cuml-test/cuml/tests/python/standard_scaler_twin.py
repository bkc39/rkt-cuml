"""Twin of cuml/tests/standard-scaler-twin-test.rkt: sklearn's StandardScaler
fits and transforms the rows the Racket side used."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from twin_common import params_of, put, run


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
    t = fitted.transform(rows(case, "Y", case["y_dtype"]))
    out = {"dtype": str(t.dtype)}
    put(out, "transform", t, float64=True)
    put(out, "inverse", fitted.inverse_transform(t), float64=True)
    return out


if __name__ == "__main__":
    run({"fit": fit, "transform": transform, "params": params_of(StandardScaler)})
