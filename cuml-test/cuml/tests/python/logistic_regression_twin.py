"""Twin of cuml/tests/logistic-regression-twin-*.rkt: cuML-Python's
LogisticRegression on the rows as sent, row-major, and
sklearn's on their float64 copy."""

import numpy as np
from cuml.linear_model import LogisticRegression as CuLR
from sklearn.linear_model import LogisticRegression as SkLR
from twin_common import from_bits, host, put, run

PENALTIES = {"none": None, "l1": "l1", "l2": "l2", "elasticnet": "elasticnet"}
RATIOS = {"none": 0.0, "l1": 1.0, "l2": 0.0}


def inputs(case):
    dt = case["dtype"]
    n = len(case["y"])
    X = np.ascontiguousarray(from_bits(case["X"], dt).reshape(n, -1))
    y = np.asarray(case["y"], dtype=np.int32)
    w = None if case["w"] is None else from_bits(case["w"], dt)
    return X, y, w


def cuml_model(case):
    X, y, w = inputs(case)
    kw = {
        "penalty": PENALTIES[case["penalty"]],
        "C": case["C"],
        "fit_intercept": case["fit_intercept"],
        "tol": case["tol"],
        "max_iter": case["max_iter"],
        "output_type": "numpy",
    }
    if case["penalty"] == "elasticnet":
        kw["l1_ratio"] = case["l1_ratio"]
    return CuLR(**kw).fit(X, y, sample_weight=w), X


def fitted(m, X):
    out = {"n_iter": int(np.asarray(host(m.n_iter_)).ravel()[0])}
    put(out, "coef", np.asarray(host(m.coef_), dtype=X.dtype))
    put(out, "intercept", np.asarray(host(m.intercept_), dtype=X.dtype))
    return out


def cuml_fit(case):
    m, X = cuml_model(case)
    return fitted(m, X)


def cuml_apply(case):
    m, X = cuml_model(case)
    out = fitted(m, X)
    put(out, "decision", np.asarray(host(m.decision_function(X)), dtype=X.dtype))
    out["predict"] = [int(v) for v in np.asarray(host(m.predict(X))).ravel()]
    proba = np.asarray(host(m.predict_proba(X)))
    out["proba_dtype"] = str(proba.dtype)
    put(out, "proba", proba)
    return out


def sklearn_fit(case):
    X, y, w = inputs(case)
    ratio = case["l1_ratio"] if case["penalty"] == "elasticnet" else RATIOS[case["penalty"]]
    m = SkLR(
        C=np.inf if case["penalty"] == "none" else case["C"],
        l1_ratio=ratio,
        fit_intercept=case["fit_intercept"],
        tol=case["tol"],
        max_iter=10**6,
        solver="lbfgs" if ratio == 0 else "saga",
    ).fit(
        X.astype(np.float64),
        y,
        sample_weight=None if w is None else w.astype(np.float64),
    )
    return {
        "coef": [[float(c) for c in row] for row in m.coef_],
        "intercept": [float(b) for b in m.intercept_],
        "predict": [int(v) for v in m.predict(X.astype(np.float64))],
    }


if __name__ == "__main__":
    run({"cuml-fit": cuml_fit, "cuml-apply": cuml_apply, "sklearn-fit": sklearn_fit})
