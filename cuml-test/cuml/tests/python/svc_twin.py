"""Twin of cuml/tests/svc-twin-*.rkt: cuML-Python's SVC on the rows as sent,
and sklearn's on their float64 copy."""

import numpy as np
from cuml.svm import SVC as CuSVC
from sklearn.svm import SVC as SkSVC
from twin_common import from_bits, host, params_of, put, run


def inputs(case):
    dt = case["dtype"]
    n = len(case["y"])
    X = np.ascontiguousarray(from_bits(case["X"], dt).reshape(n, -1))
    y = np.asarray(case["y"], dtype=np.int32)
    w = None if case["w"] is None else from_bits(case["w"], dt)
    return X, y, w


def settings(case):
    return {
        "kernel": case["kernel"],
        "degree": case["degree"],
        "gamma": case["gamma"],
        "coef0": case["coef0"],
        "C": case["C"],
        "tol": case["tol"],
        "cache_size": case["cache_size"],
    }


def cuml_fit(case):
    X, y, w = inputs(case)
    m = CuSVC(**settings(case), output_type="numpy").fit(X, y, sample_weight=w)
    out = {
        "gamma": float(m._gamma),
        "support": [int(i) for i in np.asarray(host(m.support_)).ravel()],
        "n_iter": int(np.asarray(host(m.n_iter_)).ravel()[0]),
        "predict": [int(v) for v in np.asarray(host(m.predict(X))).ravel()],
    }
    put(out, "gamma64", np.asarray([m._gamma], dtype=np.float64))
    put(out, "support_vectors", np.asarray(host(m.support_vectors_), dtype=X.dtype))
    put(out, "dual_coef", np.asarray(host(m.dual_coef_), dtype=X.dtype))
    put(out, "intercept", np.asarray(host(m.intercept_), dtype=X.dtype))
    put(out, "decision", np.asarray(host(m.decision_function(X)), dtype=X.dtype))
    return out


def sklearn_fit(case):
    X, y, _ = inputs(case)
    m = SkSVC(**settings(case)).fit(X.astype(np.float64), y)
    return {
        "decision": [float(v) for v in m.decision_function(X.astype(np.float64))],
        "predict": [int(v) for v in m.predict(X.astype(np.float64))],
        "support": [int(i) for i in m.support_],
    }


if __name__ == "__main__":
    run({"cuml-fit": cuml_fit, "sklearn-fit": sklearn_fit, "params": params_of(CuSVC)})
