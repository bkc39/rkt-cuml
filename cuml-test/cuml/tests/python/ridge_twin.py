"""Twin of cuml/tests/ridge-twin-*.rkt: cuML-Python's Ridge on the rows as
sent (raw, or already staged by the test), and sklearn's Ridge on their
float64 copy."""

import json
import sys

import numpy as np
import sklearn
from cuml.linear_model import Ridge as CuRidge
from sklearn.linear_model import Ridge as SkRidge

import cuml

BITS = {np.dtype(np.float32): np.uint32, np.dtype(np.float64): np.uint64}
WIDTH = {"float32": np.uint32, "float64": np.uint64}


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def put(out, key, a):
    a = np.ascontiguousarray(host(a))
    out[key + "_bits"] = a.view(BITS[a.dtype]).ravel().tolist()


def from_bits(bits, dtype):
    return np.asarray(bits, dtype=WIDTH[dtype]).view(np.dtype(dtype))


def inputs(case):
    dt = case["dtype"]
    n = len(case["y"])
    X = from_bits(case["X"], dt).reshape(n, -1)
    y = from_bits(case["y"], dt)
    w = None if case["w"] is None else from_bits(case["w"], dt)
    return X, y, w


def cuml_fit(case):
    X, y, w = inputs(case)
    m = CuRidge(
        alpha=case["alpha"],
        solver=case["solver"],
        fit_intercept=case["fit_intercept"],
        output_type="numpy",
    ).fit(X, y, sample_weight=w)
    out = {}
    put(out, "coef", np.asarray(host(m.coef_), dtype=X.dtype))
    put(out, "intercept", np.asarray([m.intercept_], dtype=X.dtype))
    return out


def sklearn_fit(case):
    X, y, w = inputs(case)
    m = SkRidge(
        alpha=case["alpha"],
        solver=case["solver"],
        fit_intercept=case["fit_intercept"],
    ).fit(
        X.astype(np.float64),
        y.astype(np.float64),
        sample_weight=None if w is None else w.astype(np.float64),
    )
    return {"coef": [float(c) for c in m.coef_], "intercept": float(m.intercept_)}


def main():
    with open(sys.argv[1]) as f:
        given = json.load(f)
    ops = {"cuml": cuml_fit, "sklearn": sklearn_fit}
    results = [ops[case["op"]](case) for case in given["cases"]]
    with open(sys.argv[2], "w") as f:
        json.dump(
            {
                "results": results,
                "version": {"sklearn": sklearn.__version__, "cuml": cuml.__version__},
            },
            f,
        )


if __name__ == "__main__":
    main()
