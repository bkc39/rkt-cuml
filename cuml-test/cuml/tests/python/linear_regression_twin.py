"""Twin of cuml/tests/linear-regression-twin-test.rkt: cuML-Python's
LinearRegression on the raw rows and on the rows staged as the shim stages
them, sklearn's LinearRegression on their float64 copy, and a CuPy product
for predictions."""

import json
import sys

import cupy
import numpy as np
import sklearn
from cuml.linear_model import LinearRegression as CuLinearRegression
from sklearn.linear_model import LinearRegression as SkLinearRegression

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


def unit_exponent(m):
    return 0 if m == 0 else -(int(np.frexp(m)[1]) - 1)


def even_floor(x):
    return x - (x % 2)


def stage(X, y, w, fi):
    counted = np.ones(len(y), dtype=bool) if w is None else w > 0
    e = np.array(
        [unit_exponent(float(np.max(np.abs(X[counted, j])))) for j in range(X.shape[1])]
    )
    f = unit_exponent(float(np.max(np.abs(y[counted]))))
    g = 0 if w is None else even_floor(unit_exponent(float(np.max(w))))
    first = int(np.argmax(counted))
    reference = X[first] if fi else np.zeros(X.shape[1], dtype=X.dtype)
    kept = np.array([np.any(X[counted, j] != reference[j]) for j in range(X.shape[1])])
    wide = kept.sum() > counted.sum() - (1 if fi else 0)
    if wide:
        e = np.full(
            X.shape[1], unit_exponent(float(np.max(np.abs(X[counted][:, kept]))))
        )
    return e, f, g, kept, counted, wide


def staged_fit(case):
    X, y, w = inputs(case)
    fi = case["fit_intercept"]
    dt = X.dtype.type
    e, f, g, kept, counted, wide = stage(X, y, w, fi)
    coef = np.zeros(X.shape[1], dtype=dt)
    zero = np.where(counted, 1, 0).astype(dt)
    y0 = y[int(np.argmax(counted))]
    constant_y = np.all(y[counted] == y0) if fi else np.all(y[counted] == 0)
    ys = (y * np.ldexp(dt(1), f)).astype(dt) * zero
    ws = None if w is None else (w * np.ldexp(dt(1), g)).astype(dt)
    out = {}
    if constant_y or not kept.any():
        if not fi:
            intercept = dt(0)
        elif constant_y:
            intercept = y0
        else:
            weights = np.ones(len(y)) if ws is None else ws.astype(np.float64)
            mean = dt((weights * ys.astype(np.float64)).sum() / weights.sum())
            intercept = dt(np.ldexp(mean, -f))
        put(out, "coef", coef)
        put(out, "intercept", np.asarray([intercept], dtype=dt))
        return out
    Xs = (X[:, kept] * np.ldexp(dt(1), e[kept]).astype(dt)).astype(dt) * zero[:, None]
    if 2 <= counted.sum() < len(y):
        Xs, ys, ws = Xs[counted], ys[counted], ws[counted]
    m = CuLinearRegression(
        algorithm="svd" if wide else case["algorithm"],
        fit_intercept=fi,
        output_type="numpy",
    ).fit(np.asfortranarray(Xs), ys, sample_weight=ws)
    coef[kept] = (
        host(m.coef_).astype(dt) * np.ldexp(dt(1), e[kept] - f).astype(dt)
    ).astype(dt)
    intercept = dt(np.ldexp(dt(m.intercept_), -f)) if fi else dt(0)
    put(out, "coef", coef)
    put(out, "intercept", np.asarray([intercept], dtype=dt))
    return out


def raw_fit(case):
    X, y, w = inputs(case)
    m = CuLinearRegression(
        algorithm=case["algorithm"],
        fit_intercept=case["fit_intercept"],
        output_type="numpy",
    ).fit(X, y, sample_weight=w)
    out = {}
    put(out, "coef", np.asarray(host(m.coef_), dtype=X.dtype))
    put(out, "intercept", np.asarray([m.intercept_], dtype=X.dtype))
    return out


def sklearn_fit(case):
    X, y, w = inputs(case)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    weights = np.ones(len(y)) if w is None else w.astype(np.float64)
    m = SkLinearRegression(fit_intercept=case["fit_intercept"]).fit(
        X, y, sample_weight=weights
    )
    centred = X - (
        np.average(X, axis=0, weights=weights) if case["fit_intercept"] else 0
    )
    centred = centred[:, np.any(centred != 0, axis=0)]
    s = np.linalg.svd(np.sqrt(weights)[:, None] * centred, compute_uv=False)
    out = {}
    put(out, "coef", m.coef_)
    put(out, "intercept", np.asarray([m.intercept_], dtype=np.float64))
    out["kappa"] = float(s[0] / s[-1])
    return out


def predict(case):
    X, _, _ = inputs(case)
    coef = from_bits(case["coef"], case["dtype"])
    intercept = X.dtype.type(case["intercept"])
    out = {}
    put(out, "pred", cupy.asarray(X) @ cupy.asarray(coef) + intercept)
    return out


def params(case):
    return CuLinearRegression().get_params()


def main():
    with open(sys.argv[1]) as f:
        given = json.load(f)
    ops = {
        "cuml": raw_fit,
        "cuml-staged": staged_fit,
        "sklearn": sklearn_fit,
        "predict": predict,
        "params": params,
    }
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
