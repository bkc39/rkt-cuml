"""Twin of cuml/tests/linear-regression-twin-test.rkt: cuML-Python's
LinearRegression on the raw rows and on the rows staged as the shim stages
them, sklearn's LinearRegression on their float64 copy, and a CuPy product
for predictions."""

import cupy
import numpy as np
from cuml.linear_model import LinearRegression as CuLinearRegression
from sklearn.linear_model import LinearRegression as SkLinearRegression
from twin_common import fitted, from_bits, host, inputs, put, run


def unit_exponent(m):
    return 0 if m == 0 else -(int(np.frexp(m)[1]) - 1)


def even_floor(x):
    return x - (x % 2)


def spread_exponent(column, weights, fi, dt):
    largest = unit_exponent(float(np.max(np.abs(column))))
    v = np.ldexp(column.astype(np.float64), largest)
    if fi:
        v = v - (weights * v).sum() / weights.sum()
    spread = float(np.sqrt((weights * v * v).sum()))
    reach = np.finfo(dt).maxexp - 1 - len(column).bit_length()
    e = largest + unit_exponent(spread) if spread > 0 else largest
    return min(e, largest + reach)


def stage(X, y, w, fi):
    counted = np.ones(len(y), dtype=bool) if w is None else w > 0
    g = 0 if w is None else even_floor(unit_exponent(float(np.max(w))))
    weights = (
        np.ones(int(counted.sum()))
        if w is None
        else (w[counted] * np.ldexp(X.dtype.type(1), g)).astype(np.float64)
    )
    e = np.array(
        [
            spread_exponent(X[counted, j], weights, fi, X.dtype)
            for j in range(X.shape[1])
        ]
    )
    f = unit_exponent(float(np.max(np.abs(y[counted]))))
    first = int(np.argmax(counted))
    reference = X[first] if fi else np.zeros(X.shape[1], dtype=X.dtype)
    kept = np.array([np.any(X[counted, j] != reference[j]) for j in range(X.shape[1])])
    wide = kept.sum() > counted.sum() - (1 if fi else 0)
    if wide and kept.any():
        e = np.full(X.shape[1], int(np.min(e[kept])))
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
    Xs = np.ldexp(X[:, kept], e[kept]).astype(dt) * zero[:, None]
    if 2 <= counted.sum() < len(y):
        Xs, ys, ws = Xs[counted], ys[counted], ws[counted]
    m = CuLinearRegression(
        algorithm="svd" if wide else case["algorithm"],
        fit_intercept=fi,
        output_type="numpy",
    ).fit(np.asfortranarray(Xs), ys, sample_weight=ws)
    coef[kept] = np.ldexp(host(m.coef_).astype(dt), e[kept] - f).astype(dt)
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
    return fitted(m, X.dtype)


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


if __name__ == "__main__":
    run(
        {
            "cuml": raw_fit,
            "cuml-staged": staged_fit,
            "sklearn": sklearn_fit,
            "predict": predict,
            "params": params,
        }
    )
