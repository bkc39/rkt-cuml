"""Twin of cuml/tests/mbsgd-twin-*.rkt: cuML-Python's MBSGDRegressor,
MBSGDClassifier and fit_sgd on fresh copies of the rows as sent, and
sklearn's optima of the same objective."""

import numpy as np
from cuml.linear_model import MBSGDClassifier, MBSGDRegressor
from cuml.solvers.sgd import fit_sgd
from sklearn import linear_model as sk
from twin_common import from_bits, host, put, run

SETTINGS = (
    "penalty",
    "alpha",
    "l1_ratio",
    "fit_intercept",
    "epochs",
    "tol",
    "shuffle",
    "learning_rate",
    "eta0",
    "power_t",
    "batch_size",
    "n_iter_no_change",
)


def rows(case):
    dt = case["dtype"]
    X = from_bits(case["X"], dt).reshape(case["n"], -1)
    y = None if case["y"] is None else from_bits(case["y"], dt)
    return X, y


def codes(case, dtype):
    return np.asarray(case["codes"], dtype=dtype)


def copied(a):
    return np.array(a, order="F", copy=True)


def params(case):
    kw = {k: case[k] for k in SETTINGS}
    kw["penalty"] = None if kw["penalty"] == "none" else kw["penalty"]
    return kw


def fitted(coef, intercept, dtype):
    out = {}
    put(out, "coef", np.asarray(host(coef), dtype=dtype))
    put(out, "intercept", np.asarray([intercept], dtype=dtype))
    return out


def cuml_reg(case):
    X, y = rows(case)
    m = MBSGDRegressor(output_type="numpy", **params(case)).fit(copied(X), copied(y))
    return fitted(m.coef_, m.intercept_, X.dtype)


def cuml_clf(case):
    X, _ = rows(case)
    m = MBSGDClassifier(output_type="numpy", loss=case["loss"], **params(case))
    m.fit(copied(X), copied(codes(case, np.int32)))
    out = fitted(m.coef_, m.intercept_, X.dtype)
    out["predict"] = [int(v) for v in np.asarray(host(m.predict(copied(X)))).ravel()]
    return out


def fit_sgd_op(case):
    X, _ = rows(case)
    c = codes(case, X.dtype)
    s = 2 * c - 1 if case["loss"] == "hinge" else c
    if case["fit_intercept"]:
        X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    kw = params(case)
    kw["fit_intercept"] = False
    coef, intercept = fit_sgd(
        MBSGDRegressor(output_type="numpy"), copied(X), copied(s), loss=case["loss"], **kw
    )
    return fitted(coef, intercept, X.dtype)


def permutation(seed, n):
    raw = np.random.RandomState(seed)._bit_generator.random_raw(max(n - 1, 0))
    order = np.arange(n)
    for k, i in enumerate(range(n - 1, 0, -1)):
        j = int(raw[k]) % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def permute(case):
    X, y = rows(case)
    order = permutation(case["seed"], len(y))
    m = MBSGDRegressor(output_type="numpy", **params(case))
    m.fit(copied(X[order]), copied(y[order]))
    return fitted(m.coef_, m.intercept_, X.dtype)


def objective(X, y, w, b, case):
    z = X @ w + b
    loss = case["loss"]
    if loss == "squared_loss":
        value = np.mean((y - z) ** 2)
    else:
        value = np.mean(np.logaddexp(0.0, z) - y * z)
    a, r = case["alpha"], case["l1_ratio"]
    l1, l2 = np.abs(w).sum(), w @ w
    penalty = {
        "none": 0.0,
        "l1": a * l1,
        "l2": a * l2,
        "elasticnet": a * ((1 - r) * l2 + r * l1),
    }[case["penalty"]]
    return float(value + penalty)


def sklearn_model(case, n):
    a, r, fi = case["alpha"], case["l1_ratio"], case["fit_intercept"]
    tight = {"tol": 1e-14, "max_iter": 10**6}
    if case["loss"] == "log":
        return sk.LogisticRegression(C=1.0 / (2 * a * n), fit_intercept=fi, **tight)
    if case["penalty"] == "l2":
        return sk.Ridge(alpha=n * a, fit_intercept=fi, tol=1e-14)
    if case["penalty"] == "l1":
        return sk.Lasso(alpha=a / 2, fit_intercept=fi, **tight)
    if case["penalty"] == "elasticnet":
        big = a * r / 2 + a * (1 - r)
        return sk.ElasticNet(alpha=big, l1_ratio=(a * r / 2) / big, fit_intercept=fi, **tight)
    return sk.LinearRegression(fit_intercept=fi)


def sklearn_opt(case):
    X, y = rows(case)
    X = X.astype(np.float64)
    y = codes(case, np.float64) if y is None else y.astype(np.float64)
    m = sklearn_model(case, len(y)).fit(X, y)
    w = np.asarray(m.coef_, dtype=np.float64).ravel()
    b = float(np.asarray(m.intercept_, dtype=np.float64).ravel()[0]) if case["fit_intercept"] else 0.0
    return {"coef": w.tolist(), "intercept": b, "objective": objective(X, y, w, b, case)}


if __name__ == "__main__":
    run(
        {
            "cuml-reg": cuml_reg,
            "cuml-clf": cuml_clf,
            "fit-sgd": fit_sgd_op,
            "permute": permute,
            "sklearn-opt": sklearn_opt,
        }
    )
