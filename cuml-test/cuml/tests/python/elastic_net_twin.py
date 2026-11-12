"""Twin of cuml/tests/elastic-net-twin-*.rkt: cuML-Python's ElasticNet (cd
solver) on the rows as sent, and sklearn's ElasticNet on their float64 copy."""

import warnings

import numpy as np
from cuml.linear_model import ElasticNet as CuElasticNet
from sklearn.linear_model import ElasticNet as SkElasticNet
from twin_common import fitted, inputs, run


def cuml_fit(case):
    X, y, w = inputs(case)
    m = CuElasticNet(
        alpha=case["alpha"],
        l1_ratio=case["l1_ratio"],
        fit_intercept=case["fit_intercept"],
        max_iter=case["max_iter"],
        tol=case["tol"],
        selection=case["selection"],
        solver="cd",
        output_type="numpy",
    ).fit(X, y, sample_weight=w)
    out = fitted(m, X.dtype)
    out["n_iter"] = int(m.n_iter_)
    return out


def sklearn_fit(case):
    X, y, w = inputs(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SkElasticNet(
            alpha=case["alpha"],
            l1_ratio=case["l1_ratio"],
            fit_intercept=case["fit_intercept"],
            max_iter=100000,
            tol=1e-12,
        ).fit(
            X.astype(np.float64),
            y.astype(np.float64),
            sample_weight=None if w is None else w.astype(np.float64),
        )
    return {
        "coef": [float(c) for c in m.coef_],
        "intercept": float(m.intercept_),
        "n_iter": int(m.n_iter_),
        "dual_gap": float(m.dual_gap_),
    }


if __name__ == "__main__":
    run({"cuml": cuml_fit, "sklearn": sklearn_fit})
