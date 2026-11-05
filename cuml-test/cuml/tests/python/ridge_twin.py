"""Twin of cuml/tests/ridge-twin-*.rkt: cuML-Python's Ridge on the rows as
sent (raw, or already staged by the test), and sklearn's Ridge on their
float64 copy."""

import numpy as np
from cuml.linear_model import Ridge as CuRidge
from sklearn.linear_model import Ridge as SkRidge
from twin_common import fitted, inputs, run


def cuml_fit(case):
    X, y, w = inputs(case)
    m = CuRidge(
        alpha=case["alpha"],
        solver=case["solver"],
        fit_intercept=case["fit_intercept"],
        output_type="numpy",
    ).fit(X, y, sample_weight=w)
    return fitted(m, X.dtype)


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


if __name__ == "__main__":
    run({"cuml": cuml_fit, "sklearn": sklearn_fit})
