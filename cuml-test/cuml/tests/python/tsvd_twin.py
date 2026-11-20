"""Twin of cuml/tests/tsvd-twin-test.rkt and tsvd-cuml-twin-test.rkt:
cuML-Python's TruncatedSVD fits the rows the Racket side used in their own
dtype, sklearn's arpack TruncatedSVD fits them too, and numpy's SVD of their
float64 copy is the exact reference."""

import numpy as np
from cuml.decomposition import TruncatedSVD as CuTSVD
from sklearn.decomposition import TruncatedSVD as SkTSVD
from sklearn.utils.extmath import svd_flip
from twin_common import params_of, put, run


def rows(fixtures, case):
    X = np.asarray(fixtures[case["fixture"]], dtype=np.float64)
    return X.reshape(-1, case["d"]).astype(case["dtype"])


def put_fitted(out, model):
    put(out, "components", model.components_, float64=True)
    put(out, "singular_values", model.singular_values_, float64=True)
    put(out, "explained_variance", model.explained_variance_, float64=True)
    put(out, "explained_variance_ratio", model.explained_variance_ratio_, float64=True)
    return out


def cuml_model(case):
    return CuTSVD(
        n_components=case["k"],
        algorithm=case["solver"],
        n_iter=15,
        tol=0.0,
        output_type="numpy",
    )


def transform(fixtures, case):
    X = np.asfortranarray(rows(fixtures, case))
    model = cuml_model(case).fit(X)
    out = put_fitted({}, model)
    T = model.transform(X)
    put(out, "transform", T, float64=True)
    put(out, "inverse", model.inverse_transform(np.asfortranarray(T)), float64=True)
    return out


def sklearn_fit(fixtures, case):
    X = rows(fixtures, case)
    model = SkTSVD(n_components=case["k"], algorithm="arpack", random_state=0)
    return put_fitted({}, model.fit(X))


def exact(fixtures, case):
    X = rows(fixtures, case).astype(np.float64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    _, Vt = svd_flip(U, Vt, u_based_decision=False)
    return {"singular_values": s.tolist(), "components": Vt.ravel().tolist()}


if __name__ == "__main__":
    run(
        {
            "transform": transform,
            "sklearn": sklearn_fit,
            "exact": exact,
            "params": params_of(CuTSVD),
        },
        fixtures=True,
    )
