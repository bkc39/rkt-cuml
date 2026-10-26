"""Twin of cuml/tests/tsvd-twin-test.rkt and tsvd-cuml-twin-test.rkt:
cuML-Python's TruncatedSVD fits the rows the Racket side used in their own
dtype, sklearn's arpack TruncatedSVD fits them too, and numpy's SVD of their
float64 copy is the exact reference."""

import json
import sys

import numpy as np
import sklearn
from cuml.decomposition import TruncatedSVD as CuTSVD
from sklearn.decomposition import TruncatedSVD as SkTSVD
from sklearn.utils.extmath import svd_flip

import cuml

BITS = {np.dtype(np.float32): np.uint32, np.dtype(np.float64): np.uint64}


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def put(out, key, a):
    a = np.ascontiguousarray(host(a))
    out[key] = a.astype(np.float64).ravel().tolist()
    out[key + "_bits"] = a.view(BITS[a.dtype]).ravel().tolist()


def rows(fixtures, case):
    X = np.asarray(fixtures[case["fixture"]], dtype=np.float64)
    return X.reshape(-1, case["d"]).astype(case["dtype"])


def fitted(out, model):
    put(out, "components", model.components_)
    put(out, "singular_values", model.singular_values_)
    put(out, "explained_variance", model.explained_variance_)
    put(out, "explained_variance_ratio", model.explained_variance_ratio_)
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
    out = fitted({}, model)
    T = model.transform(X)
    put(out, "transform", T)
    put(out, "inverse", model.inverse_transform(np.asfortranarray(T)))
    return out


def sklearn_fit(fixtures, case):
    X = rows(fixtures, case)
    model = SkTSVD(n_components=case["k"], algorithm="arpack", random_state=0)
    return fitted({}, model.fit(X))


def exact(fixtures, case):
    X = rows(fixtures, case).astype(np.float64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    _, Vt = svd_flip(U, Vt, u_based_decision=False)
    return {"singular_values": s.tolist(), "components": Vt.ravel().tolist()}


def params(fixtures, case):
    return CuTSVD().get_params()


def main():
    with open(sys.argv[1]) as f:
        given = json.load(f)
    ops = {
        "transform": transform,
        "sklearn": sklearn_fit,
        "exact": exact,
        "params": params,
    }
    fixtures = given.get("fixtures", {})
    results = [ops[case["op"]](fixtures, case) for case in given["cases"]]
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
