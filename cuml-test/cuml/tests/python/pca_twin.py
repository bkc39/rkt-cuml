"""Twin of cuml/tests/pca-twin-test.rkt and pca-cuml-twin-test.rkt: sklearn's
PCA (svd_solver='full') fits float64 copies of the rows the Racket side used,
and cuML-Python's PCA fits them in their own dtype."""

import json
import sys

import numpy as np
import sklearn
from cuml.decomposition import PCA as CuPCA
from sklearn.decomposition import PCA as SkPCA

import cuml


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def flat(a):
    return host(a).astype(np.float64).ravel().tolist()


def rows(fixtures, case):
    return np.asarray(fixtures[case["fixture"]], dtype=np.float64).reshape(
        -1, case["d"]
    )


def gaps(variances, k):
    return [
        float(np.min(np.abs(np.delete(variances, i) - variances[i])))
        if len(variances) > 1
        else float("inf")
        for i in range(k)
    ]


def fitted_keys(model, X, Y):
    t = model.transform(Y)
    return {
        "components": flat(model.components_),
        "explained_variance": flat(model.explained_variance_),
        "explained_variance_ratio": flat(model.explained_variance_ratio_),
        "singular_values": flat(model.singular_values_),
        "mean": flat(model.mean_),
        "noise_variance": float(host(model.noise_variance_)),
        "n_samples": int(X.shape[0]),
        "transform": flat(t),
        "inverse": flat(model.inverse_transform(t)),
    }


def sklearn_fit(fixtures, case):
    X = rows(fixtures, case)
    model = SkPCA(svd_solver="full", n_components=case["k"], whiten=case["whiten"])
    out = fitted_keys(model.fit(X), X, X)
    spectrum = SkPCA(svd_solver="full").fit(X).explained_variance_
    centred = X - X.mean(axis=0)
    out["gaps"] = gaps(spectrum, case["k"])
    out["total"] = float(np.sum(spectrum))
    out["std"] = X.std(axis=0, ddof=1).tolist()
    out["row_norms"] = np.linalg.norm(centred, axis=1).tolist()
    return out


def cuml_fit(fixtures, case):
    X = np.asfortranarray(rows(fixtures, case).astype(case["dtype"]))
    jacobi = case["solver"] == "jacobi"
    model = CuPCA(
        svd_solver=case["solver"],
        tol=0.0 if jacobi else 1e-7,
        iterated_power=15,
        n_components=case["k"],
        whiten=case["whiten"],
        output_type="numpy",
    )
    out = fitted_keys(model.fit(X), X, X)
    out["transform_rows"] = flat(model.transform(X[: case["rows"]]))
    return out


def params(fixtures, case):
    return CuPCA().get_params()


def main():
    with open(sys.argv[1]) as f:
        given = json.load(f)
    ops = {"sklearn": sklearn_fit, "cuml": cuml_fit, "params": params}
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
