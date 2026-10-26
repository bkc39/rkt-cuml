"""Twin of cuml/tests/guide-twin-test.rkt: sklearn's exact PCA and arpack
TruncatedSVD of the float32 rows the decomposition chapter fits, in float64."""

import json
import sys

import numpy as np
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD


def rows(case):
    return np.float64(np.asarray(case["rows"], dtype=np.float32))


def pca(case):
    X = rows(case)
    model = PCA(svd_solver="full", n_components=case["k"]).fit(X)
    spectrum = PCA(svd_solver="full").fit(X).explained_variance_
    return {
        "ratio": model.explained_variance_ratio_.tolist(),
        "singular_values": model.singular_values_.tolist(),
        "components": model.components_.tolist(),
        "explained_variance": model.explained_variance_.tolist(),
        "gaps": (spectrum[:-1] - spectrum[1:]).tolist(),
    }


def tsvd(case):
    X = rows(case)
    model = TruncatedSVD(algorithm="arpack", tol=0.0, n_components=case["k"]).fit(X)
    return {
        "ratio": model.explained_variance_ratio_.tolist(),
        "singular_values": model.singular_values_.tolist(),
        "components": model.components_.tolist(),
    }


def main():
    with open(sys.argv[1]) as f:
        given = json.load(f)
    ops = {"pca": pca, "tsvd": tsvd}
    results = [ops[case["op"]](case) for case in given["cases"]]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": sklearn.__version__}, f)


if __name__ == "__main__":
    main()
