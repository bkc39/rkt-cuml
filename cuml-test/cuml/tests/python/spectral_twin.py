"""Twin of cuml/tests/spectral-twin-test.rkt: cuml's SpectralClustering fits
the data, graphs and parameters the Racket side used; sklearn's
spectral_clustering partitions the graphs."""

import json
import sys

import numpy as np
import scipy.sparse as sp
from cuml.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering

import cuml


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def model(case, affinity):
    return SpectralClustering(
        n_clusters=case["n_clusters"],
        n_components=case["n_components"],
        random_state=case["random_state"],
        n_neighbors=case["n_neighbors"],
        n_init=case["n_init"],
        eigen_tol=case["eigen_tol"],
        affinity=affinity,
        output_type="numpy",
    )


def square(case):
    n = case["n"]
    return sp.coo_matrix(
        (np.asarray(case["vals"], dtype=np.float32), (case["rows"], case["cols"])),
        shape=(n, n),
    ).toarray()


def labels_of(found):
    found = host(found)
    return {"labels": found.astype(np.int64).tolist(), "dtype": str(found.dtype)}


def fit(case):
    X = np.asarray(case["X"], dtype=np.float32)
    return labels_of(model(case, "nearest_neighbors").fit_predict(X))


def precomputed(case):
    return labels_of(model(case, "precomputed").fit_predict(square(case)))


def sklearn_fit(case):
    A = square(case).astype(np.float64)
    found = spectral_clustering(A, n_clusters=case["n_clusters"], random_state=0)
    return {"labels": found.astype(np.int64).tolist()}


def params(case):
    return SpectralClustering().get_params()


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {
        "fit": fit,
        "precomputed": precomputed,
        "sklearn": sklearn_fit,
        "params": params,
    }
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": cuml.__version__}, f)


if __name__ == "__main__":
    main()
