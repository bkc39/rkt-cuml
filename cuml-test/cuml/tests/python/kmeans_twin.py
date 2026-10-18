"""Twin of cuml/tests/kmeans-twin-test.rkt: cuml.KMeans fits and applies the
data and parameters the Racket side used, and scikit-learn scores the labels."""

import json
import sys

import cuml
import numpy as np
from sklearn.metrics import adjusted_rand_score


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def weights(case, dtype):
    w = case.get("sample_weight")
    return None if w is None else np.asarray(w, dtype=dtype)


def fit(case):
    dtype = np.dtype(case["dtype"])
    X = np.asarray(case["X"], dtype=dtype)
    init = case["init"]
    if isinstance(init, list):
        init = np.asarray(init, dtype=dtype)
    km = cuml.KMeans(
        n_clusters=case["n_clusters"],
        init=init,
        n_init=case["n_init"],
        max_iter=case["max_iter"],
        tol=case["tol"],
        random_state=case["random_state"],
        oversampling_factor=case["oversampling_factor"],
        max_samples_per_batch=case["max_samples_per_batch"],
        output_type="numpy",
    ).fit(X, sample_weight=weights(case, dtype))
    labels = host(km.labels_).astype(np.int64)
    return {
        "labels": labels.tolist(),
        "centroids": host(km.cluster_centers_).astype(np.float64).tolist(),
        "inertia": float(km.inertia_),
        "n_iter": int(km.n_iter_),
        "ari": float(adjusted_rand_score(case["labels"], labels)),
    }


def apply(case):
    dtype = np.dtype(case["dtype"])
    Y = np.asarray(case["Y"], dtype=dtype)
    C = np.asarray(case["centroids"], dtype=dtype)
    km = cuml.KMeans(
        n_clusters=C.shape[0], init=C, n_init=1, max_iter=0, output_type="numpy"
    ).fit(Y)
    return {
        "labels": host(km.predict(Y)).astype(np.int64).tolist(),
        "transform": host(km.transform(Y)).astype(np.float64).tolist(),
        "score": float(km.score(Y)),
        "weighted_score": float(km.score(Y, sample_weight=weights(case, dtype))),
    }


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {"fit": fit, "apply": apply}
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": cuml.__version__}, f)


if __name__ == "__main__":
    main()
