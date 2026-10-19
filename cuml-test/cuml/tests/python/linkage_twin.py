"""Twin of cuml/tests/linkage-twin-test.rkt: cuml.AgglomerativeClustering
fits the data and parameters the Racket side used; scipy's single linkage
gives the cophenetic matrix the Racket children must induce; sklearn's
AgglomerativeClustering gives the partitions."""

import json
import sys

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative

import cuml


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def samples(case):
    return np.asarray(case["X"], dtype=np.float32)


def fit(case):
    model = cuml.AgglomerativeClustering(
        n_clusters=case["n_clusters"],
        metric="euclidean",
        connectivity=case["connectivity"],
        c=case["c"],
        output_type="numpy",
    ).fit(samples(case))
    labels = host(model.labels_)
    children = host(model.children_)
    return {
        "labels": labels.astype(np.int64).tolist(),
        "children": np.sort(children, axis=1).astype(np.int64).tolist(),
        "dtype": str(labels.dtype),
    }


def cophenetic_of(children, distances):
    n = len(distances)
    members = [[i] for i in range(n)]
    found = np.zeros((n, n))
    for a, b in children:
        left, right = members[a], members[b]
        height = distances[np.ix_(left, right)].min()
        found[np.ix_(left, right)] = height
        found[np.ix_(right, left)] = height
        members.append(left + right)
    return found


def scipy_fit(case):
    X = samples(case).astype(np.float64)
    Z = linkage(X, method="single")
    expected = squareform(cophenet(Z))
    found = cophenetic_of(case["children"], squareform(pdist(X)))
    largest = float(expected.max()) if expected.size else 0.0
    return {
        "worst": float(np.abs(found - expected).max()) if expected.size else 0.0,
        "largest": largest,
        "pairs": np.sort(Z[:, :2].astype(np.int64), axis=1).tolist(),
    }


def sklearn_fit(case):
    model = SklearnAgglomerative(n_clusters=case["n_clusters"], linkage="single").fit(
        samples(case)
    )
    return {"labels": model.labels_.astype(np.int64).tolist()}


def params(case):
    return cuml.AgglomerativeClustering().get_params()


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {"fit": fit, "scipy": scipy_fit, "sklearn": sklearn_fit, "params": params}
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": cuml.__version__}, f)


if __name__ == "__main__":
    main()
