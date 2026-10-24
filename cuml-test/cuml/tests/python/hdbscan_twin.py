"""Twin of cuml/tests/hdbscan-twin-test.rkt: cuml's HDBSCAN fits the data and
settings the Racket side used, with its trees, predictions and memberships;
sklearn's HDBSCAN at min_samples + 1 labels the same data."""

import json
import sys

import numpy as np
from cuml.cluster import HDBSCAN
from cuml.cluster.hdbscan import (
    all_points_membership_vectors,
    approximate_predict,
    membership_vector,
)
from sklearn.cluster import HDBSCAN as SkHDBSCAN

import cuml


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def floats(a):
    return host(a).astype(np.float32).astype(np.float64).tolist()


def ints(a):
    return host(a).astype(np.int64).tolist()


def settings(case):
    return {
        "min_cluster_size": case["min_cluster_size"],
        "min_samples": case["min_samples"],
        "cluster_selection_epsilon": case["cluster_selection_epsilon"],
        "max_cluster_size": case["max_cluster_size"],
        "alpha": case["alpha"],
        "cluster_selection_method": case["cluster_selection_method"],
        "allow_single_cluster": case["allow_single_cluster"],
    }


def fit(case):
    X = np.asarray(case["X"], dtype=np.float32)
    points = np.asarray(case["points"], dtype=np.float32)
    model = HDBSCAN(
        **settings(case),
        gen_min_span_tree=True,
        prediction_data=True,
        output_type="numpy",
    ).fit(X)
    tree = model._condensed_tree
    linkage = np.asarray(model._single_linkage_tree)
    mst = np.asarray(model._min_spanning_tree)
    labels, probabilities = approximate_predict(model, points)
    pairs = np.sort(linkage[:, :2].astype(np.int64), axis=1)
    return {
        "labels": ints(model.labels_),
        "probabilities": floats(model.probabilities_),
        "persistence": floats(model.cluster_persistence_),
        "parents": ints(tree["parent"]),
        "children": ints(tree["child"]),
        "lambdas": floats(tree["lambda_val"]),
        "sizes": ints(tree["child_size"]),
        "linkage": pairs.tolist(),
        "distances": floats(linkage[:, 2]),
        "linkage_sizes": ints(linkage[:, 3]),
        "sources": ints(mst[:, 0]),
        "destinations": ints(mst[:, 1]),
        "weights": floats(mst[:, 2]),
        "predicted": [ints(labels), floats(probabilities)],
        "all_points": host(all_points_membership_vectors(model)).astype(np.float64).tolist(),
        "membership": host(membership_vector(model, points)).astype(np.float64).tolist(),
    }


def sklearn_fit(case):
    X = np.asarray(case["X"], dtype=np.float32).astype(np.float64)
    ms = case["min_samples"] or case["min_cluster_size"]
    found = SkHDBSCAN(
        min_cluster_size=case["min_cluster_size"],
        min_samples=ms + 1,
        cluster_selection_epsilon=case["cluster_selection_epsilon"],
        alpha=case["alpha"],
        allow_single_cluster=case["allow_single_cluster"],
    ).fit_predict(X)
    return {"labels": ints(found)}


def params(case):
    return HDBSCAN().get_params()


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {"fit": fit, "sklearn": sklearn_fit, "params": params}
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": cuml.__version__}, f)


if __name__ == "__main__":
    main()
