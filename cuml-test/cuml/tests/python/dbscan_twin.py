"""Twin of cuml/tests/dbscan-twin-test.rkt: cuml.DBSCAN fits the data and
parameters the Racket side used; sklearn.cluster.DBSCAN fits the cosine
cases, since cuml 26.08 normalises cosine input by column."""

import json
import sys

import numpy as np
from sklearn.cluster import DBSCAN

import cuml


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def fit(case):
    dtype = np.dtype(case["dtype"])
    X = np.asarray(case["X"], dtype=dtype)
    w = case.get("sample_weight")
    db = cuml.DBSCAN(
        eps=case["eps"],
        min_samples=case["min_samples"],
        metric=case["metric"],
        max_mbytes_per_batch=case["max_mbytes_per_batch"],
        output_type="numpy",
    ).fit(
        X,
        out_dtype=case["out_dtype"],
        sample_weight=None if w is None else np.asarray(w, dtype=dtype),
    )
    labels = host(db.labels_)
    core = host(db.core_sample_indices_)
    return {
        "labels": labels.astype(np.int64).tolist(),
        "core": core.astype(np.int64).tolist(),
        "dtypes": [str(labels.dtype), str(core.dtype)],
    }


def sklearn_fit(case):
    if case["max_mbytes_per_batch"] is not None:
        raise ValueError("sklearn_fit: max_mbytes_per_batch must be null")
    if case["out_dtype"] != "int32":
        raise ValueError("sklearn_fit: out_dtype must be int32")
    dtype = np.dtype(case["dtype"])
    X = np.asarray(case["X"], dtype=dtype)
    w = case["sample_weight"]
    db = DBSCAN(
        eps=case["eps"], min_samples=case["min_samples"], metric=case["metric"]
    ).fit(X, sample_weight=None if w is None else np.asarray(w, dtype=dtype))
    labels = db.labels_
    return {
        "labels": labels.astype(np.int64).tolist(),
        "core": db.core_sample_indices_.astype(np.int64).tolist(),
        "dtypes": [str(labels.dtype), str(db.core_sample_indices_.dtype)],
    }


def params(case):
    return cuml.DBSCAN().get_params()


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {"fit": fit, "sklearn_fit": sklearn_fit, "params": params}
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": cuml.__version__}, f)


if __name__ == "__main__":
    main()
