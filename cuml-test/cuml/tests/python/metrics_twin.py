import json
import math
import sys

import numpy as np
from cuml.metrics import accuracy_score as cu_acc
from cuml.metrics import adjusted_rand_score as cu_ari
from cuml.metrics import mean_squared_error as cu_mse
from cuml.metrics import r2_score as cu_r2
from cuml.metrics import trustworthiness as cu_trustworthiness
from cuml.metrics.cluster import silhouette_samples as cu_samples
from cuml.metrics.cluster import silhouette_score as cu_score
from sklearn.manifold import trustworthiness as sk_trustworthiness
from sklearn.metrics import accuracy_score as sk_acc
from sklearn.metrics import adjusted_rand_score as sk_ari
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2
from sklearn.metrics import silhouette_samples as sk_samples
from sklearn.metrics import silhouette_score as sk_score
from sklearn.metrics.cluster import contingency_matrix as sk_contingency

import cuml


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def contingency(case):
    counts = sk_contingency(np.asarray(case["truth"]), np.asarray(case["found"]))
    return {"counts": counts.astype(np.int64).tolist(), "dtype": str(counts.dtype)}


def ari(case):
    truth = np.asarray(case["truth"], dtype=np.int32)
    found = np.asarray(case["found"], dtype=np.int32)
    theirs = float(cu_ari(truth, found)) if case["cuml"] else None
    return {"sklearn": float(sk_ari(truth, found)), "cuml": theirs}


def silhouette(case):
    X = np.asarray(case["X"], dtype=case["dtype"])
    labels = np.asarray(case["labels"], dtype=np.int32)
    metric = case["metric"]
    cast = X.astype(np.float64)
    result = {
        "score": float(sk_score(cast, labels, metric=metric)),
        "samples": sk_samples(cast, labels, metric=metric).tolist(),
        "cuml_score": None,
        "cuml_samples": None,
    }
    if case["cuml"]:
        result["cuml_score"] = float(cu_score(X, labels, metric=metric))
        result["cuml_samples"] = (
            host(cu_samples(X, labels, metric=metric)).astype(np.float64).tolist()
        )
    return result


def trustworthiness(case):
    X = np.asarray(case["X"], dtype=np.float32)
    E = np.asarray(case["E"], dtype=np.float32)
    k = case["n_neighbors"]
    return {
        "sklearn": float(
            sk_trustworthiness(
                X.astype(np.float64), E.astype(np.float64), n_neighbors=k
            )
        ),
        "cuml": float(cu_trustworthiness(X, E, n_neighbors=k)),
    }


def finite(x):
    x = float(x)
    return None if math.isnan(x) else x


def weight(case):
    w = case["weight"]
    return None if w is None else np.asarray(w, dtype=np.float64)


def regression(case):
    yt = np.asarray(case["y_true"], dtype=case["dtype_true"])
    yp = np.asarray(case["y_pred"], dtype=case["dtype_pred"])
    w = weight(case)
    sk, cu = {"r2": (sk_r2, cu_r2), "mse": (sk_mse, cu_mse)}[case["op"]]
    return {
        "sklearn": finite(
            sk(yt.astype(np.float64), yp.astype(np.float64), sample_weight=w)
        ),
        "cuml": finite(cu(yt, yp, sample_weight=w)),
    }


def accuracy(case):
    a = np.asarray(case["y_true"], dtype=case["dtype_true"])
    b = np.asarray(case["y_pred"], dtype=case["dtype_pred"])
    w = weight(case)
    return {
        "sklearn": float(sk_acc(a, b, sample_weight=w)),
        "cuml": float(cu_acc(a, b, sample_weight=w)),
    }


def version(case):
    return cuml.__version__


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {
        "contingency": contingency,
        "ari": ari,
        "silhouette": silhouette,
        "trustworthiness": trustworthiness,
        "r2": regression,
        "mse": regression,
        "accuracy": accuracy,
        "version": version,
    }
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": cuml.__version__}, f)


if __name__ == "__main__":
    main()
