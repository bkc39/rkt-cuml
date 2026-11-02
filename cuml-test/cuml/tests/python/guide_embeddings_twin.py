"""Twin of cuml/tests/guide-embeddings-twin-test.rkt: Python cuML's HDBSCAN,
trustworthiness, UMAP and TSNE, and sklearn's ARI, on the float32 rows and
the fitted parameters of the embeddings chapter."""

import json
import sys

import numpy as np
import tsne_twin
import umap_twin


def f32(rows):
    return np.ascontiguousarray(rows, dtype=np.float32)


def hdbscan(case):
    from cuml.cluster import HDBSCAN

    found = HDBSCAN(
        min_cluster_size=60, min_samples=10, output_type="numpy"
    ).fit_predict(f32(case["X"]))
    return {"labels": np.asarray(found).astype(int).tolist()}


def trust(case):
    from cuml.metrics import trustworthiness

    return {"t": float(trustworthiness(f32(case["X"]), f32(case["E"]), n_neighbors=5))}


def ari(case):
    from sklearn.metrics import adjusted_rand_score

    return {"ari": float(adjusted_rand_score(case["y"], case["labels"]))}


def umap(case):
    m, _ = umap_twin.fitted(case, case["seed"])
    return {"embedding": umap_twin.host(m.embedding_).astype(np.float64).tolist()}


def tsne(case):
    _, _, E = tsne_twin.fitted(case, case["seed"], tsne_twin.ours_rule(case))
    return {"embedding": E.astype(np.float64).tolist()}


def versions():
    import sklearn

    import cuml

    return {"cuml": cuml.__version__, "sklearn": sklearn.__version__}


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {
        "digits": umap_twin.digits,
        "hdbscan": hdbscan,
        "trust": trust,
        "ari": ari,
        "umap": umap,
        "tsne": tsne,
    }
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": versions()}, f)


if __name__ == "__main__":
    main()
