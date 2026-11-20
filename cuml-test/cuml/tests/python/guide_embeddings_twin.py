"""Twin of cuml/tests/guide-embeddings-twin-test.rkt: Python cuML's HDBSCAN,
trustworthiness, UMAP and TSNE, and sklearn's ARI, on the float32 rows and
the fitted parameters of the embeddings chapter."""

import numpy as np
import tsne_twin
import umap_twin
from twin_common import host, run


def f32(rows):
    return np.ascontiguousarray(rows, dtype=np.float32)


def hdbscan(case):
    from cuml.cluster import HDBSCAN

    found = HDBSCAN(
        min_cluster_size=case["min_cluster_size"],
        min_samples=case["min_samples"],
        output_type="numpy",
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
    return {"embedding": host(m.embedding_).astype(np.float64).tolist()}


if __name__ == "__main__":
    run(
        {
            "digits": umap_twin.digits,
            "hdbscan": hdbscan,
            "trust": trust,
            "ari": ari,
            "umap": umap,
            "tsne": tsne_twin.tsne,
        }
    )
