"""Twin of cuml/tests/tsne-*-twin-test.rkt: cuml's TSNE embeds the data the
Racket side used, cuml's trustworthiness and HDBSCAN score the embeddings,
and sklearn's exact TSNE is the reference for method exact."""

import json
import sys

import numpy as np


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def rows(case):
    return np.ascontiguousarray(case["X"], dtype=np.float32)


def model(case, seed, min_grad_norm):
    from cuml.manifold import TSNE

    return TSNE(
        n_components=2,
        perplexity=case["perplexity"],
        early_exaggeration=case["early_exaggeration"],
        learning_rate=case["learning_rate"],
        learning_rate_method="none",
        max_iter=case["max_iter"],
        n_neighbors=case["n_neighbors"],
        method=case["method"],
        metric=case["metric"],
        init=case["init"],
        random_state=seed,
        square_distances=True,
        min_grad_norm=min_grad_norm,
        output_type="numpy",
    )


def ours_rule(case):
    return 0.0 if case["method"] == "fft" else 1e-7


def fitted(case, seed, min_grad_norm):
    X = rows(case)
    m = model(case, seed, min_grad_norm)
    E = host(m.fit_transform(X)).astype(np.float32)
    return m, X, E


def tsne(case):
    m, _, E = fitted(case, case["seed"], ours_rule(case))
    return {
        "embedding": E.astype(np.float64).tolist(),
        "kl": float(m.kl_divergence_),
        "n_iter": int(m.n_iter_),
    }


def python_default(case):
    m, _, E = fitted(case, case["seed"], 1e-7)
    return {"n_iter": int(m.n_iter_), "std": float(np.std(E))}


def quality(case):
    from cuml.cluster import HDBSCAN
    from cuml.metrics import adjusted_rand_score, trustworthiness

    trust, ari = [], []
    truth = case.get("truth")
    for seed in case["seeds"]:
        _, X, E = fitted(case, seed, ours_rule(case))
        trust.append(float(trustworthiness(X, E, n_neighbors=10)))
        if truth is not None:
            found = HDBSCAN(min_cluster_size=20, output_type="numpy").fit_predict(E)
            ari.append(
                float(
                    adjusted_rand_score(
                        np.asarray(truth, dtype=np.int32), host(found).astype(np.int32)
                    )
                )
            )
    return {"trustworthiness": trust, "ari": ari}


def sklearn(case):
    from cuml.metrics import trustworthiness
    from sklearn.manifold import TSNE

    X = rows(case)
    E = TSNE(
        n_components=2,
        perplexity=case["perplexity"],
        method="exact",
        random_state=1,
        min_grad_norm=1e-12,
    ).fit_transform(X)
    return {"trustworthiness": float(trustworthiness(X, E.astype(np.float32), n_neighbors=10))}


def params(case):
    from cuml.manifold import TSNE

    return {
        k: v
        for k, v in TSNE().get_params().items()
        if v is None or isinstance(v, (bool, int, float, str))
    }


def versions():
    out = {}
    try:
        import sklearn

        import cuml

        out.update({"cuml": cuml.__version__, "sklearn": sklearn.__version__})
    except ImportError:
        pass
    return out


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    ops = {
        "tsne": tsne,
        "python-default": python_default,
        "quality": quality,
        "sklearn": sklearn,
        "params": params,
    }
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": versions()}, f)


if __name__ == "__main__":
    main()
