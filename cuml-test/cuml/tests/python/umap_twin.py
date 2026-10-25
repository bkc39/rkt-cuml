"""Twin of cuml/tests/umap-*-twin-test.rkt: scipy fits the curve parameters,
cuml's UMAP embeds the data the Racket side used, and cuml's trustworthiness
and HDBSCAN score the embeddings."""

import json
import sys

import numpy as np


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def curve(x, a, b):
    return 1.0 / (1.0 + a * x ** (2 * b))


def targets(spread, min_dist, xv):
    return np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))


def ab(case):
    from scipy.optimize import curve_fit, least_squares

    spread, min_dist = case["spread"], case["min_dist"]
    xv = np.linspace(0, spread * 3, 300)
    yv = targets(spread, min_dist, xv)
    params, _ = curve_fit(curve, xv, yv)

    def residual(p):
        return curve(xv, p[0], p[1]) - yv

    def jacobian(p):
        a, b = p
        with np.errstate(divide="ignore", invalid="ignore"):
            x2b = np.where(xv > 0, xv ** (2 * b), 0.0)
            logx = np.where(xv > 0, np.log(np.where(xv > 0, xv, 1.0)), 0.0)
        d = (1.0 + a * x2b) ** 2
        return np.stack([-x2b / d, -a * x2b * 2 * logx / d], axis=1)

    tight = least_squares(
        residual,
        params,
        jac=jacobian,
        method="lm",
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
        max_nfev=100000,
    )
    return {
        "a": float(params[0]),
        "b": float(params[1]),
        "tight_a": float(tight.x[0]),
        "tight_b": float(tight.x[1]),
    }


def model(case, seed):
    from cuml.manifold import UMAP

    return UMAP(
        n_neighbors=case["n_neighbors"],
        n_components=case["n_components"],
        metric=case["metric"],
        n_epochs=case.get("n_epochs"),
        learning_rate=case["learning_rate"],
        min_dist=case["min_dist"],
        spread=case["spread"],
        a=case["a"],
        b=case["b"],
        init=case["init"],
        negative_sample_rate=case["negative_sample_rate"],
        random_state=seed,
        build_algo=case.get("build_algo", "brute_force_knn"),
        output_type="numpy",
    )


def fitted(case, seed):
    X = np.ascontiguousarray(case["X"], dtype=np.float32)
    y = case.get("y")
    m = model(case, seed)
    m.fit(X, y=None if y is None else np.asarray(y, dtype=np.float32))
    return m, X


def umap(case):
    m, _ = fitted(case, case["seed"])
    out = {
        "embedding": host(m.embedding_).astype(np.float64).tolist(),
        "a": float(m._a),
        "b": float(m._b),
        "n_neighbors": int(m._n_neighbors),
    }
    if case.get("Y") is not None:
        Y = np.ascontiguousarray(case["Y"], dtype=np.float32)
        out["transform"] = host(m.transform(Y)).astype(np.float64).tolist()
    return out


def quality(case):
    from cuml.cluster import HDBSCAN
    from cuml.metrics import adjusted_rand_score, trustworthiness

    trust, ari = [], []
    truth = case.get("truth")
    for seed in case["seeds"]:
        m, X = fitted(case, seed)
        if case.get("Y") is not None:
            Y = np.ascontiguousarray(case["Y"], dtype=np.float32)
            E, scored = host(m.transform(Y)), Y
        else:
            E, scored = host(m.embedding_), X
        trust.append(float(trustworthiness(scored, E, n_neighbors=10)))
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


def digits(case):
    from sklearn.datasets import load_digits

    d = load_digits()
    return {"X": d.data.astype(np.float64).tolist(), "y": d.target.astype(int).tolist()}


def params(case):
    from cuml.manifold import UMAP

    return {
        k: v
        for k, v in UMAP().get_params().items()
        if v is None or isinstance(v, (bool, int, float, str))
    }


def versions():
    import scipy

    out = {"scipy": scipy.__version__}
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
        "ab": ab,
        "umap": umap,
        "quality": quality,
        "digits": digits,
        "params": params,
    }
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": versions()}, f)


if __name__ == "__main__":
    main()
