"""Twin of cuml/tests/spectral-embedding-*-twin-test.rkt: cuml's
spectral_embedding embeds the data or graph the Racket side used, and
sklearn's spectral_embedding and SpectralEmbedding are the references."""

import json
import sys

import numpy as np


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def sign_rule(E):
    E = np.array(E, dtype=np.float32, copy=True)
    first = np.argmax(np.abs(E), axis=0)
    signs = np.sign(E[first, np.arange(E.shape[1])])
    signs[signs == 0] = 1
    return E * signs.astype(np.float32)


def rows(case):
    return np.ascontiguousarray(case["X"], dtype=np.float32)


def listed(E):
    return np.asarray(E).astype(np.float64).tolist()


def cuml_embedding(data, case, affinity, n_neighbors):
    from cuml.manifold import spectral_embedding

    return host(
        spectral_embedding(
            data,
            n_components=case["k"],
            affinity=affinity,
            n_neighbors=n_neighbors,
            random_state=case["seed"],
            norm_laplacian=case["norm"],
            drop_first=case["drop"],
        )
    ).astype(np.float32)


def embed(case):
    E = cuml_embedding(rows(case), case, "nearest_neighbors", case["n_neighbors"])
    return {"embedding": listed(sign_rule(E))}


def raw_tiny(case):
    X = np.ldexp(rows(case), -70).astype(np.float32)
    E = cuml_embedding(X, case, "nearest_neighbors", case["n_neighbors"])
    return {"embedding": listed(E)}


def device_graph(case):
    import cupy as cp
    import cupyx.scipy.sparse as cps

    n = case["n"]
    return cps.coo_matrix(
        (
            cp.asarray(case["vals"], dtype=cp.float32),
            (
                cp.asarray(case["rows"], dtype=cp.int32),
                cp.asarray(case["cols"], dtype=cp.int32),
            ),
        ),
        shape=(n, n),
    )


def graph(case):
    E = cuml_embedding(device_graph(case), case, "precomputed", None)
    return {"embedding": listed(sign_rule(E))}


def host_graph(case):
    import scipy.sparse as sp

    n = case["n"]
    return sp.csr_matrix(
        (
            np.asarray(case["vals"], dtype=np.float64),
            (np.asarray(case["rows"]), np.asarray(case["cols"])),
        ),
        shape=(n, n),
    )


def sklearn_graph(case):
    from sklearn.manifold import spectral_embedding

    E = spectral_embedding(
        host_graph(case),
        n_components=case["k"],
        norm_laplacian=case["norm"],
        drop_first=case["drop"],
        random_state=0,
    )
    return {"embedding": listed(E)}


def sklearn_dense(case):
    from sklearn.manifold import SpectralEmbedding

    E = SpectralEmbedding(
        n_components=case["k"],
        affinity="nearest_neighbors",
        n_neighbors=case["n_neighbors"],
        random_state=0,
    ).fit_transform(np.asarray(case["X"], dtype=np.float64))
    return {"embedding": listed(E)}


def params(case):
    from cuml.manifold import SpectralEmbedding

    return {
        k: v
        for k, v in SpectralEmbedding().get_params().items()
        if v is None or isinstance(v, (bool, int, float, str))
    }


def digits(case):
    from sklearn.datasets import load_digits

    d = load_digits()
    return {"X": d.data.tolist(), "y": d.target.tolist()}


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
        "embed": embed,
        "raw-tiny": raw_tiny,
        "graph": graph,
        "sklearn": sklearn_graph,
        "sklearn-dense": sklearn_dense,
        "params": params,
        "digits": digits,
    }
    results = [ops[case["op"]](case) for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": versions()}, f)


if __name__ == "__main__":
    main()
