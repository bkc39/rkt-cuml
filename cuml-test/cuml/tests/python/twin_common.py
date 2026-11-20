"""The runner, the version record and the bits-only payload the twins share."""

import json
import sys

import numpy as np

BITS = {np.dtype(np.float32): np.uint32, np.dtype(np.float64): np.uint64}
WIDTH = {"float32": np.uint32, "float64": np.uint64}


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def put(out, key, a):
    a = np.ascontiguousarray(host(a))
    out[key + "_bits"] = a.view(BITS[a.dtype]).ravel().tolist()


def from_bits(bits, dtype):
    return np.asarray(bits, dtype=WIDTH[dtype]).view(np.dtype(dtype))


def inputs(case):
    dt = case["dtype"]
    n = len(case["y"])
    X = from_bits(case["X"], dt).reshape(n, -1)
    y = from_bits(case["y"], dt)
    w = None if case["w"] is None else from_bits(case["w"], dt)
    return X, y, w


def fitted(m, dtype):
    out = {}
    put(out, "coef", np.asarray(host(m.coef_), dtype=dtype))
    put(out, "intercept", np.asarray([m.intercept_], dtype=dtype))
    return out


def versions(**extra):
    out = {}
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import cuml

        out["cuml"] = cuml.__version__
    except ImportError:
        pass
    out.update(extra)
    return out


def params_of(cls):
    return lambda case: cls().get_params()


def run(ops, *, fixtures=False, version=None):
    with open(sys.argv[1]) as f:
        given = json.load(f)
    if fixtures:
        shared = given.get("fixtures", {})
        results = [ops[case["op"]](shared, case) for case in given["cases"]]
    else:
        results = [ops[case["op"]](case) for case in given["cases"]]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": version or versions()}, f)
