"""The runner, the version record and the bits-only payload the twins share."""

import importlib
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


def versions(*modules):
    out = {}
    for name in ("sklearn", "cuml", *modules):
        try:
            out[name] = importlib.import_module(name).__version__
        except ImportError:
            pass
    return out


def params_of(cls):
    return lambda *_: cls().get_params()


def run(ops, *, fixtures=False, versions_of=()):
    with open(sys.argv[1]) as f:
        given = json.load(f)
    shared = (given.get("fixtures", {}),) if fixtures else ()
    results = [ops[case["op"]](*shared, case) for case in given["cases"]]
    with open(sys.argv[2], "w") as f:
        json.dump({"results": results, "version": versions(*versions_of)}, f)
