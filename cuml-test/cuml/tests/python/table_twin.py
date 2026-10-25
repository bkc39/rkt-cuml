"""Twin of cuml/tests/table-twin-test.rkt: each case's named columns, as a
pandas DataFrame, through to_numpy at the case's dtype and onto the GPU with
cupy."""

import json
import sys

import cupy as cp
import pandas as pd


def rows(case):
    names = case["names"]
    frame = pd.DataFrame(dict(zip(names, case["columns"])))
    host = frame[names].to_numpy(dtype=case["dtype"])
    return cp.asnumpy(cp.asarray(host)).tolist()


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    out = [{"name": case["name"], "rows": rows(case)} for case in cases]
    with open(sys.argv[2], "w") as f:
        json.dump({"cases": out}, f)


if __name__ == "__main__":
    main()
