import json
import math
import sys

import cupy as cp
import pandas as pd


def rows(case):
    names = case["names"]
    frame = pd.DataFrame(dict(zip(names, case["columns"])))
    host = frame[names].to_numpy(dtype=case["dtype"])
    return cp.asnumpy(cp.asarray(host)).tolist()


def signs(rows):
    return [[math.copysign(1.0, x) for x in row] for row in rows]


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)["cases"]
    out = []
    for case in cases:
        got = rows(case)
        out.append({"name": case["name"], "rows": got, "signs": signs(got)})
    with open(sys.argv[2], "w") as f:
        json.dump({"cases": out}, f)


if __name__ == "__main__":
    main()
