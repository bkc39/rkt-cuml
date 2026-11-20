import math

import cupy as cp
import pandas as pd
from twin_common import run


def rows(case):
    names = case["names"]
    frame = pd.DataFrame(dict(zip(names, case["columns"])))
    host = frame[names].to_numpy(dtype=case["dtype"])
    return cp.asnumpy(cp.asarray(host)).tolist()


def signs(rows):
    return [[math.copysign(1.0, x) for x in row] for row in rows]


def selection(case):
    got = rows(case)
    return {"name": case["name"], "rows": got, "signs": signs(got)}


if __name__ == "__main__":
    run({"selection": selection})
