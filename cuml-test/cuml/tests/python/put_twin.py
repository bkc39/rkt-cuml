"""Twin of cuml/tests/twin-common-test.rkt: put writes the bits, and with float64= the float64 list too."""

import numpy as np
from twin_common import put, run


def stored(case, **options):
    out = {}
    put(out, "a", np.asarray(case["values"], dtype=case["dtype"]), **options)
    return out


if __name__ == "__main__":
    run({"put": stored, "put_float64": lambda case: stored(case, float64=True)})
