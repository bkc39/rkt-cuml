"""Writes private/normals-0.rktd: the 600 standard normals numpy's
default_rng(0) draws first. The HDBSCAN fixtures b50, b10 and one-blob are
built from them, as the probes of rkt-cuml #12 drew them."""

import sys

import numpy as np


def main():
    draws = np.random.default_rng(0).normal(0, 1.0, (300, 2))
    with open(sys.argv[1], "w") as f:
        f.write("(\n")
        for a, b in draws:
            f.write(f" ({float(a)!r} {float(b)!r})\n")
        f.write(")\n")


if __name__ == "__main__":
    main()
