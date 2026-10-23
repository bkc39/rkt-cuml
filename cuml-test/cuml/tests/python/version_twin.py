"""Twin of cuml/tests/twin-test.rkt: the cuML release the Python twin imports."""

import json
import sys

import cuml


def main():
    with open(sys.argv[2], "w") as f:
        json.dump({"version": cuml.__version__}, f)


if __name__ == "__main__":
    main()
