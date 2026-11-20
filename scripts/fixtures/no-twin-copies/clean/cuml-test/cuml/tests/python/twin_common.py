import json
import sys


def host(a):
    return a


def run(ops):
    with open(sys.argv[2], "w") as f:
        json.dump({"results": [op() for op in ops]}, f)
