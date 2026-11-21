import json
import sys


def main(out):
    with open(sys.argv[2], "w") as f:
        f.write(json.dumps(out))
