import json
import sys


def main():
    with open(sys.argv[2], "w") as f:
        json.dump({"results": []}, f)
