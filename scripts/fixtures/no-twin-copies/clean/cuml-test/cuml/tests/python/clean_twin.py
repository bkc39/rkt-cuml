import json

from twin_common import host, run


def rows():
    return json.dumps(host([1.5]))


if __name__ == "__main__":
    run([rows])
