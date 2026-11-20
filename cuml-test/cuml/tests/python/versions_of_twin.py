"""Twin of cuml/tests/twin-common-test.rkt: run(versions_of=...) records the named modules."""

from params_of_twin import Estimator
from twin_common import params_of, run

if __name__ == "__main__":
    run({"params": params_of(Estimator)}, versions_of=("numpy",))
