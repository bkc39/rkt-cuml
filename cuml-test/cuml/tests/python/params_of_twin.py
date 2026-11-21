"""Twin of cuml/tests/twin-common-test.rkt: params_of under run(fixtures=True)."""

from twin_common import params_of, run


class Estimator:
    def get_params(self):
        return {"alpha": 1.5, "solver": "auto"}


if __name__ == "__main__":
    run({"params": params_of(Estimator)}, fixtures=True)
