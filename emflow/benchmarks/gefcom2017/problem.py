"""GEFCom2017 — registered, data not yet ingested (see manifest.yaml)."""

from emflow.problems import ProblemNotIngestedError


def get_problem():
    raise ProblemNotIngestedError(
        "gefcom2017 is registered but its dataset has not been ingested yet — "
        "see emflow/benchmarks/gefcom2017/manifest.yaml"
    )
