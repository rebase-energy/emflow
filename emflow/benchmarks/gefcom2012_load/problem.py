"""GEFCom2012 load track — registered, data not yet ingested (see manifest.yaml).

Licensing note: sourced from Kaggle — the HF dataset repo must stay private.
"""

from emflow.problems import ProblemNotIngestedError


def get_problem():
    raise ProblemNotIngestedError(
        "gefcom2012-load is registered but its dataset has not been ingested yet — "
        "see emflow/benchmarks/gefcom2012_load/manifest.yaml (Kaggle source; keep repo private)"
    )
