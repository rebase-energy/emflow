"""GEFCom2014 — probabilistic load / price / wind / solar forecasting.

Registered but not yet ingested: see ``manifest.yaml`` for sources and the
ingestion plan. The solar track's raw CSVs already ship in
``emflow/examples/data`` (from the legacy example) and are the natural first
track to ingest.
"""

from emflow.problems import ProblemNotIngestedError

_MSG = ("gefcom2014:{track} is registered but its dataset has not been ingested yet — "
        "see emflow/benchmarks/gefcom2014/manifest.yaml for sources and plan")


def list_problem_variants():
    return ["load", "price", "wind", "solar"]


def get_problem_load():
    raise ProblemNotIngestedError(_MSG.format(track="load"))


def get_problem_price():
    raise ProblemNotIngestedError(_MSG.format(track="price"))


def get_problem_wind():
    raise ProblemNotIngestedError(_MSG.format(track="wind"))


def get_problem_solar():
    raise ProblemNotIngestedError(_MSG.format(track="solar"))
