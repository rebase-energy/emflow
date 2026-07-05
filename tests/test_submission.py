"""load_submission + evaluate: the stable external integration surface."""

import textwrap

import pandas as pd
import pytest

import emflow as ef
from emflow.run.submission import evaluate, load_submission

from conftest import make_toy_problem

PERSISTENCE_SOURCE = textwrap.dedent(
    '''
    import pandas as pd
    from emflow.models.predictor import Predictor


    class Persistence(Predictor):
        def predict(self, obs):
            last = obs.history("power")["power"].dropna().iloc[-1]
            return pd.DataFrame({"point": last}, index=obs.target_index)
    '''
)


def write_submission(tmp_path, body: str, name="submission.py"):
    path = tmp_path / name
    path.write_text(PERSISTENCE_SOURCE + textwrap.dedent(body))
    return path


class TestLoadSubmission:
    def test_factory_form(self, tmp_path):
        path = write_submission(tmp_path, "\ndef get_model():\n    return Persistence()\n")
        model = load_submission(path)
        assert isinstance(model, ef.Predictor)

    def test_named_model_form(self, tmp_path):
        path = write_submission(tmp_path, "\nmodel = Persistence()\n")
        assert isinstance(load_submission(path), ef.Predictor)

    def test_sole_instance_form(self, tmp_path):
        path = write_submission(tmp_path, "\nsomething = Persistence()\n")
        assert isinstance(load_submission(path), ef.Predictor)

    def test_ambiguous_instances_rejected(self, tmp_path):
        path = write_submission(tmp_path, "\na = Persistence()\nb = Persistence()\n")
        with pytest.raises(AttributeError, match="several"):
            load_submission(path)

    def test_no_model_rejected(self, tmp_path):
        path = tmp_path / "empty.py"
        path.write_text("x = 1\n")
        with pytest.raises(AttributeError, match="get_model"):
            load_submission(path)


class TestEvaluate:
    def test_evaluate_predictor_instance(self):
        problem = make_toy_problem()

        class Persistence(ef.Predictor):
            def predict(self, obs):
                last = obs.history("power")["power"].dropna().iloc[-1]
                return pd.DataFrame({"point": last}, index=obs.target_index)

        result = evaluate(problem, Persistence(), split="validation")
        assert result.n_scored > 0
        assert result.score == pytest.approx(result.score)  # a real float

    def test_evaluate_from_path(self, tmp_path):
        problem = make_toy_problem()
        path = write_submission(tmp_path, "\ndef get_model():\n    return Persistence()\n")
        result = evaluate(problem, path, split="validation")
        assert result.n_scored > 0

    def test_splits_score_independently(self, tmp_path):
        problem = make_toy_problem()
        path = write_submission(tmp_path, "\ndef get_model():\n    return Persistence()\n")
        val = evaluate(problem, path, split="validation")
        hold = evaluate(problem, path, split="holdout")
        assert val.n_scored > 0 and hold.n_scored > 0

    def test_top_level_exports(self):
        assert ef.evaluate is evaluate
        assert ef.load_submission is load_submission
