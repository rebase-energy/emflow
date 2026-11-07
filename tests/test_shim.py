import warnings


def test_emflow_forwards_to_sdflow() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import emflow
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    import sdflow
    from emflow.models import Predictor
    from emflow.problems.registry import load_problem

    assert Predictor is sdflow.models.Predictor
    assert load_problem is sdflow.problems.registry.load_problem
    assert emflow.Problem is sdflow.Problem
    assert emflow.__version__ == "0.3.2"


def test_energy_assets_keep_their_old_path() -> None:
    import emflow
    from emflow.assets.wind import WindFarm
    import sdflow.energy

    assert WindFarm is sdflow.energy.WindFarm
    assert emflow.PVSystem is sdflow.energy.PVSystem
