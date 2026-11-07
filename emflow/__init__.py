"""emflow is now sdflow.

This release of emflow is a compatibility shim: every ``emflow`` name and every
``emflow.<module>`` import resolves to the same object in `sdflow
<https://github.com/rebase-energy/sdflow>`_, the domain-neutral successor, so existing
code keeps working while it migrates. Replace ``import emflow`` with ``import sdflow``
(and ``emflow.PVSystem`` and the other asset classes with ``sdflow.energy.PVSystem``);
this shim will not be maintained beyond the 0.3 line.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings

import sdflow as _sdflow

__version__ = "0.3.2"

warnings.warn(
    "emflow has become sdflow: `import sdflow as sd` (energy asset classes: sdflow.energy). "
    "This emflow release only forwards to sdflow and will not be maintained.",
    DeprecationWarning,
    stacklevel=2,
)


class _Forwarder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``import emflow.x.y`` returns the ``sdflow.x.y`` module itself."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "emflow" or not fullname.startswith("emflow."):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        target = "sdflow" + spec.name[len("emflow") :]
        try:
            module = importlib.import_module(target)
        except ImportError:
            # Energy assets moved under sdflow.energy; keep the old path alive.
            if spec.name.startswith("emflow.assets"):
                module = importlib.import_module("sdflow.energy" + spec.name[len("emflow") :])
            else:
                raise
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        pass


if not any(isinstance(finder, _Forwarder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Forwarder())

# Top-level names: everything sdflow exports, plus the energy asset classes that
# emflow exported from its root.
from sdflow import *  # noqa: E402,F403
from sdflow import __all__ as _core_all  # noqa: E402

try:
    from sdflow.energy import *  # noqa: E402,F403
    from sdflow.energy import __all__ as _energy_all  # noqa: E402
except ImportError:  # sdflow installed without the energy extra
    _energy_all = []

__all__ = [*_core_all, *_energy_all, "__version__"]
# The star import above brought sdflow's version along; this package has its own.
__version__ = "0.3.2"


def __getattr__(name: str):
    return getattr(_sdflow, name)
