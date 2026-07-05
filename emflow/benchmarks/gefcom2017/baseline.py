"""Reference baseline for GEFCom2017: hour-of-day climatology deciles.

The generic climatology baseline over a trailing year of zonal demand — no
temperature, no calendar effects beyond hour of day. Beats persistence by a
wide margin; the published field beat *this* by using weather and holiday
structure, which is the gap agents are meant to close.
"""

from __future__ import annotations

from ..gefcom2014.baseline import ClimatologyQuantiles
from .problem import QUANTILES


def get_model() -> ClimatologyQuantiles:
    """Submission-convention factory: a fresh, untrained baseline."""
    return ClimatologyQuantiles(window="365D", quantiles=QUANTILES,
                                target_field="load")
