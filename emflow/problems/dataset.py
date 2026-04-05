from dataclasses import dataclass
import typing as t
import pandas as pd

import energydatamodel as edm

@dataclass
class Dataset:
    name: str
    description: t.Optional[str] = None
    collection: t.Optional[edm.EnergyCollection] = None
    data: t.Optional[t.Dict[str, pd.DataFrame]] = None

    @property
    def list_data(self):
        return list(self.data.keys())