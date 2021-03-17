import abc
from typing import Any

import pandas as pd


class Recommender(abc.ABC):
    @abc.abstractmethod
    def recommend(self, **kwargs) -> pd.DataFrame:
        pass
