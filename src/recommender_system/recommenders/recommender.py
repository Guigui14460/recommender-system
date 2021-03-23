import abc
from typing import Any

import pandas as pd


class Recommender(abc.ABC):
    """Abstract class used to create system recommenders."""

    @abc.abstractmethod
    def recommend(self, **kwargs) -> pd.DataFrame:
        """Recommend movies in function of recommender."""
        pass
