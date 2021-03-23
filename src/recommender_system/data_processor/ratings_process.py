import pandas as pd

from recommender_system import constants


class RatingsProcess:
    """Process the ratings."""

    def __init__(self, small_dataset: bool = True) -> None:
        """Initialize the instance.

        Parameters :
        ------------
            small_dataset : bool = True
                boolean to know if it loads the small dataset or the large dataset
        """
        ratings_filename = constants.USERS_RATINGS_SMALL if small_dataset else constants.USERS_RATINGS
        self.data = pd.read_csv(
            f'{constants.DATA_DIRECTORY}/{ratings_filename}')
        self.data["movie_id"] = self.data["movieId"]
        self.data.drop(inplace=True, columns=["timestamp", "movieId"])
