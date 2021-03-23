import os
import shutil

from recommender_system import constants
from recommender_system.data_processor import ProcessData


if __name__ == "__main__":
    if os.path.exists(constants.PROCESSED_DATA_DIRECTORY):
        shutil.rmtree(constants.PROCESSED_DATA_DIRECTORY)
    os.makedirs(constants.PROCESSED_DATA_DIRECTORY)
    ProcessData.save_model()
