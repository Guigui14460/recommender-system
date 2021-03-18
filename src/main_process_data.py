import os
import shutil

from recommender_system.data_processor.process_data import ProcessData


if __name__ == "__main__":
    if os.path.exists(ProcessData.PROCESSED_DATA_DIRECTORY):
        shutil.rmtree(ProcessData.PROCESSED_DATA_DIRECTORY)
    os.makedirs(ProcessData.PROCESSED_DATA_DIRECTORY)
    ProcessData.save_model()
