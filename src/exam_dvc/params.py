import os
from pathlib import Path


# Local project path define in the .env: Make navigating the project folder easier
LOCAL_PROJECT_PATH = Path(os.environ.get("LOCAL_PROJECT_PATH"))


#####################################
# Raw data import and preprocessing #
#####################################

# Raw csv data url
RAW_DATA_URL = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

# Name of the target col
TARGET_COL = "silica_concentrate"

# Name of the datetime col to drop.
# We choose to make the modelisation without
# datetime column because we modelize an industrial
# process, that in theory run 24/7. Time information
# may indeed lead to add noise and worsen generalization.
DT_COL = "date"

# Local raw data path
RAW_DATA_PATH = LOCAL_PROJECT_PATH / "data" / "raw_data"

# X and y raw data path
X_RAW_DATA_PATH = RAW_DATA_PATH / "X_raw.csv"
Y_RAW_DATA_PATH = RAW_DATA_PATH / "y_raw.csv"


#####################
# Split data params #
#####################

# Train test split train ratio
TRAIN_RATIO = 0.7

# Preprocessed data path
PREPROCESSED_DATA_PATH = LOCAL_PROJECT_PATH / "data" / "processed_data"

# Data path for split X and y
X_TRAIN_PATH = PREPROCESSED_DATA_PATH / "X_train.csv"
X_TEST_PATH = PREPROCESSED_DATA_PATH / "X_test.csv"
Y_TRAIN_PATH = PREPROCESSED_DATA_PATH / "y_train.csv"
Y_TEST_PATH = PREPROCESSED_DATA_PATH / "y_test.csv"

# Data path for scaled features
X_TRAIN_SCALED_PATH = PREPROCESSED_DATA_PATH / "X_train_scaled.csv"
X_TEST_SCALED_PATH = PREPROCESSED_DATA_PATH / "X_test_scaled.csv"
