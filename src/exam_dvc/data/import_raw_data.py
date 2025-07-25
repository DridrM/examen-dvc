import pandas as pd

from exam_dvc.params import (
    RAW_DATA_URL, 
    TARGET_COL,
    DT_COL,
    X_RAW_DATA_PATH,
    Y_RAW_DATA_PATH
)


# Read raw csv from remote s3 storage
raw_data = pd.read_csv(RAW_DATA_URL)

# Split into variables and target
X = raw_data.drop([TARGET_COL, DT_COL], axis=1)
y = raw_data[[TARGET_COL]]

# Save to data/raw_data
X.to_csv(X_RAW_DATA_PATH)
y.to_csv(Y_RAW_DATA_PATH)