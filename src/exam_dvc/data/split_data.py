import pandas as pd
from sklearn.model_selection import train_test_split

from exam_dvc.params import (
    X_RAW_DATA_PATH,
    Y_RAW_DATA_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    TRAIN_RATIO
)


# Load X and y
X = pd.read_csv(X_RAW_DATA_PATH, index_col=0)
y = pd.read_csv(Y_RAW_DATA_PATH, index_col=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_RATIO, random_state=42)

# Save to data/preprocessed
X_train.to_csv(X_TRAIN_PATH)
X_test.to_csv(X_TEST_PATH)
y_train.to_csv(Y_TRAIN_PATH)
y_test.to_csv(Y_TEST_PATH)