import pandas as pd
from sklearn.preprocessing import StandardScaler

from exam_dvc.params import (
    X_TRAIN_PATH,
    X_TEST_PATH,
    X_TRAIN_SCALED_PATH,
    X_TEST_SCALED_PATH
)


# Load X_train, X_test
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

# Instanciate the standard scaler
std_scaler = StandardScaler()

# Fit on train
std_scaler.fit(X_train)

# Transform train and test
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

# Save into data/processed_data
pd.DataFrame(
    X_train_scaled, 
    columns=std_scaler.get_feature_names_out()
).to_csv(X_TRAIN_SCALED_PATH)
pd.DataFrame(
    X_test_scaled,
    columns=std_scaler.get_feature_names_out()
).to_csv(X_TEST_SCALED_PATH)