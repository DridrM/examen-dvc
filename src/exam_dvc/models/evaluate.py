import json

import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

from exam_dvc.params import (
    BEST_MODEL_PATH,
    X_TEST_SCALED_PATH,
    Y_TEST_PATH,
    METRICS_JSON_DUMP
)


# Load the test scaled data
X_test_scaled = pd.read_csv(X_TEST_SCALED_PATH, index_col=0)
y_test = pd.read_csv(Y_TEST_PATH, index_col=0)

# Load best model
best_model = pd.read_pickle(BEST_MODEL_PATH)

# Predict with X test
y_pred = best_model.predict(X_test_scaled)

# Evaluate with the MAE, the RMSE and the R2
scores = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": root_mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}

# Make a json dump of the metrics
with open(METRICS_JSON_DUMP, "w") as outfile:
    json.dump(scores, outfile, indent=4)