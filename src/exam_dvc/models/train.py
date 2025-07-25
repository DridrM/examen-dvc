import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from exam_dvc.params import (
    X_TRAIN_SCALED_PATH,
    Y_TRAIN_PATH,
    BEST_PARAMS_PATH,
    BEST_MODEL_PATH
)


# Load the train scaled data
X_train_scaled = pd.read_csv(X_TRAIN_SCALED_PATH, index_col=0)
y_train = pd.read_csv(Y_TRAIN_PATH, index_col=0)

# Load best params
best_params = pd.read_pickle(BEST_PARAMS_PATH)

# Instanciate the model with best params
best_model = RandomForestRegressor(**best_params)

# Fit model on train
best_model.fit(X_train_scaled, y_train.iloc[:, 0])

# Save best model
with open(BEST_MODEL_PATH, "wb") as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)