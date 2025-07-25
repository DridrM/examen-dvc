import pickle

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from exam_dvc.params import (
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    BEST_PARAMS_PATH
)


# Load the train unscaled data: Indeed we will use grid search 
# with cross-validation, so we don't want data leakage between our folds
X_train = pd.read_csv(X_TRAIN_PATH, index_col=0)
y_train = pd.read_csv(Y_TRAIN_PATH, index_col=0)

# We build our predicting pipeline with our model
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestRegressor()
)

# The param grid will be composed of only the L2 regularization param
param_grid = {
    "randomforestregressor__n_estimators": np.arange(50, 500, 50),
    "randomforestregressor__max_depth": np.arange(1, 10),
    "randomforestregressor__max_features": np.linspace(0.1, 1., 10)
}

# Instanciate our grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1
)

# Fit our grid search
results = grid_search.fit(
    X_train, 
    y_train.iloc[:, 0]
)

# Extract best hyper-parameters
best_params = {param.split("__")[-1]: value for param, value in results.best_params_.items()}

# Save best params
with open(BEST_PARAMS_PATH, "wb") as handle:
    pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)