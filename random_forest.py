import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from utils import cross_validate


def rand_forest_reg(input_X, input_Y, max_depth=None, max_features="auto", nb_folds=10, n_estimators=100):

    forest = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth)
    predictions = cross_validate(input_X, input_Y, forest, nb_folds=nb_folds)
    residuals = predictions - input_Y
    ms_error = np.mean(residuals ** 2)

    residuals = pd.DataFrame(residuals)
    residuals.columns = ['residuals']

    return predictions, residuals, ms_error
