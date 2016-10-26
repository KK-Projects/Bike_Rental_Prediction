import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from utils import subdivise_data


def fitting_forest(data_subdivised, set, n_estimators=100):

    print('Fitting the random forest for the {}'.format(set))

    X_train = np.array(data_subdivised[set]['X_train'])
    X_test = np.array(data_subdivised[set]['X_test'])
    Y_train = np.array(data_subdivised[set]['Y_train'])
    Y_test = np.array(data_subdivised[set]['Y_test'])

    forest = RandomForestRegressor(n_estimators=n_estimators, max_features="auto")
    fit_forest = forest.fit(X_train, np.transpose(Y_train)[0])
    residuals = (forest.predict(X_test) - np.transpose(Y_test)[0])
    ms_error = np.mean(residuals ** 2)
    print('Mean squared error: {}'.format(ms_error))

    return fit_forest, residuals, ms_error


def rand_forest_reg(input_X, input_Y, test_size=0.2, number_sets=10, n_estimators=100):

    data_subdivised = subdivise_data(input_X, input_Y, test_size, number_sets)

    ms_errors = {}
    for set in data_subdivised.keys():

        fit_forest, residuals, ms_error = fitting_forest(data_subdivised, set, n_estimators=n_estimators)
        ms_errors[set] = ms_error

    optimal_set = min(ms_errors, key=ms_errors.get)
    fit_forest, residuals, ms_error = fitting_forest(data_subdivised, optimal_set, n_estimators=n_estimators)

    return fit_forest, residuals, data_subdivised[optimal_set]
