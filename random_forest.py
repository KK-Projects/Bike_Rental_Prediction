import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, mean_squared_error

from utils import subdivise_data


def fitting_forest(data_subdivised, set, n_estimators=100):

    print('Fitting the regression for the {}'.format(set))

    X_train = data_subdivised[set]['X_train']
    X_test = data_subdivised[set]['X_test']
    Y_train = data_subdivised[set]['Y_train']
    Y_test = data_subdivised[set]['Y_test']

    forest = RandomForestRegressor(n_estimators=n_estimators, max_features="auto")
    fit_forest = forest.fit(X_train, Y_train)
    residuals = (forest.predict(X_test) - Y_test)
    ms_error = np.mean(residuals ** 2)

    print('Mean squared error: {}'.format(ms_error))

    return fit_forest, residuals, ms_error


def lin_reg_(input_X, input_Y, test_size=0.2, number_sets=10):

    input_X_std = pd.DataFrame(StandardScaler().fit_transform(input_X))
    input_X_std.columns = input_X.columns

    data_subdivised = subdivise_data(input_X_std, input_Y, test_size, number_sets)

    ms_errors = {}
    for set in data_subdivised.keys():

        fit_forest, residuals, ms_error = fitting_forest(data_subdivised, set, n_estimators=100)

        ms_errors[set] = ms_error[0]

    optimal_set = min(ms_errors, key=ms_errors.get)
    fit_forest, residuals, ms_error = fitting_forest(data_subdivised, optimal_set, n_estimators=100)

    return fit_forest, residuals
