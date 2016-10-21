import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils import subdivise_data


def fitting_reg(data_subdivised, set):

    print('Fitting the regression for the {}'.format(set))

    X_train = data_subdivised[set]['X_train']
    X_test = data_subdivised[set]['X_test']
    Y_train = data_subdivised[set]['Y_train']
    Y_test = data_subdivised[set]['Y_test']

    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    coefficients = reg.coef_
    residuals = (reg.predict(X_test) - Y_test)
    ms_error = np.mean(residuals ** 2)
    r_squared = reg.score(X_test, Y_test)

    print('Coefficients: \n', coefficients)
    print('Mean squared error: {}'.format(ms_error))
    print('Variance score: {}'.format(r_squared))

    return reg, coefficients, residuals, ms_error, r_squared


def lin_reg_(input_X, input_Y, test_size=0.2, number_sets=10):

    input_X_std = pd.DataFrame(StandardScaler().fit_transform(input_X))
    input_X_std.columns = input_X.columns

    data_subdivised = subdivise_data(input_X_std, input_Y, test_size, number_sets)

    ms_errors = {}
    var_scores = {}
    estimated_coefs = {}
    for set in data_subdivised.keys():

        reg, coefficients, residuals, ms_error, r_squared = fitting_reg(data_subdivised, set)

        estimated_coefs[set] = coefficients
        ms_errors[set] = ms_error[0]
        var_scores[set] = r_squared

    optimal_set = min(ms_errors, key=ms_errors.get)
    reg, coefficients, residuals, ms_error, r_squared = fitting_reg(data_subdivised, optimal_set)

    return reg, residuals