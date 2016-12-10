import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from utils import subdivise_data


def fitting_regularized_lin_reg(data_subdivised, set, lasso_ridge="ridge"):

    print('Fitting the regression for the {}'.format(set))

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(data_subdivised[set]['X_train']))
    X_test = pd.DataFrame(scaler.transform(data_subdivised[set]['X_test']))
    Y_train = data_subdivised[set]['Y_train']
    Y_test = data_subdivised[set]['Y_test']

    if lasso_ridge == "ridge":
        reg = linear_model.RidgeCV(alphas=[1e-2, 0.1, 1.0, 10.0, 100.0])
    elif lasso_ridge == "lasso":
        reg = linear_model.LassoCV(cv=20)

    reg.fit(X_train, Y_train)
    coefficients = reg.coef_
    residuals = (reg.predict(X_test).reshape(Y_test.size, 1) - Y_test)
    residuals.columns = ['residuals']
    ms_error = np.mean(residuals ** 2)
    r_squared = reg.score(X_test, Y_test)

    print('Coefficients: \n', coefficients)
    print('Mean squared error: {}'.format(ms_error))
    print('Variance score: {}'.format(r_squared))

    return reg, coefficients, residuals, ms_error, r_squared


def regularized_lin_reg_(input_X, input_Y, test_size=0.2, number_sets=10, lasso_ridge="ridge"):

    data_subdivised = subdivise_data(input_X, input_Y, test_size, number_sets)

    ms_errors = {}
    var_scores = {}
    estimated_coefs = {}
    for set in data_subdivised.keys():

        reg, coefficients, residuals, ms_error, r_squared = fitting_regularized_lin_reg(data_subdivised, set, lasso_ridge)

        estimated_coefs[set] = coefficients
        ms_errors[set] = ms_error[0]
        var_scores[set] = r_squared

    optimal_set = min(ms_errors, key=ms_errors.get)
    reg, coefficients, residuals, ms_error, r_squared = fitting_regularized_lin_reg(data_subdivised, optimal_set, lasso_ridge)

    return reg, residuals, data_subdivised[optimal_set]