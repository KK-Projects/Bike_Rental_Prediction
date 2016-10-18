import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from utils import subdivise_data
from sklearn.preprocessing import StandardScaler


def lin_reg_(input_X, input_Y, split_data=True, test_size=0.2, number_sets=10):

    reg = LinearRegression()
    input_X_std = pd.DataFrame(StandardScaler().fit_transform(input_X))
    input_X_std.columns = input_X.columns

    if split_data:
        data_subdivised = subdivise_data(input_X_std, input_Y, test_size, number_sets)
    else:
        data_subdivised = subdivise_data(input_X_std, input_Y, 0, 1)

    for set in data_subdivised.keys():

        print('Fitting the regression for the {} out of {}'.format(set, number_sets))
        X_train = data_subdivised[set]['X_train']
        X_test = data_subdivised[set]['X_test']
        Y_train = data_subdivised[set]['Y_train']
        Y_test = data_subdivised[set]['Y_test']

        reg.fit(X_train, Y_train)

        coefficients = reg.coef_
        print('Coefficients: \n', coefficients)

        if split_data:
            ms_error = np.mean((reg.predict(X_test) - Y_test) ** 2)
            print('Mean squared error: {}'.format(ms_error))
            # Explained variance score: 1 is perfect prediction
            r_squared = reg.score(X_test, Y_test)
            print('Variance score: {}'.format(r_squared))
        else:
            # Plot outputs
            # plt.scatter(Y_test, reg.predict(X_test),  color='black')
            # plt.show()
            return reg