
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import subdivise_data
from sklearn.cross_validation import train_test_split
from statsmodels.tsa import arima_model

input_train_sample = pd.read_csv('train.csv')
output_test_sample = pd.read_csv('test.csv')
my_sub = pd.read_csv('my_submission.csv')

features = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

input_X = input_train_sample [features]
input_Y = input_train_sample[['cnt']]
output_X = output_test_sample[features]
test_size = 0.2
number_sets = 10

data_subdivised = subdivise_data(input_X, input_Y, test_size, number_sets)


def lin_reg_(input_X, input_Y):

    reg = LinearRegression()

    for set in data_subdivised.keys():

        X_train = data_subdivised[set]['X_train']
        X_test = data_subdivised[set]['X_test']
        Y_train = data_subdivised[set]['Y_train']
        Y_test = data_subdivised[set]['Y_test']

        reg.fit(X_train, Y_train)

        coefficients = reg.coef_
        print('Coefficients: \n', coefficients)
        ms_error = np.mean((reg.predict(X_test) - Y_test) ** 2)
        print('Mean squared error: {}'.format(ms_error))
        # Explained variance score: 1 is perfect prediction
        r_squared = reg.score(X_test, Y_test)
        print('Variance score: {}'.format(r_squared))


        # Plot outputs
        plt.scatter(Y_test, reg.predict(X_test),  color='black')

        plt.show()

