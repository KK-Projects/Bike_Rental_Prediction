import numpy as np
from sklearn import svm

from features_description import divise_in_classes

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def fitting_svm(data_subdivised, set, n_estimators=100):

    print('Fitting the regression for the {}'.format(set))

    X_train = np.array(data_subdivised[set]['X_train'])
    X_test = np.array(data_subdivised[set]['X_test'])
    Y_train = np.array(data_subdivised[set]['Y_train'])
    Y_test = np.array(data_subdivised[set]['Y_test'])

    classification = svm.SVC()
    classification.fit(X_train, Y_train)
    classification.predict(X_test)
    # get support vectors
    classification.support_vectors_
    # get indices of support vectors
    classification.support_
    # get number of support vectors for each class
    classification.n_support_

    errors = (classification.predict(X_test) - Y_test)
    error_rate = np.mean(errors ** 2)/ len(Y_test)
    print('Mean squared error: {}'.format(error_rate))

    return classification, errors, error_rate

# def svc(input_X, input_Y, test_size=0.2, number_sets=10):


