import numpy as np
from sklearn import svm

from utils import subdivise_data
from sklearn.metrics import confusion_matrix


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def fitting_svm(data_subdivised, set):

    print('Fitting the SVM Classification for the {}'.format(set))

    X_train = np.array(data_subdivised[set]['X_train'])
    X_test = np.array(data_subdivised[set]['X_test'])
    Y_train = np.array(data_subdivised[set]['Y_train'])
    Y_test = np.array(data_subdivised[set]['Y_test'])

    classification = svm.SVC()
    classification.fit(X_train, np.transpose(Y_train)[0])
    classification.predict(X_test)

    y_pred = classification.predict(X_test)
    y_true = np.transpose(Y_test)[0]
    conf_matrix = confusion_matrix(y_true, y_pred)
    errors = (y_pred - y_true )

    errors = np.array([1 if error != 0 else error for error in errors])
    error_rate = float(np.sum(errors ** 2)) / len(Y_test)
    print('Error rate of misclassification: {}'.format(error_rate))

    return classification, conf_matrix, error_rate


def svc(categorical_input_X , class_input_Y, test_size=0.2, number_sets=10):

    data_subdivised = subdivise_data(categorical_input_X, class_input_Y, test_size, number_sets)

    error_rates = {}
    for set in data_subdivised.keys():

        classification, conf_matrix, error_rate = fitting_svm(data_subdivised, set)
        error_rates[set] = error_rate

    optimal_set = min(error_rates, key=error_rates.get)
    classification, conf_matrix, error_rate = fitting_svm(data_subdivised, optimal_set)

    return classification, conf_matrix

