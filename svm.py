import numpy as np
import pandas as pd
from sklearn import svm

from utils import cross_validate
from sklearn.metrics import confusion_matrix


def svc(categorical_input_X, class_input_Y, nb_folds= 10, C=1.0, kernel='rbf', degree=3, gamma='auto'):

    classification = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)

    predictions = cross_validate(categorical_input_X, class_input_Y, classification, nb_folds=nb_folds)

    conf_matrix = confusion_matrix(class_input_Y, predictions)
    errors = (predictions - class_input_Y)

    errors = np.array([1 if error != 0 else error for error in errors])
    error_rate = float(np.sum(errors ** 2)) / len(class_input_Y)
    print('Error rate of misclassification: {}'.format(error_rate))

    return classification, conf_matrix


def svr(input_X, input_Y, nb_folds=10, C=1.0, kernel='rbf', degree=3, gamma='auto'):

    classification = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)

    predictions = cross_validate(input_X, input_Y, classification, nb_folds=nb_folds)

    residuals = predictions - input_Y
    ms_error = np.mean(residuals ** 2)

    residuals = pd.DataFrame(residuals)
    residuals.columns = ['residuals']

    return predictions, residuals, ms_error

