import numpy as np
import pandas as pd

from sklearn import neighbors
from sklearn import grid_search
from utils import cross_validate
from utils import cross_validate_with_grid


def k_nearest_neighbors(input_X, input_Y, nb_folds=10, n_neighbors=15, weights="uniform", algo="auto"):

    k_nearest = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algo)
    predictions = cross_validate(input_X, input_Y, k_nearest, nb_folds=nb_folds)
    residuals = predictions - input_Y
    ms_error = np.mean(residuals ** 2)

    residuals = pd.DataFrame(residuals)
    residuals.columns = ['residuals']

    return predictions, residuals, ms_error


def k_nearest_neighbors_with_grid(input_X, input_Y, knn_neighbors, knn_weights, knn_algorithm, nb_folds=10):

    X = np.asarray(input_X)
    y = np.asarray(input_Y).reshape(input_Y.size)
    param_grid = {'algorithm': knn_algorithm, 'weights': knn_weights, 'n_neighbors': knn_neighbors}
    k_nearest = grid_search.GridSearchCV(neighbors.KNeighborsClassifier(), param_grid)
    predictions = cross_validate_with_grid(X, y, k_nearest, nb_folds=nb_folds)
    residuals = predictions - y
    ms_error = np.mean(residuals ** 2)

    residuals = pd.DataFrame(residuals)
    residuals.columns = ['residuals']

    return predictions, residuals, ms_error
