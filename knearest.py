import numpy as np
import pandas as pd

from sklearn import neighbors

from utils import cross_validate


def k_nearest_neighbors(input_X, input_Y, nb_folds=10, n_neighbors=15):

    k_nearest = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    predictions = cross_validate(input_X, input_Y, k_nearest, nb_folds=nb_folds)
    residuals = predictions - input_Y
    ms_error = np.mean(residuals ** 2)

    residuals = pd.DataFrame(residuals)
    residuals.columns = ['residuals']

    return predictions, residuals, ms_error
