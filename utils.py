import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold


def df_train_test_split(data_X, data_Y, test_size):
    """
    Function that gives the same result as train_test_split but as dataframe
    :param data_X:
    :param data_Y:
    :param test_size:
    :return:
    """
    features = data_X.columns
    output = data_Y.columns

    X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=test_size)

    df_X_train = pd.DataFrame(X_train)
    df_X_test = pd.DataFrame(X_test)
    df_Y_train = pd.DataFrame(Y_train)
    df_Y_test = pd.DataFrame(Y_test)

    df_X_train.columns = features
    df_X_test.columns = features
    df_Y_train.columns = output
    df_Y_test.columns = output

    return df_X_train, df_X_test, df_Y_train, df_Y_test


def subdivise_data(data_X, data_Y, test_size, number_sets):
    """
    This function takes the data of some inputs data_X and variable that you want to predict Y, and returns
    a dictionnary number of sets of train and test data according to the test size ( = data_test size / data_X)
    :param data_X: Dataframe
    :param data_Y: Dataframe
    :param test_size: float between 0 and 1
    :param number_sets:
    :return: Dict
    """
    dict_train = {}
    for i in range(number_sets):

        df_X_train, df_X_test, df_Y_train, df_Y_test = df_train_test_split(data_X, data_Y, test_size)

        set = 'set_{}'.format(i)
        dict_train[set] = {}
        dict_train[set]['X_train'] = df_X_train
        dict_train[set]['X_test'] = df_X_test
        dict_train[set]['Y_train'] = df_Y_train
        dict_train[set]['Y_test'] = df_Y_test

    return dict_train


def cat_var_to_dummies(categorical_input_X):
    """
    :param categorical_input_X: DataFrame of categorical variables
    :return:
    """

    df = pd.DataFrame()
    for col in categorical_input_X.columns:
        col_dummies = pd.get_dummies(categorical_input_X[col], prefix=col)
        df = pd.concat([df, col_dummies], axis=1)

    return df


def get_my_input(_input_X, cat_feat, non_cat_feat):
    """

    :param _input_X:
    :param cat_feat:
    :param non_cat_feat:
    :return:
    """
    categorical_input_X = _input_X[cat_feat]
    non_cat_input_X = _input_X[non_cat_feat]
    cat_input_X = cat_var_to_dummies(categorical_input_X)
    input_X = pd.concat([cat_input_X, non_cat_input_X], axis=1)
    return input_X


def cross_validate(_input_X, _input_Y, classifier, nb_folds=10):
    """ Perform a cross-validation and returns the predictions.
        Use a scaler to scale the features to mean 0, standard deviation 1.

        Parameters:
        -----------
        _input_X: (n_samples, n_features) np.array
        _input_Y: (n_samples, ) np.array
        classifier:  sklearn classifier object
            Classifier instance; must have the following methods:
            - fit(X, y) to train the classifier on the data X, y
            - predict(X) to apply the trained classifier to the data X and return probability estimates
        nb_folds: snumber of folds for the cross validation

        Return:
        -------
        pred: (n_samples, ) np.array
            Vectors of predictions
        """
    cv_folds = StratifiedKFold(np.array(_input_Y).reshape(_input_Y.shape[0]), nb_folds, shuffle=True)
    pred = np.zeros(_input_Y.shape)  # vector of 0 in which to store the predictions
    for tr, te in cv_folds:
        # Restrict data to train/test folds
        Xtr = np.array(_input_X)[tr, :]
        ytr = np.array(_input_Y)[tr]
        Xte = np.array(_input_X)[te, :]
        yte = np.array(_input_Y)[te]

        # Scale data
        scaler = preprocessing.StandardScaler()  # create scaler
        Xtr = scaler.fit_transform(Xtr)  # fit the scaler to the training data and transform training data
        Xte = scaler.transform(Xte)  # transform test data

        # Fit classifier
        classifier.fit(Xtr, ytr)

        # Predictions
        pred[te] = classifier.predict(Xte).reshape(yte.size,1)
    return pred

