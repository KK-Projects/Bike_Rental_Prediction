
import pandas as pd
from sklearn.cross_validation import train_test_split


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