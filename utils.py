
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


def plot_vars(input_train_sample, feature_1, feature_2):
    """
    From a dataframe input_train_sample it plots 2 variables choses
    :param input_train_sample: Dataframe
    :param feature_1: String - Name of the feature 1 to plot as x
    :param feature_2: String - Name of the feature 2 to plot as y
    :return: plot
    """
    xlabel = feature_1
    ylabel = feature_2
    x = input_train_sample[xlabel]
    y = input_train_sample[ylabel]
    plt.plot(x, y, 'o')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


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


