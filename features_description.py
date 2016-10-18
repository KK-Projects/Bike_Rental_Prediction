import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def divise_in_classes(input_train_sample, var_Y):

    var_y = var_Y[0]
    description = input_train_sample[var_y].describe()
    print(description)
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    bins_labels = ['first_quartile', 'second_quartile', 'third_quartile', 'fourth_quartile']
    input_train_sample['category'] = pd.qcut(input_train_sample[var_y], bins, bins_labels)

    return input_train_sample


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


def get_pca_infos(input_X):

    X = input_X
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components='mle')
    pca.fit(X_std)
    e_values = pd.Series(pca.explained_variance_ratio_ * 100)

    coef = np.transpose(pca.components_)
    cols = ['PC-{}'.format(x) for x in range(len(e_values))]
    pc_infos = pd.DataFrame(coef, columns=cols, index=X.columns)

    return e_values, pc_infos


def plot_pca_pcs(input_X):

    e_values, pc_infos = get_pca_infos(input_X)
    e_values.plot(kind='bar', title="EigenValues")
    plt.ylabel('% of information')
    plt.show()


def plot_circleOfCorrelations(input_X):

    e_values, pc_infos = get_pca_infos(input_X)
    plt.Circle((0, 0), radius=10, color='g', fill=False)
    circle1 = plt.Circle((0, 0), radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(pc_infos["PC-0"])):
        x = pc_infos["PC-0"][idx]
        y = pc_infos["PC-1"][idx]
        plt.plot([0.0, x], [0.0, y], 'k-')
        plt.plot(x, y, 'rx')
        plt.annotate(pc_infos.index[idx], xy=(x, y))
    plt.xlabel("PC-0 (%s%%)" % str(e_values[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(e_values[1])[:4].lstrip("0."))
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.title("Circle of Correlations")
    plt.show()