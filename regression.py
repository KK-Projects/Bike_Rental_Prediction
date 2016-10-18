
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import subdivise_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

input_train_sample = pd.read_csv('train.csv')
output_test_sample = pd.read_csv('test.csv')
my_sub = pd.read_csv('my_submission.csv')

cat_feat = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
non_cat_feat = ['temp', 'atemp', 'hum', 'windspeed']
features = cat_feat + non_cat_feat

_input_X = input_train_sample[features]
categorical_input_X = _input_X[cat_feat]
non_cat_input_X = _input_X[non_cat_feat]


# Converting Categorical Variables to dummies
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


input_X = get_my_input(_input_X, cat_feat, non_cat_feat)
input_Y = input_train_sample[['cnt']]
output_X = output_test_sample[features]

test_size = 0.2
number_sets = 10


def lin_reg_(input_X, input_Y, test_size, number_sets):

    input_X_std = pd.DataFrame(StandardScaler().fit_transform(input_X))
    input_X_std.columns = input_X.columns
    data_subdivised = subdivise_data(input_X_std, input_Y, test_size, number_sets)

    reg = LinearRegression()

    for set in data_subdivised.keys():

        print('Fitting the regression for the {} out of {}'.format(set, number_sets))
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