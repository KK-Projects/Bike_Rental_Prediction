import pandas as pd
import numpy as np

from utils import get_my_input, cat_var_to_dummies, cross_validate
#from regression import lin_reg_
from features_description import plot_vars, divise_in_classes
from random_forest import rand_forest_reg
from knearest import k_nearest_neighbors
from sklearn.ensemble import RandomForestRegressor
from svm import svc, svr
import matplotlib.pyplot as plt
from datetime import datetime

from pylab import savefig

pd.set_option('display.width', 250)


input_train_sample = pd.read_csv('train.csv')

output_test_sample = pd.read_csv('test.csv')
my_sub = pd.read_csv('my_submission.csv')

cat_feat = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'yr']
non_cat_feat = ['temp', 'atemp', 'hum', 'windspeed']
var_Y = ['cnt']
features = cat_feat + non_cat_feat


_input_X = input_train_sample[features]
_categorical_input_X = _input_X[cat_feat]
non_cat_input_X = _input_X[non_cat_feat]

input_X = get_my_input(_input_X, cat_feat, non_cat_feat, buckets=False)
categorical_input_X = cat_var_to_dummies(_categorical_input_X)
input_Y = input_train_sample[var_Y]

output_X = output_test_sample[features]

# Describing y training variable
input_Y.describe()
features_uniques = {}
for feat in cat_feat:
    features_uniques[feat] = _input_X[feat].unique().tolist()

aggregations = {
    'cnt': {
        'total_cnt': 'sum',
        'average_cnt': 'mean',
        'max_cnt': 'max',
        'min_cnt': 'min',
        '25%_quantile_cnt': 'quantile'
    }
}


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')

input_train_sample['day'] = input_train_sample['dteday'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').day)
aggregations_ = [np.sum, np.mean, np.median, np.min, np.max, percentile(25), percentile(75)]
for x_feat in features_uniques.keys():

    y_feat = 'cnt'

    data_plot = input_train_sample.groupby([x_feat])[y_feat].agg(aggregations_)

    fig, ax1 = plt.subplots()
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax1.plot(data_plot.index.tolist(), data_plot['mean'], 'bo', label='mean')
    ax1.plot(data_plot.index.tolist(), data_plot['median'], 'b-', label='median')
    ax1.plot(data_plot.index.tolist(), data_plot['percentile_25'], 'g-', label='percentile_25')
    ax1.plot(data_plot.index.tolist(), data_plot['percentile_75'], 'g-', label='percentile_75')
    plt.legend(loc="upper center", bbox_to_anchor=[0.5, 1.12],
               ncol=3, shadow=True, title="Legend", fancybox=True,
               prop=fontP)
    ax1.get_legend().get_title().set_color("red")

    ax1.set_xlabel(x_feat)
    ax1.set_ylabel('cnt', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2 = ax1.twinx()
    ax2.plot(data_plot.index.tolist(), data_plot['sum'], 'kx')
    ax2.set_ylabel('total_cnt', color='k')
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    plt.show()
    savefig('cnt descriptions according to the {}.png'.format(x_feat))

# SVR
# # linear: \langle x, x'\rangle.
# # poly: (\gamma \langle x, x'\rangle + r)^d. d is specified by keyword degree, r by coef0.
# # rbf: \exp(-\gamma |x-x'|^2). \gamma is specified by keyword gamma, must be greater than 0.
# # sigmoid (\tanh(\gamma \langle x,x'\rangle + r)), where r is specified by coef0.
nb_folds = 10

# Random_Forest 3 Params to cross validate

bucks = {}
bucks['hr'] = [-1, 6, 9, 12, 16, 18, 20, 24]
bucks['mnth'] = [0, 3, 6, 12]
#bucks['atemp'] = [-1., 0.2, 0.58, 0.75, 1]
#bucks['hum'] = [-1., 0.2, 0.8, 1]
bucks['atemp'] = [-1., 0.17, 0.3, 0.58, 0.61, 0.71, 1]
bucks['hum'] = [-1., 0.15, 0.46, 0.58, 0.66, 0.74, 0.84, 0.91, 1]

_input_X = input_train_sample[features]
_categorical_input_X = _input_X[cat_feat]
non_cat_input_X = _input_X[non_cat_feat]

input_X = get_my_input(_input_X, cat_feat, non_cat_feat, bucks=bucks, buckets=True, drop_feat=False)
categorical_input_X = cat_var_to_dummies(_categorical_input_X)
input_Y = input_train_sample[var_Y]

rf_predictions = []
rf_ms_errors = []
rf_residuals = []
max_depths = [None, 300, 500, 800, 900, 1000, 1200, 1500, 2000, 5000]
max_feats = ["auto"]# "sqrt", "log2"]
n_estimators = [50]#, 300, 400, 500, 750, 850, 950, 1050, 1200, 1500]
for depth in max_depths:
    for max_f in max_feats:
        for estim in n_estimators:
            print("Estimating random forest cross validation with "
                  "{} trees, {} as max_depth, {} max_feats".format(estim, depth, max_f))
            predictions, residuals, ms_error = rand_forest_reg(input_X, input_Y,
                                                               max_features=max_f, nb_folds=nb_folds,
                                                               max_depth=depth, n_estimators=estim)
            rf_ms_errors.append(ms_error)
            rf_residuals.append(residuals)
            print('ms_error of fitting:{}'.format(ms_error))
            rf_predictions.append(predictions)

print('Min of ms_errors: {}'.format(np.min(rf_ms_errors)))
min_index = np.argmin(rf_ms_errors)
min_residuals = rf_residuals[min_index]

rf_optimal_mse = rf_ms_errors[min_index]
optimal_preds = np.array(rf_predictions[min_index])

optimal_res = optimal_preds - input_Y
optimal_mse = np.mean(optimal_res**2)
print('final mse for optimal n_estimators for each season = {}'.format(optimal_mse))

# res_output = pd.concat([input_Y, min_residuals], axis=1)
# res_output.columns = [var_Y[0], 'residuals']
# plot_vars(res_output, var_Y[0], 'residuals')

# K Nearest Neighbors

nb_folds = 10
knn_ms_errors = []
knn_residuals = []
knn_algorithm = ["auto"]  # ["ball_tree", "kd_tree", "brute", "auto"]
knn_weights = ["uniform"]  # ["uniform", "distance"]
knn_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
# [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
for algorithm in knn_algorithm:
    for weight in knn_weights:
        for neigh in knn_neighbors:
            print("Estimating KNN with algorithm = {}, weight = {}, neighbors = {}".format(algorithm, weight, neigh))
            predictions, residuals, ms_error = k_nearest_neighbors(input_X, input_Y, nb_folds=nb_folds,
                                                                   n_neighbors=neigh, weights=weight,
                                                                   algo=algorithm)
            knn_ms_errors.append(ms_error)
            knn_residuals.append(residuals)
            print('ms_error of fitting:{}'.format(ms_error))

print('Min of ms_errors: {}'.format(np.min(knn_ms_errors)))
knn_min_index = np.argmin(knn_ms_errors)
knn_min_residuals = knn_residuals[knn_min_index]

knn_min_ms_error = knn_ms_errors[knn_min_index]

knn_res_output = pd.concat([input_Y, knn_min_residuals], axis=1)
knn_res_output.columns = [var_Y[0], 'residuals']
plot_vars(knn_res_output, var_Y[0], 'residuals')

# K Nearest Neighbors

nb_folds = 5
knn_ms_errors = []
knn_residuals = []
knn_algorithm = ["auto"]  # ["ball_tree", "kd_tree", "brute", "auto"]
knn_weights = ["uniform"]  # ["uniform", "distance"]
knn_neighbors = [1, 3, 5, 7]
# [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
for algorithm in knn_algorithm:
    for weight in knn_weights:
        for neigh in knn_neighbors:
            print("Estimating KNN with algorithm = {}, weight = {}, neighbors = {}".format(algorithm, weight, neigh))
            predictions, residuals, ms_error = k_nearest_neighbors(input_X, input_Y, nb_folds=nb_folds,
                                                                   n_neighbors=neigh, weights=weight,
                                                                   algo=algorithm)
            knn_ms_errors.append(ms_error)
            knn_residuals.append(residuals)
            print('ms_error of fitting:{}'.format(ms_error))

print('Min of ms_errors: {}'.format(np.min(knn_ms_errors)))
knn_min_index = np.argmin(knn_ms_errors)
knn_min_residuals = knn_residuals[knn_min_index]

knn_min_ms_error = knn_ms_errors[knn_min_index]

knn_res_output = pd.concat([input_Y, knn_min_residuals], axis=1)
knn_res_output.columns = [var_Y[0], 'residuals']
plot_vars(knn_res_output, var_Y[0], 'residuals')


# Classification(input_train_sample, var_Y, categorical_input_X):
input_train_div = divise_in_classes(input_train_sample, var_Y)
class_input_Y = input_train_div[["category"]]
classification, conf_matrix = svc(categorical_input_X, class_input_Y,
                                  nb_folds=10, C=1.0,
                                  kernel='rbf', degree=3,
                                  gamma='auto')

# Linear Regression
reg, residuals, input_data = lin_reg_(input_X, input_Y, test_size=0.2, number_sets=10)
res_output = pd.concat([input_data['Y_test'], residuals], axis=1)
res_output.columns = [var_Y[0], 'residuals']
plot_vars(res_output, var_Y[0], 'residuals')


# Get Output Data for Kaggle

optimal_n_estimators = 900
forest = RandomForestRegressor(n_estimators=optimal_n_estimators, max_depth=None, max_features="sqrt")

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

from sklearn import tree
clf = tree.DecisionTreeClassifier()
import pydotplus
from IPython.display import Image, display
from sklearn.datasets import load_iris
iris = load_iris()
data = input_train_sample[['temp']+['cnt']]
_data = np.array([data['temp']]).transpose()
target = np.array(data['cnt'])
clf = clf.fit(_data, target)
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
img = Image(graph.create_png())

bucks = {}
bucks['hr'] = [-1, 6, 9, 12, 16, 18, 20, 24]
bucks['mnth'] = [0, 6, 12]
bucks['atemp'] = [-1., 0.2, 0.58, 0.75, 1]
bucks['hum'] = [-1., 0.2, 0.8, 1]
_input_X = input_train_sample[features]
_categorical_input_X = _input_X[cat_feat]
non_cat_input_X = _input_X[non_cat_feat]
input_X = get_my_input(_input_X, cat_feat, non_cat_feat, bucks=bucks, buckets=True, drop_feat=False)

xtr = scaler.fit_transform(input_X)
_output_X = get_my_input(output_X, cat_feat, non_cat_feat, bucks=bucks, buckets=True, drop_feat=False)
xte = scaler.transform(_output_X)  # transform test data
# Fit classifier
forest.fit(xtr, input_Y)
# Predictions

pred = forest.predict(xte)
my_sub = pd.read_csv('my_submission.csv')
my_sub['Prediction'] = pred
my_sub.Prediction = my_sub.Prediction.astype(int)
my_sub.to_csv('sub_3.csv', index=False)