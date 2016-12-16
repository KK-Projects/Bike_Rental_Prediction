import pandas as pd
import numpy as np

from utils import get_my_input, cat_var_to_dummies, cross_validate
from regression import lin_reg_, linear_reg
from features_description import plot_vars, divise_in_classes
from random_forest import rand_forest_reg
from knearest import k_nearest_neighbors
from sklearn.ensemble import RandomForestRegressor
from svm import svc, svr
import matplotlib.pyplot as plt
from datetime import datetime
from regularized_lin_regression import ridge
from pylab import savefig

pd.set_option('display.width', 250)


input_train_sample = pd.read_csv('train.csv')
input_train_sample["trimestre"] = input_train_sample.yr * 4 + input_train_sample.season

output_test_sample = pd.read_csv('test.csv')
output_test_sample["trimestre"] = output_test_sample.yr * 4 + output_test_sample.season
my_sub = pd.read_csv('my_submission.csv')

cat_feat = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'yr']#, 'trimestre']
non_cat_feat = ['temp', 'atemp', 'hum', 'windspeed']
var_Y = ['cnt']
features = cat_feat + non_cat_feat

import matplotlib.pyplot as plt
feat_ = 'windspeed'
plt.plot(input_train_sample[feat_], input_train_sample.cnt, 'go')
plt.axis([0, input_train_sample[feat_].max() + 0.1, 0, input_train_sample.cnt.max() + 0.1])
plt.title("Number of bikes rented")
plt.xlabel(feat_)
plt.ylabel('cnt')
plt.show()
savefig('Number of bikes rented to the {}.png'.format(feat_))

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
nb_folds = 5
bucks = {}
bucks['hr'] = [-1, 6, 9, 12, 16, 18, 20, 24]
bucks['mnth'] = [0, 3, 6, 12]
bucks['atemp'] = [-1., 0.17, 0.3, 0.58, 0.61, 0.71, 1]
bucks['hum'] = [-1., 0.2, 0.46, 0.58, 0.66, 0.74, 0.84, 0.91, 1]
bucks['windspeed'] = [-1., 0.3, 0.5, 1]

_input_X = input_train_sample[features]
_categorical_input_X = _input_X[cat_feat]
non_cat_input_X = _input_X[non_cat_feat]

input_X = get_my_input(_input_X, cat_feat, non_cat_feat, bucks=bucks, buckets=True, drop_feat=False)
categorical_input_X = cat_var_to_dummies(_categorical_input_X)
input_Y = input_train_sample[var_Y]

svr_predictions = []
svr_ms_errors = []
svr_residuals = []
svr_log_mse = []
gammas = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, "auto"]
kernels = ["linear", "rbf", "poly", "sigmoid"]
C_changing = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 12, 15, 20]
for gam in gammas:
    for ker in kernels:
        for c in C_changing:
            print("Estimating SVR cross validation with "
                  "{} as C, {} as kernel, {} gamma".format(c, ker, gam))
            predictions, residuals, ms_error = svr(input_X, input_Y,
                                                   nb_folds=nb_folds, C=c,
                                                   kernel=ker, degree=3, gamma=gam)

            svr_ms_errors.append(ms_error)
            svr_residuals.append(residuals)
            print('ms_error of fitting:{}'.format(ms_error))
            svr_predictions.append(predictions)

            _log_res = np.log(predictions + 1) - np.log(input_Y + 1)
            log_mse = np.sqrt(np.mean(_log_res ** 2))
            svr_log_mse.append(log_mse)
            print('final RMSLE for optimal n_estimators for each season = {}'.format(log_mse))

import matplotlib.pyplot as plt

plt.plot(C_changing, svr_log_mse, 'b-')
plt.title("Cross-validated score(RMSLE) for different values ofC")
plt.xlabel('C')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different values of C.png')

plt.plot(svr_log_mse, 'b-')
plt.title("Cross-validated score(RMSLE) for different max features functions")
plt.xlabel('Kernels (0 = Linear, 1 = rbf, 2 = poly, 3 = sigmoid)')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different Kernels.png')

gammas[-1] = -1
plt.plot(gammas, svr_log_mse, 'b-')
plt.title("Cross-validated score(RMSLE) for different values of gamma")
plt.xlabel('gamma (-1 = auto, i.e 1/n-features')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different values of gamma.png')

# Random_Forest 3 Params to cross validate

_input_X = input_train_sample[features]
_categorical_input_X = _input_X[cat_feat]
non_cat_input_X = _input_X[non_cat_feat]

input_X = get_my_input(_input_X, cat_feat, non_cat_feat, bucks=bucks, buckets=True, drop_feat=False)
categorical_input_X = cat_var_to_dummies(_categorical_input_X)
input_Y = input_train_sample[var_Y]

rf_predictions = []
rf_ms_errors = []
rf_residuals = []
rf_log_mse = []
max_depths = [None, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
max_feats = [None, "sqrt"]#, "log2"]
n_estimators = [1, 5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 750, 950, 1050]
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

            _log_res = np.log(predictions + 1) - np.log(input_Y + 1)
            log_mse = np.sqrt(np.mean(_log_res ** 2))
            rf_log_mse.append(log_mse)
            print('final RMSLE for optimal n_estimators for each season = {}'.format(log_mse))

print('Min of ms_errors: {}'.format(np.min(rf_ms_errors)))
min_index = np.argmin(rf_ms_errors)
min_residuals = rf_residuals[min_index]

rf_optimal_mse = rf_ms_errors[min_index]
optimal_preds = np.array(rf_predictions[min_index])

log_optimal_res = np.log(optimal_preds + 1) - np.log(input_Y + 1)
optimal_mse = np.sqrt(np.mean(log_optimal_res**2))
print('final RMSLE for optimal n_estimators for each season = {}'.format(optimal_mse))

import matplotlib.pyplot as plt
plt.plot(n_estimators, rf_log_mse_, 'b-')
plt.title("Cross-validated score(RMSLE) for different values of Number of Trees")
plt.xlabel('number of trees')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different values of Number of Trees.png')

plt.plot(rf_log_mse, 'b-')
plt.title("Cross-validated score(RMSLE) for different max features functions")
plt.xlabel('max_feat_function')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different max features functions.png')


plt.plot(max_depths , rf_log_mse, 'b-')
plt.title("Cross-validated score(RMSLE) for different values of max depth")
plt.xlabel('max_depth')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different values of max depth.png')


# res_output = pd.concat([input_Y, min_residuals], axis=1)
# res_output.columns = [var_Y[0], 'residuals']
# plot_vars(res_output, var_Y[0], 'residuals')

# K Nearest Neighbors

nb_folds = 5
knn_ms_errors = []
knn_residuals = []
knn_rmsle = []
knn_algorithm = ["ball_tree"]  # ["ball_tree", "kd_tree", "brute", "auto"]
knn_weights = ["uniform"]  # ["uniform", "distance"]
knn_neighbors = [5]
# [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
for algorithm in knn_algorithm:
    for weight in knn_weights:
        for neigh in knn_neighbors:
            print("Estimating KNN with algorithm = {}, weight = {}, neighbors = {}".format(algorithm, weight, neigh))
            predictions, residuals, ms_error, rmsle = k_nearest_neighbors(input_X, input_Y, nb_folds=nb_folds,
                                                                   n_neighbors=neigh, weights=weight,
                                                                   algo=algorithm)
            knn_ms_errors.append(ms_error)
            knn_residuals.append(residuals)
            knn_rmsle.append(rmsle)
            print('RMSLE of fitting:{}'.format(rmsle))

plt.plot(knn_neighbors, knn_rmsle, 'b-')
plt.title("Cross-validated score(RMSLE) for different values of Number of Neighbours")
plt.xlabel('number of Neighbours')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different values of Number of Neighbours.png')

plt.plot(knn_rmsle, 'ro')
plt.title("Cross-validated score(RMSLE) for different weights functions")
plt.xlabel('weights ( 0 : uniform, 1: distance)')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different weights functions.png')


plt.plot(knn_rmsle, 'b-')
plt.title("Cross-validated score(RMSLE) for different Algorithms")
plt.xlabel('algorithms ( 0 : ball_tree, 1: kd_tree, 2: brute, 3: auto)')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different Algorithms.png')

print('Min of RMSLE: {}'.format(np.min(knn_rmsle)))
knn_min_index = np.argmin(knn_rmsle)
knn_min_residuals = knn_residuals[knn_min_index]

knn_min_ms_error = knn_ms_errors[knn_min_index]

knn_res_output = pd.concat([input_Y, knn_min_residuals], axis=1)
knn_res_output.columns = [var_Y[0], 'residuals']
plot_vars(knn_res_output, var_Y[0], 'residuals')


# Ridge

nb_folds = 5
ridge_ms_errors = []
ridge_residuals = []
ridge_rmsle = []
ridge_alpha = [0.0001,0.001,0.01,0.1,1,10,100]
# [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
for a in ridge_alpha:
    print("Estimating Ridge with alpha = {}".format(a))
    predictions, residuals, ms_error, rmsle = ridge(input_X, input_Y, nb_folds=nb_folds, alpha=a)
    ridge_ms_errors.append(ms_error)
    ridge_residuals.append(residuals)
    ridge_rmsle.append(rmsle)
    print('RMSLE of fitting:{}'.format(rmsle))

plt.plot(ridge_alpha, ridge_rmsle, 'b-')
plt.title("Cross-validated score(RMSLE) for different values of alpha")
plt.xlabel('alpha')
plt.ylabel('rmsle')
plt.show()
savefig('Cross-validated score(RMSLE) for different values of alpha.png')

print('Min of RMSLE: {}'.format(np.min(ridge_rmsle)))
ridge_min_index = np.argmin(ridge_rmsle)
ridge_min_residuals = ridge_residuals[ridge_min_index]

ridge_min_ms_error = ridge_ms_errors[ridge_min_index]

ridge_res_output = pd.concat([input_Y, ridge_min_residuals], axis=1)
ridge_res_output.columns = [var_Y[0], 'residuals']
plot_vars(ridge_res_output, var_Y[0], 'residuals')







# Classification(input_train_sample, var_Y, categorical_input_X):
input_train_div = divise_in_classes(input_train_sample, var_Y)
class_input_Y = input_train_div[["category"]]
classification, conf_matrix = svc(categorical_input_X, class_input_Y,
                                  nb_folds=10, C=1.0,
                                  kernel='rbf', degree=3,
                                  gamma='auto')

# Linear Regression

print("Estimating linear reggression cross validation ")
predictions, residuals, ms_error = linear_reg(input_X, input_Y, nb_folds=nb_folds)
print('ms_error of fitting:{}'.format(ms_error))
_log_res = np.log(predictions + 1) - np.log(input_Y + 1)
log_mse = np.sqrt(np.mean(_log_res ** 2))
print('final RMSLE for optimal n_estimators for each season = {}'.format(log_mse))

# Get Output Data for Kaggle

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

optimal_n_estimators = 900
forest = RandomForestRegressor(n_estimators=optimal_n_estimators, max_depth=None, max_features="auto")
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
linear = LinearRegression()
ridge = RidgeCV(alphas=[1e-2, 0.1, 1.0, 10.0, 100.0])
from sklearn import svm
optimal_C = 22
optimal_kernel = 'poly'
optimal_gamma = 0.025
_svr = svm.SVR(C=optimal_C, kernel=optimal_kernel, gamma=optimal_gamma)
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=15, weights="uniform", algorithm="auto")


classifier = knn
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

_input_X = input_train_sample[features]
_categorical_input_X = _input_X[cat_feat]
non_cat_input_X = _input_X[non_cat_feat]
input_X = get_my_input(_input_X, cat_feat, non_cat_feat, bucks=bucks, buckets=True, drop_feat=False)

xtr = scaler.fit_transform(input_X)
_output_X = get_my_input(output_X, cat_feat, non_cat_feat, bucks=bucks, buckets=True, drop_feat=False)
xte = scaler.transform(_output_X)  # transform test data
# Fit classifier
classifier.fit(xtr, input_Y)
# Predictions

pred = classifier.predict(xte)
my_sub = pd.read_csv('my_submission.csv')
my_sub['Prediction'] = pred
my_sub.Prediction = my_sub.Prediction.astype(int)
my_sub.to_csv('sub_{}.csv'.format(classifier), index=False)