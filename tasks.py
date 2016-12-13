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


#### idée 1 : RF ou SVM sur 4 classes (4 quartiles) puis faire une regression sur chaque classe
#### idée 2 : SVR, random forest for regression
#### Comparer avec les residus a chaque fois
#### Scale les feature continues

pd.set_option('display.width', 250)


input_train_sample = pd.read_csv('train.csv')

hour_filter=False
if hour_filter:
    input_train_sample = input_train_sample[input_train_sample.hr <= 5]

output_test_sample = pd.read_csv('test.csv')
my_sub = pd.read_csv('my_submission.csv')

cat_feat = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
non_cat_feat = ['temp', 'atemp', 'hum', 'windspeed']
var_Y = ['cnt']
features = cat_feat + non_cat_feat


_input_X = input_train_sample[features]
_categorical_input_X = _input_X[cat_feat]
non_cat_input_X = _input_X[non_cat_feat]

input_X = get_my_input(_input_X, cat_feat, non_cat_feat)
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

    #fig = FF.create_candlestick(data_plot.mean, data_plot.amin, data_plot.amax,
    #                            data_plot.percentile_25, data_plot.percentile_75,
    #                            dates=data_plot.index)
    #py.iplot(fig, validate=False)

    fig, ax1 = plt.subplots()
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    ax1.plot(data_plot.index.tolist(), data_plot['mean'], 'bo', label='mean')
    ax1.plot(data_plot.index.tolist(), data_plot['median'], 'b-', label='median')
    #ax1.plot(data_plot.index.tolist(), data_plot['amin'], 'r-', label='amin')
    #ax1.plot(data_plot.index.tolist(), data_plot['amax'], 'r-', label='amax')
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


# Divide in subtables
# # season

inputs_X = []
inputs_Y = []
for i in input_train_sample.season.unique():

    _input_ = input_train_sample[input_train_sample.season == i ]
    _input_X = _input_[features]
    input_X_ = get_my_input(_input_X, cat_feat, non_cat_feat)

    input_Y_ = _input_[var_Y]

    inputs_X.append(input_X_)
    inputs_Y.append(input_Y_)

# SVR
# # linear: \langle x, x'\rangle.
# # poly: (\gamma \langle x, x'\rangle + r)^d. d is specified by keyword degree, r by coef0.
# # rbf: \exp(-\gamma |x-x'|^2). \gamma is specified by keyword gamma, must be greater than 0.
# # sigmoid (\tanh(\gamma \langle x,x'\rangle + r)), where r is specified by coef0.
nb_folds = 5

season_predictions = {}
season_kernels = {}
season_C = {}

for i in range(4):
    svr_predictions = {}
    svr_ms_errors = {}
    svr_residuals = {}
    changing_C = [0.1, 0.5]# 1, 2, 2.5, 5, 10, 20, 50]
    kernels = ['rbf', 'linear']#, 'poly', 'sigmoid']

    for ker in kernels:
        svr_ms_errors[ker] = {}
        svr_residuals[ker] = {}
        svr_predictions[ker] = {}
        print("Estimating SVR cross validation with Kernel:{}".format(ker))
        for c in changing_C:
            print("Estimating SVR cross validation with C = {}".format(c))
            predictions, residuals, ms_error = svr(inputs_X[i], inputs_Y[i],
                                                   nb_folds=nb_folds, C=c,
                                                   kernel=ker, degree=3,
                                                   gamma='auto')

            print('ms_error of fitting:{}'.format(ms_error))
            svr_ms_errors[ker][str(c)] = float(ms_error)
            svr_residuals[ker][str(c)] = residuals
            svr_predictions[ker][str(c)] = predictions

    print('Min of ms_errors: {}'.format(np.min(svr_ms_errors)))

    mimimums = {}
    for key in svr_ms_errors.keys():
        d = svr_ms_errors[key]
        mimimums[key] = min(d.items(), key=lambda x: x[1])

    optimal_kernel = min(mimimums, key=mimimums.get)
    optimal_C = mimimums[optimal_kernel][0]

    season_kernels['season{}'.format(i + 1)] = optimal_kernel
    season_C['season{}'.format(i + 1)] = optimal_C
    season_predictions['season{}'.format(i + 1)] = svr_predictions[optimal_kernel][optimal_C]

optimal_preds = []
for key in season_predictions.keys():
    preds = np.transpose(season_predictions[key])[0].tolist()
    optimal_preds = optimal_preds + preds
optimal_preds = np.array(optimal_preds)

real_preds = []
for i in range(4):
    real_preds = real_preds + inputs_Y[i]['cnt'].tolist()
real_preds = np.array(real_preds)

optimal_res = optimal_preds - real_preds
optimal_mse = np.mean(optimal_res**2)

# min_residuals = svr_residuals[optimal_kernel][mimimums[optimal_kernel]]
# res_output = pd.concat([input_Y, min_residuals], axis=1)
# res_output.columns = [var_Y[0], 'residuals']
# plot_vars(res_output, var_Y[0], 'residuals')


# Random_Forest

rf_ms_errors = []
rf_residuals = []
n_estimators = [20, 50, 100, 200, 250, 500, 1000, 1500]
for estim in n_estimators:
    print("Estimating random forest cross validation with n_estimators = {}".format(estim))
    predictions, residuals, ms_error = rand_forest_reg(input_X, input_Y, nb_folds=nb_folds, n_estimators=estim)
    rf_ms_errors.append(ms_error)
    rf_residuals.append(residuals)
    print('ms_error of fitting:{}'.format(ms_error))

print('Min of ms_errors: {}'.format(np.min(rf_ms_errors)))
min_index = np.argmin(rf_ms_errors)
min_residuals = rf_residuals[min_index]

optimal_n_estim = rf_ms_errors[min_index]

res_output = pd.concat([input_Y, min_residuals], axis=1)
res_output.columns = [var_Y[0], 'residuals']
plot_vars(res_output, var_Y[0], 'residuals')

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
