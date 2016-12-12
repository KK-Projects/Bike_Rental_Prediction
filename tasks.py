import pandas as pd
import numpy as np

from utils import get_my_input, cat_var_to_dummies, cross_validate
#from regression import lin_reg_
from features_description import plot_vars #, divise_in_classes
from random_forest import rand_forest_reg
from sklearn.ensemble import RandomForestRegressor
#from svm import svc
import matplotlib.pyplot as plt
from datetime import datetime


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
        'average_cnt': 'mean'
    }
}

input_train_sample['day'] = input_train_sample['dteday'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').day)

for x_feat in ['mnth', 'hr', 'day']:

    y_feat = 'cnt'
    aggregations = {
        y_feat: {
            'total_cnt': 'sum',
            'average_cnt': 'mean'
        }
    }
    data_plot = input_train_sample.groupby([x_feat]).agg(aggregations)[y_feat]
    fig, ax1 = plt.subplots()
    ax1.plot(data_plot.index.tolist(), data_plot['average_cnt'], 'b-')
    ax1.set_xlabel(x_feat)
    ax1.set_ylabel('average_cnt', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2 = ax1.twinx()
    ax2.plot(data_plot.index.tolist(), data_plot['total_cnt'], 'ro')
    ax2.set_ylabel('total_cnt', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.show()

# Classification
input_train_div = divise_in_classes(input_train_sample, var_Y)
class_input_Y = input_train_div[["category"]]
classification, conf_matrix = svc(categorical_input_X, class_input_Y, test_size=0.2, number_sets=10)

# Linear Regression
reg, residuals, input_data = lin_reg_(input_X, input_Y, test_size=0.2, number_sets=10)
res_output = pd.concat([input_data['Y_test'], residuals], axis=1)
res_output.columns = [var_Y[0], 'residuals']
plot_vars(res_output, var_Y[0], 'residuals')

# Random Forest
nb_folds = 10
n_estimators = [20, 50, 100, 150, 200, 250]
ms_errors = []
residuals = []
for estim in n_estimators:
    forest, residuals, ms_error = rand_forest_reg(input_X, input_Y, nb_folds=nb_folds, n_estimators=estim)
    ms_errors.append(ms_error)
    residuals.append(residuals)

print('Mean of ms_errors: {}'.format(np.min(ms_errors)))
min_index = np.argmin(ms_errors)
min_residuals = residuals[min_index]

res_output = pd.concat([input_Y, min_residuals], axis=1)
res_output.columns = [var_Y[0], 'residuals']
plot_vars(res_output, var_Y[0], 'residuals')

# Random Forest test Gridsearch




