import pandas as pd

from utils import get_my_input, cat_var_to_dummies
from regression import lin_reg_
from features_description import plot_vars, divise_in_classes
from random_forest import rand_forest_reg
from svm import svc

#### idée 1 : RF ou SVM sur 4 classes (4 quartiles) puis faire une regression sur chaque classe
#### idée 2 : SVR, random forest for regression
#### Comparer avec les residus a chaque fois

input_train_sample = pd.read_csv('train.csv')
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

# describing y training variable
input_Y.describe()

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
forest, residuals, input_data = rand_forest_reg(input_X, input_Y, test_size=0.2, number_sets=10, n_estimators=300)
residuals = pd.DataFrame(residuals)
residuals.columns = ['residuals']
res_output = pd.concat([input_data['Y_test'], residuals], axis=1)
res_output.columns = [var_Y[0], 'residuals']
plot_vars(res_output, var_Y[0], 'residuals')
