from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
from operator import itemgetter

data_set=load_iris()
X=data_set.data
y=data_set.target
attribute_mean=X.mean(axis=0)
X_d=np.array(X>attribute_mean,dtype='int')

def train(X, y_true, feature):#feature:0,1,2,3
    n_samples, n_features = X.shape
    values = set(X[:, feature])
    predictors = dict()
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error

def train_feature_value(X, y_true, feature, value):
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_true):#sample是每一行X，y是每一行X对应的y_true
        if sample[feature] == value:#sample[feature]是一行里的某一列(feature)值
            class_counts[y] += 1
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    n_samples = X.shape[1]
    error = sum([class_count for class_value, class_count in class_counts.items()
                 if class_value != most_frequent_class])
    return most_frequent_class, error


random_state=4
X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=random_state)
all_predictors = {variable: train(X_train, y_train, variable) for variable in range(X_train.shape[1])}
errors = {variable: error for variable, (mapping, error) in all_predictors.items()}
best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0] #找到最佳特征feather
print("The best model is based on variable {0} and has error {1:.2f}".format(best_variable, best_error))
model = {'variable': best_variable,'predictor': all_predictors[best_variable][0]}
print(model)


def predict(X_test, model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted

y_predicted = predict(X_test, model)




