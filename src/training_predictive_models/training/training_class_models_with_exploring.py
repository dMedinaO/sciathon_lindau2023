import sys

#for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

#for prepare dataset
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random

#for check overfitting
from sklearn.model_selection import cross_validate
import numpy as np

#for training models
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#function to obtain metrics using the testing dataset
def get_performances(description, predict_label, real_label):
    accuracy_value = accuracy_score(real_label, predict_label)
    f1_score_value = f1_score(real_label, predict_label, average='weighted')
    precision_values = precision_score(real_label, predict_label, average='weighted')
    recall_values = recall_score(real_label, predict_label, average='weighted')

    row = [description, accuracy_value, f1_score_value, precision_values, recall_values]
    return row

#function to process average performance in cross val training process
def process_performance_cross_val(performances, keys):
    
    row_response = []
    for i in range(len(keys)):
        value = np.mean(performances[keys[i]])
        row_response.append(value)
    return row_response

#function to train a predictive model
def training_process(model, X_train, y_train, X_test, y_test, scores, cv_value, description, keys):
    print("Train model with cross validation")
    model.fit(X_train, y_train)
    response_cv = cross_validate(model, X_train, y_train, cv=cv_value, scoring=scores)
    performances_cv = process_performance_cross_val(response_cv, keys)

    print("Predict responses and make evaluation")
    responses_prediction = clf.predict(X_test)
    response = get_performances(description, responses_prediction, y_test)
    response = response + performances_cv
    return response

#function to create a balanced dataset class
def create_balanced_dataset(dataset, column_response):
    print("Split dataset")
    positive_class = dataset.loc[dataset[column_response] == True]
    negative_class = dataset.loc[dataset[column_response] == False]

    #process the negative class 
    index_values = [index for index in negative_class.index]
    random.shuffle(index_values)
    index_values = index_values[:len(positive_class)] # balanced with the lenght of positive values

    negative_class['index'] = [index for index in negative_class.index]
    negative_class['is_in_select'] = negative_class['index'].isin(index_values)
    negative_class_filter = negative_class.loc[negative_class['is_in_select'] == True]
    negative_class_filter = negative_class_filter.drop(columns=['index', 'is_in_select', 'response_activity'])
    negative_class_filter = negative_class_filter.reset_index()

    #remove column with activity for positive class
    positive_class = positive_class.drop(columns=['response_activity'])
    positive_class = positive_class.reset_index()

    #create column response
    positive_values = [1 for i in range(len(positive_class))]
    negative_values = [0 for i in range(len(negative_class_filter))]

    #concat the dataset
    full_df = pd.concat([positive_class, negative_class_filter], axis=0)
    full_df = full_df.drop(columns=['index'])
    responses = positive_values + negative_values

    return full_df, responses

print("Read datasets")
dataset = pd.read_csv(sys.argv[1])
name_export = sys.argv[2]
scale_ds = int(sys.argv[3])

k_fold_value = int(sys.argv[4])

iteration = int(sys.argv[5])

#define the type of metrics
scoring = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted', 'test_precision_weighted', 'test_accuracy']

if scale_ds == 1:
    responses = dataset['response_activity'].astype(int)#cast a int for class
    dataset = dataset.drop(columns=['response_activity'])#remove response column

    print("Scaling dataset")
    scaler = preprocessing.RobustScaler()
    scaler.fit(dataset)
    dataset_scaler = scaler.transform(dataset)
    dataset = pd.DataFrame(dataset_scaler, columns=dataset.columns)
    dataset['response_activity'] = responses

print("Starting iteration process exploring")
print("Exploring iteration: ", iteration)

print("Creating balanced dataset")
dataset, responses = create_balanced_dataset(dataset, 'response_activity')

print("Preprocessing")
X_train, X_test, y_train, y_test = train_test_split(dataset, responses, test_size=0.2, random_state=42)

print("Exploring Training predictive models")
matrix_data = []

print("Exploring SVC")
clf = SVC()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "SVC", keys)
response.insert(0, iteration)
matrix_data.append(response)

print("Exploring KNN")
clf = KNeighborsClassifier()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "KNN", keys)
response.insert(0, iteration)
matrix_data.append(response)
    
print("Exploring GausianNB")
clf = GaussianNB()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "GausianNB", keys)
response.insert(0, iteration)
matrix_data.append(response)

print("Exploring decision tree")
clf = DecisionTreeClassifier()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "DT", keys)
response.insert(0, iteration)
matrix_data.append(response)
        
print("Exploring bagging method based DT")
clf = BaggingClassifier(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "bagging", keys)
response.insert(0, iteration)
matrix_data.append(response)
    

print("Exploring RF")
clf = RandomForestClassifier(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "RF", keys)
response.insert(0, iteration)
matrix_data.append(response)
        

print("Exploring Adaboost")
clf = AdaBoostClassifier()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "Adaboost", keys)
response.insert(0, iteration)
matrix_data.append(response)
    

print("Exploring GradientTreeBoost")
clf = GradientBoostingClassifier()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "GradientBoostingClassifier", keys)
response.insert(0, iteration)
matrix_data.append(response)


df_export = pd.DataFrame(matrix_data, columns=['iteration', 'description', 'test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'fit_time', 'score_time', 'train_f1_weighted', 'train_recall_weighted', 'train_precision_weighted', 'train_accuracy'])
df_export.to_csv(name_export, index=False)