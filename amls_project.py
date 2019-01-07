'''
Author: Tarek Mokhtar
Date: 6/1/19
'''

import sys
import os
import numpy as np
import random
from collections import Counter
import sklearn

import amls_helper as helper

from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import GridSearchCV

def preprocess_dataset(dataset):
    # converts face locations from a 2D array to 1D
    output = []
    random.shuffle(dataset)
    for file_name, features, label in dataset:
        features = features.flatten()
        output.append((file_name, features, label))
    return output

def create_train_test_split(dataset, split=0.8):
    # splits data up into training, validation  & testing and labels them accordingly
    dataset_size = len(dataset)
    train_size = int(split*dataset_size)
    valid_size = int(0.2*train_size)
    train_set = dataset[:train_size]
    train = train_set[:-valid_size]
    valid = train_set[-valid_size:]
    test = dataset[train_size:]
    train_data = np.array([t[1] for t in train])
    train_labels = np.array([t[2] for t in train])
    test_data = np.array([t[1] for t in test])
    test_labels = np.array([t[2] for t in test])
    test_files = [t[0] for t in test]
    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)
    print('Train Set Distribution: {}'.format(train_dist))
    print('Test Set Distribution: {}'.format(test_dist))
    return train_data, train_labels, test_data, test_labels, test_files

def create_test_file(test_accuracy, test_predictions, test_files, task=1):
    save_path = 'task_'+str(task)+'.csv'
    with open(save_path, 'w') as f:
        f.write('{}\n'.format(test_accuracy))
        for test_file, test_pred in zip(test_files, test_predictions):
            f.write('{0},{1}\n'.format(test_file, test_pred))

def train(train_x, train_y, test_x, test_y, algo, hyperparams, cv=3):
    if algo == 'SVM':
        model = GridSearchCV(SVC(), hyperparams, cv=cv)
        #model = SVC(C=C, kernel=kernel, degree=degree, tol=tol)

    elif algo == 'RF':
        model = GridSearchCV(RF(), hyperparams, cv=cv)
        #model = RF(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    print('Fitting Model and Tuning Hyperparameters with GridSearch using {}-fold cross-validation...'.format(cv))
    model.fit(train_x, train_y)

    best_params = model.best_params_
    print('Best Parameters Found: {}'.format(best_params))

    best_score = model.best_score_
    print('Mean cross-validated score of the best_estimator: {}'.format(best_score))
    
    print('Getting Predictions...')
    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)

    train_accuracy = accuracy_score(train_y, train_predictions)
    print('Train Set Accuracy: {}'.format(train_accuracy))

    test_accuracy = accuracy_score(test_y, test_predictions)
    print('Test Set Accuracy: {}'.format(test_accuracy))
    
    print('Test Set Classification Report')
    test_report = classification_report(test_y, test_predictions)
    print(test_report)
    return test_accuracy, test_predictions

def get_params(task, algo):
    #parameters for every algo and task
    hyperparams = {}
    if algo == 'SVM':
        hyperparams['kernel'] = ['linear','poly','rbf']
        hyperparams['tol'] = [1e-1,1e-2,1e-3,1e-4]
        hyperparams['degree'] = [1,2,3,4]
        if task in {1,3}:
            hyperparams['C'] = [0.16,0.17,0.18]
        elif task == 2:
            hyperparams['C'] = [0.15,0.16,0.17]
        elif task == 4:
            hyperparams['C'] = [0.14,0.15,0.16]
        elif task == 5:
            hyperparams['C'] = [0.23,0.24,0.25]
    elif algo == 'RF':
        hyperparams['min_samples_split'] = [2,3,4,5]
        hyperparams['min_samples_leaf'] = [1,2,3]
        if task == 1:
            hyperparams['n_estimators'] = [9,10,11]
        elif task == 2:
            hyperparams['n_estimators'] = [11,12,13]
        elif task in {3,4,5}:
            hyperparams['n_estimators'] = [10,11,12]
    return hyperparams

def main():
    images_dir = 'dataset'
    labels_file = 'attribute_list.csv'
    #sample_size = 1000
    task = int(sys.argv[1])
    algo = sys.argv[2].upper()
    cv = 3
    #method='landmark_features'
    method='face_locations'

    hyperparams = get_params(task, algo)

    dataset, noisy_detected = helper.extract_features_labels(images_dir, labels_file, task=task, method=method, model='hog', sample_times=2)

    # get dataset distribution of given task
    task_distribution = Counter(d[-1] for d in dataset)
    print('Task Distribution After Filtering Noisy Images: {}'.format(task_distribution))
    
    # get true distribution of given task
    attributes = [line.strip().split(',') for line in open(labels_file)][2:] #[:sample_size]
    noisy_images = set([row[0] for row in attributes if set(row[1:]) == {'-1'}])
    true_distribution = Counter(row[3] for row in attributes if row[0] not in noisy_images)
    print('True Distribution: {}'.format(true_distribution))

    total_noisy_detected = len(noisy_detected)
    total_noisy = len(noisy_images)
    noisy_intersection = noisy_detected.intersection(noisy_images)
    noisy_precision = len(noisy_intersection) / float(total_noisy_detected)
    noisy_recall = len(noisy_intersection) / float(total_noisy)
    print('Noisy Images Detection Precision: {}'.format(noisy_precision))
    print('Noisy Images Detection Recall: {}'.format(noisy_recall))
    
    preprocessed_dataset = preprocess_dataset(dataset)
    train_data, train_labels, test_data, test_labels, test_files = create_train_test_split(preprocessed_dataset)

    test_accuracy, test_predictions = train(train_data, train_labels, test_data, test_labels, algo, hyperparams)
    create_test_file(test_accuracy, test_predictions, test_files, task=task)
    
if __name__ == '__main__':
    main()
