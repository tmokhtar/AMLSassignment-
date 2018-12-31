'''
Author: Tarek Mohtar
Date: 

'''

import sys
import os
import numpy as np
import random
from collections import Counter
import sklearn
import matplotlib.pyplot as plt

import amls_helper as helper

from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC


def preprocess_dataset(dataset):
    # converts face locations from a 2D array to 1D
    output = []
    random.shuffle(dataset)
    for file_name, features, label in dataset:
        features = features.flatten()
        output.append((file_name, features, label))
    return output

def create_train_test_split(dataset, split=0.8):
    # splits data up into training & testing and labels them accordingly
    dataset_size = len(dataset)
    train_size = int(split*dataset_size)
    train = dataset[:train_size]
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

def train_SVM(train_x, train_y, test_x, test_y, kernel='poly', degree=3, C=0.15, tol=1e-3):
    model = SVC(C=C, kernel=kernel, degree=degree, tol=tol)
    print('Fitting SVM Model...')
    model.fit(train_x, train_y)
    print('Getting Predictions...')
    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)
    train_accuracy = accuracy_score(train_y, train_predictions)
    print('Train Set Accuracy: {}'.format(train_accuracy))
    test_accuracy = accuracy_score(test_y, test_predictions)
    print('Test Set Accuracy: {}'.format(test_accuracy))
    print('Classification Report')
    report = classification_report(test_y, test_predictions)
    print(report)
    return test_accuracy, test_predictions


def train_NN(train_x, train_y, test_x, test_y):
    pass


def main():
    images_dir = 'dataset'
    labels_file = 'attribute_list.csv'
    # method 1
    dataset = helper.extract_features_labels(images_dir, labels_file, task=1, method='landmark_features', model='hog', sample_times=1)
    # method 2
    #dataset = helper.extract_features_labels(images_dir, labels_file, task=1, method='face_locations', model='hog', sample_times=1)
    #dataset = helper.extract_features_labels(images_dir, labels_file, task=1, method='face_locations', model='cnn', sample_times=2)

    # get dataset distribution of given task
    task_distribution = Counter(d[-1] for d in dataset)

    print('Task Distribution: {}'.format(task_distribution))

    # get true distribution of given task
    attributes = [line.strip().split(',') for line in open(labels_file)][2:]
    noisy_images = set([row[0] for row in attributes if set(row[1:]) == {'-1'}])
    true_distribution = Counter(row[3] for row in attributes if row[0] not in noisy_images)
    print('True Distribution: {}'.format(true_distribution))
    preprocessed_dataset = preprocess_dataset(dataset)
    train_data, train_labels, test_data, test_labels, test_files = create_train_test_split(preprocessed_dataset)

    test_accuracy, test_predictions = train_SVM(train_data, train_labels, test_data, test_labels)
    create_test_file(test_accuracy, test_predictions, test_files, task=1)
    
if __name__ == '__main__':
    main()
