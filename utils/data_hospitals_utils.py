import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import utils.parameters as params
FIXED_PARAMETERS = params.load_parameters()
os.environ['PYTHONHASHSEED']=str(int(FIXED_PARAMETERS['version'][1:]))
np.random.seed(int(FIXED_PARAMETERS['version'][1:]))
rn.seed(int(FIXED_PARAMETERS['version'][1:]))
tf.set_random_seed(int(FIXED_PARAMETERS['version'][1:]))

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.utils import to_categorical
from sklearn.cluster import KMeans
import pandas as pd


def _load_processed_dataset(FIXED_PARAMETERS, dataset, merge_bool=True, data_string = ''):
    X = np.load(FIXED_PARAMETERS['data_path'] + '/X2' + '_' + dataset + '.npy')
    y = np.load(FIXED_PARAMETERS['data_path'] + '/Y_CPC1_2_' + dataset + '.npy')

    # if data_string == 'Coronary_Angiography':
    #     y = np.load(FIXED_PARAMETERS['data_path'] + '/Y_' + data_string + '_' + dataset + '.npy')
    #
    # if (data_string == 'Cardiac_Stent') or (data_string == 'CABG'):
    #     y = np.load(FIXED_PARAMETERS['data_path'] + '/Y_' + data_string + '_' + dataset + '.npy')

    if merge_bool:
        y[y == 2] = 1
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=0, prob_duplicate=0.0):
    train_labels, train_counts = np.unique(y, return_counts=True)
    X_train, Y_train = [], []
    X_val, Y_val = [], []

    for label, max_cnt in zip(train_labels, train_counts):
        samples = X[y == label, :]
        try:
            train_samples, val_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
        except:
            val_samples = samples ## Error handling sends all samples to the validation set
            train_samples = np.array([])

        for i in range(len(train_samples)):
            X_train.append(train_samples[i])
            Y_train.append(label)
            if (rn.random() < prob_duplicate) and (label == 0):
                X_train.append(train_samples[i])
                Y_train.append(label)

        for i in range(len(val_samples)):
            X_val.append(val_samples[i])
            Y_val.append(label)

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    Y_train = np.asarray(Y_train)
    Y_val = np.asarray(Y_val)

    return X_train, X_val, Y_train, Y_val

def prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list, print_bool=False, data_string = '', hospital_list=[]):
    X_train, y_train, X_val, y_val, X_test, y_test, X_sen, y_sen, metadata = {},{},{},{},{},{},{},{}, {}

    for a in FIXED_PARAMETERS[agency_list].split(' '):
        X, y = _load_processed_dataset(FIXED_PARAMETERS, a, data_string=data_string)
        X, X_sen[a], y, y_sen[a] = train_test_split(X, y, test_size=0.15, random_state=0)
        X_train[a], X_test[a], y_train[a], y_test[a] = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train[a], X_val[a], y_train[a], y_val[a] = train_val_split(X_train[a], y_train[a], test_size=0.2, random_state=0)

    X_train = np.concatenate([X_train[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X_val = np.concatenate([X_val[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X_test = np.concatenate([X_test[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X_sen = np.concatenate([X_sen[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)

    y_train = np.concatenate([y_train[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    y_val = np.concatenate([y_val[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    y_test = np.concatenate([y_test[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    y_sen = np.concatenate([y_sen[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)

    X_train = X_train.astype(np.int32)
    X_val = X_val.astype(np.int32)
    X_test = X_test.astype(np.int32)
    X_sen = X_sen.astype(np.int32)

    if len(hospital_list) != 0:
        y_train = y_train[[e in hospital_list for e in X_train[:,4]]]
        y_val = y_val[[e in hospital_list for e in X_val[:,4]]]
        y_test = y_test[[e in hospital_list for e in X_test[:,4]]]
        y_sen = y_sen[[e in hospital_list for e in X_sen[:,4]]]

        X_train = X_train[[e in hospital_list for e in X_train[:,4]],:]
        X_val = X_val[[e in hospital_list for e in X_val[:,4]],:]
        X_test = X_test[[e in hospital_list for e in X_test[:,4]],:]
        X_sen = X_sen[[e in hospital_list for e in X_sen[:,4]],:]

    if data_string == 'Coronary_Angiography':
        y_train = X_train[:,23]
        y_val = X_val[:,23]
        y_test = X_test[:,23]
        y_sen = X_sen[:,23]

        X_train = X_train[:,:23]
        X_val = X_val[:,:23]
        X_test = X_test[:,:23]
        X_sen = X_sen[:,:23]

    if (data_string == 'Cardiac_Stent') or (data_string == 'CABG'):
        if data_string == 'Cardiac_Stent':
            y_train, y_val, y_test, y_sen = X_train[:,24], X_val[:,24], X_test[:,24], X_sen[:,24]
        if data_string == 'CABG':
            y_train, y_val, y_test, y_sen = X_train[:,25], X_val[:,25], X_test[:,25], X_sen[:,25]

        X_train = X_train[:,:24]
        X_val = X_val[:,:24]
        X_test = X_test[:,:24]
        X_sen = X_sen[:,:24]

    if data_string == 'CPC1_2':
        X_train = X_train[:,:25]
        X_val = X_val[:,:25]
        X_test = X_test[:,:25]
        X_sen = X_sen[:,:25]


    X = np.concatenate((X_train, X_val), axis=0)
    num_cols = X.shape[1]
    metadata = []
    for col_id in range(num_cols):
        data = X[:, col_id]
        num_unique = max(np.unique(data))+1
        metadata.append(num_unique)

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            if X_test[i][j] >= metadata[j]:
                X_test[i][j] = metadata[j]-1

    for i in range(X_sen.shape[0]):
        for j in range(X_sen.shape[1]):
            if X_sen[i][j] >= metadata[j]:
                X_sen[i][j] = metadata[j]-1

    X_train = [X_train[:, i] for i in range(num_cols)]
    X_val = [X_val[:, i] for i in range(num_cols)]
    X_test = [X_test[:, i] for i in range(num_cols)]
    X_sen = [X_sen[:, i] for i in range(num_cols)]

    train_labels, train_counts = np.unique(y_train, return_counts=True)
    val_labels, val_counts = np.unique(y_val, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)
    sen_labels, sen_counts = np.unique(y_sen, return_counts=True)


    if print_bool: print("Agency set : ", a)
    if print_bool: print("Train set : ", (X_train[0].shape[0], num_cols), "  Class Label Counts: ", train_counts, " CPC1/2 Rate: ", 1-train_counts[0]/sum(train_counts))
    if print_bool: print("Val set : ", (X_val[0].shape[0], num_cols), "  Class Label Counts: ", val_counts, " CPC1/2 Rate: ", 1-val_counts[0]/sum(val_counts))
    if print_bool: print("Test set : ", (X_test[0].shape[0], num_cols), "  Class Label Counts: ", test_counts, " CPC1/2 Rate: ", 1-test_counts[0]/sum(test_counts))
    if print_bool: print("Sen set : ", (X_sen[0].shape[0], num_cols), "  Class Label Counts: ", sen_counts, " CPC1/2 Rate: ", 1-sen_counts[0]/sum(sen_counts))
    if print_bool: print()


    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_sen = to_categorical(y_sen, num_classes=num_classes)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_sen, y_sen), metadata



def change_problem(X, y, FIXED_PARAMETERS):
    keep = [int(i) for i in FIXED_PARAMETERS['inputs'].split(' ')]
    label = [int(i) for i in FIXED_PARAMETERS['outputs'].split(' ')]

    y_new = {}
    for l in label:
        y_new[l] = to_categorical(X[l], num_classes=2)
    # X_new = [X[e] for e in keep]
    return X, y_new

def change_meta(metadata, FIXED_PARAMETERS):
    keep = [int(i) for i in FIXED_PARAMETERS['inputs'].split(' ')]
    metadata = [metadata[e] for e in keep]
    return metadata


def subset_inputs(X_train, X_val, metadata, features):
    X_train = [X_train[e] for e in features]
    X_val = [X_val[e] for e in features]
    metadata = [metadata[e] for e in features]

    return X_train, X_val, metadata

def subset_inputs_all(X_train, X_val, X_test, metadata, features):
    X_train = [X_train[e] for e in features]
    X_val = [X_val[e] for e in features]
    X_test = [X_test[e] for e in features]
    metadata = [metadata[e] for e in features]

    return X_train, X_val, X_test, metadata