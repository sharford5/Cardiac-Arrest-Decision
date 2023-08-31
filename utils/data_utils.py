import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import utils.parameters as params
FIXED_PARAMETERS = params.load_parameters()
#Set Seeds for Reproducability
os.environ['PYTHONHASHSEED']=str(int(FIXED_PARAMETERS['version'][1:]))
np.random.seed(int(FIXED_PARAMETERS['version'][1:]))
rn.seed(int(FIXED_PARAMETERS['version'][1:]))
tf.random.set_seed(int(FIXED_PARAMETERS['version'][1:]))
from keras import backend as K

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.utils import to_categorical
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.utils import shuffle


def _load_processed_dataset(FIXED_PARAMETERS, dataset, merge_bool=True, data_string = ''):
    X = np.load(FIXED_PARAMETERS['data_path'] + '/X2' + '_' + dataset + '.npy')
    y = np.load(FIXED_PARAMETERS['data_path'] + '/Y_CPC1_2_' + dataset + '.npy')

    if merge_bool:
        y[y == 2] = 1
    return X, y

def remove_elements(df, labels):
    ca = df[:,23]
    cs = df[:,24]

    remove_list = []
    for i in range(len(ca)):
        if (ca[i] == 0) and (cs[i] == 1):
            remove_list.append(i)

    df = np.delete(df, remove_list, axis=0)
    labels = np.delete(labels, remove_list, axis=0)
    return df, labels

def remove_elements_v2(df, df2, labels):
    ca = df[:,23]
    cs = df[:,24]

    remove_list = []
    for i in range(len(ca)):
        if (ca[i] == 0) and (cs[i] == 1):
            remove_list.append(i)

    df = np.delete(df, remove_list, axis=0)
    df2 = np.delete(df2, remove_list, axis=0)
    labels = np.delete(labels, remove_list, axis=0)
    return df, df2, labels


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

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_val, Y_val = shuffle(X_val, Y_val, random_state=0)

    return X_train, X_val, Y_train, Y_val


def train_val_split_v2(X, X2, y, test_size=0.2, random_state=0, prob_duplicate=0.0):
    train_labels, train_counts = np.unique(y, return_counts=True)
    X_train, X2_train, Y_train = [], [], []
    X_val, X2_val, Y_val = [], [], []

    for label, max_cnt in zip(train_labels, train_counts):
        samples = X[y == label, :]
        samples2 = X2[y == label, :]

        try:
            train_samples, val_samples, train_samples2, val_samples2 = train_test_split(samples, samples2, test_size=test_size, random_state=random_state)
        except:
            val_samples = samples ## Error handling sends all samples to the validation set
            train_samples = np.array([])

            val_samples2 = samples2 ## Error handling sends all samples to the validation set
            train_samples2 = np.array([])

        for i in range(len(train_samples)):
            X_train.append(train_samples[i])
            Y_train.append(label)
            if (rn.random() < prob_duplicate) and (label == 0):
                X_train.append(train_samples[i])
                Y_train.append(label)

        for i in range(len(val_samples)):
            X_val.append(val_samples[i])
            Y_val.append(label)

        for i in range(len(train_samples2)):
            X2_train.append(train_samples2[i])
            if (rn.random() < prob_duplicate) and (label == 0):
                X2_train.append(train_samples2[i])

        for i in range(len(val_samples2)):
            X2_val.append(val_samples2[i])

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    X2_train = np.asarray(X2_train)
    X2_val = np.asarray(X2_val)
    Y_train = np.asarray(Y_train)
    Y_val = np.asarray(Y_val)

    X_train, X2_train, Y_train = shuffle(X_train, X2_train, Y_train, random_state=0)
    X_val, X2_val, Y_val = shuffle(X_val, X2_val, Y_val, random_state=0)

    return X_train, X2_train, X_val, X2_val, Y_train, Y_val


def prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list, print_bool=False, data_string = ''):
    X_train, y_train, X_val, y_val, X_test, y_test, X_sen, y_sen, metadata = {},{},{},{},{},{},{},{}, {}

    for a in FIXED_PARAMETERS[agency_list].split(' '):
        X, y = _load_processed_dataset(FIXED_PARAMETERS, a, data_string=data_string)
        X, X_sen[a], y, y_sen[a] = train_test_split(X, y, test_size=0.25, random_state=0)
        X_train[a], X_test[a], y_train[a], y_test[a] = train_test_split(X, y, test_size=1/3, random_state=0)
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

    X_train, y_train = remove_elements(X_train, y_train)
    X_val, y_val = remove_elements(X_val, y_val)
    X_test, y_test = remove_elements(X_test, y_test)
    X_sen, y_sen = remove_elements(X_sen, y_sen)

    df_train = pd.DataFrame(np.concatenate([X_train, np.expand_dims(y_train, axis=-1)], axis=1))
    df_val = pd.DataFrame(np.concatenate([X_val, np.expand_dims(y_val, axis=-1)], axis=1))
    df_test = pd.DataFrame(np.concatenate([X_test, np.expand_dims(y_test, axis=-1)], axis=1))
    df_sen = pd.DataFrame(np.concatenate([X_sen, np.expand_dims(y_sen, axis=-1)], axis=1))
    # df_train.to_csv('./sets/train.csv')
    # df_val.to_csv('./sets/val.csv')
    # df_test.to_csv('./sets/test.csv')
    # df_sen.to_csv('./sets/sen.csv')


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

        #NOTE Removed Cardiac Stent (meaning 25 -> 24)
        X_train = X_train[:,:24]
        X_val = X_val[:,:24]
        X_test = X_test[:,:24]
        X_sen = X_sen[:,:24]


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


def _load_processed_dataset_v2(FIXED_PARAMETERS, dataset, merge_bool=True, data_string = ''):
    X = np.load(FIXED_PARAMETERS['data_path'] + '/X2' + '_' + dataset + '.npy')
    X2 = np.load(FIXED_PARAMETERS['data_path'] + '/X3' + '_' + dataset + '.npy')

    y = np.load(FIXED_PARAMETERS['data_path'] + '/Y_CPC1_2_' + dataset + '.npy')

    if merge_bool:
        y[y == 2] = 1
    return X, X2, y

def prepare_dataset_merged_keras_combined_v2(FIXED_PARAMETERS, agency_list, print_bool=False, data_string = ''):
    X_train, X2_train, y_train, X_val, X2_val, y_val, X_test, X2_test, y_test, X_sen, X2_sen, y_sen, metadata = {},{},{},{},{},{},{},{},{},{},{},{},{}

    for a in FIXED_PARAMETERS[agency_list].split(' '):
        X, X2, y = _load_processed_dataset_v2(FIXED_PARAMETERS, a, data_string=data_string)
        X, X_sen[a], X2, X2_sen[a], y, y_sen[a] = train_test_split(X, X2, y, test_size=0.25, random_state=0)
        X_train[a], X_test[a], X2_train[a], X2_test[a], y_train[a], y_test[a] = train_test_split(X, X2, y, test_size=1/3, random_state=0)
        X_train[a], X2_train[a], X_val[a], X2_val[a], y_train[a], y_val[a] = train_val_split_v2(X_train[a], X2_train[a], y_train[a], test_size=0.2, random_state=0)

    X_train = np.concatenate([X_train[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X_val = np.concatenate([X_val[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X_test = np.concatenate([X_test[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X_sen = np.concatenate([X_sen[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)

    X2_train = np.concatenate([X2_train[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X2_val = np.concatenate([X2_val[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X2_test = np.concatenate([X2_test[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    X2_sen = np.concatenate([X2_sen[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)

    y_train = np.concatenate([y_train[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    y_val = np.concatenate([y_val[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    y_test = np.concatenate([y_test[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)
    y_sen = np.concatenate([y_sen[a] for a in FIXED_PARAMETERS[agency_list].split(' ')], axis=0)

    X_train = X_train.astype(np.int32)
    X_val = X_val.astype(np.int32)
    X_test = X_test.astype(np.int32)
    X_sen = X_sen.astype(np.int32)

    # X2_train = X2_train.astype(np.int32)
    # X2_val = X2_val.astype(np.int32)
    # X2_test = X2_test.astype(np.int32)
    # X2_sen = X2_sen.astype(np.int32)

    X_train, X2_train, y_train = remove_elements_v2(X_train, X2_train, y_train)
    X_val, X2_val, y_val = remove_elements_v2(X_val, X2_val, y_val)
    X_test, X2_test, y_test = remove_elements_v2(X_test, X2_test, y_test)
    X_sen, X2_sen, y_sen = remove_elements_v2(X_sen, X2_sen, y_sen)

    # df_train = pd.DataFrame(np.concatenate([X_train, np.expand_dims(y_train, axis=-1)], axis=1))
    # df_val = pd.DataFrame(np.concatenate([X_val, np.expand_dims(y_val, axis=-1)], axis=1))
    # df_test = pd.DataFrame(np.concatenate([X_test, np.expand_dims(y_test, axis=-1)], axis=1))
    # df_sen = pd.DataFrame(np.concatenate([X_sen, np.expand_dims(y_sen, axis=-1)], axis=1))
    # df_train.to_csv('./sets/train.csv')
    # df_val.to_csv('./sets/val.csv')
    # df_test.to_csv('./sets/test.csv')
    # df_sen.to_csv('./sets/sen.csv')


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

    return (X_train, X2_train, y_train), (X_val, X2_val, y_val), (X_test, X2_test, y_test), (X_sen, X2_sen, y_sen), metadata



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