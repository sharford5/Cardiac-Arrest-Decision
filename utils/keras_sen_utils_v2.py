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
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, roc_auc_score

from utils.data_utils import change_problem, change_meta, prepare_dataset_merged_keras_combined, subset_inputs, subset_inputs_all
from utils.metrics import F1Callback
import pandas as pd


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def train_model(model_fn, FIXED_PARAMETERS, even_check=False, model_type = '', hospital=[]):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_sen, y_sen), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list='agency_list_train', data_string=model_type, print_bool=True)
    num_classes = len(np.unique(np.argmax(y_train, axis=-1)))

    X_train = np.concatenate([np.asarray(X_train), np.asarray(X_val), np.asarray(X_test), np.asarray(X_sen)], axis=1)
    X_train = [X_train[i,:] for i in range(len(metadata))]
    y_train = np.concatenate([y_train, y_val, y_test, y_sen])

    if len(hospital) != 0:
        bool_list = np.isin(X_train[4], hospital)
        X_train = [X_train[i][bool_list] for i in range(len(X_train))]
        y_train = y_train[bool_list]

    if len(hospital) != 0:
        bool_list = np.isin(X_val[4], hospital)
        X_val = [X_val[i][bool_list] for i in range(len(X_val))]
        y_val = y_val[bool_list]

    K.clear_session()
    with tf.device(FIXED_PARAMETERS['device']):
        print(FIXED_PARAMETERS['device'])
        model = model_fn(num_classes=num_classes, metadata=metadata, conv_layers=FIXED_PARAMETERS['conv_layers'], conv_nodes=FIXED_PARAMETERS['conv_nodes'],
                         embedding_dim=FIXED_PARAMETERS['embed_dim'], dropout_rate=FIXED_PARAMETERS['dropout'])

    optimizer = Adam(FIXED_PARAMETERS['learning_rate'])
    model.compile(optimizer, loss=focal_loss(gamma=FIXED_PARAMETERS['gamma'], alpha=FIXED_PARAMETERS['alpha']), metrics=['accuracy'])

    weights_path = './weights/sen_%s%s_v2.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type)
    f1_callback = F1Callback(weights_path)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=6, verbose=False, min_lr=2e-5)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    callbacks = [lr, es, f1_callback]

    classes = np.unique(np.argmax(y_train, axis=-1))
    le = LabelEncoder()
    y_ind = le.fit_transform(np.argmax(y_train, axis=-1).ravel())
    recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))

    class_weight = recip_freq[le.transform(classes)]
    class_weight = [cw / min(class_weight) for cw in class_weight]
    print("Class weights : ", class_weight)
    history = model.fit(X_train, y_train, FIXED_PARAMETERS['batchsize'], FIXED_PARAMETERS['epochs'], callbacks=callbacks, class_weight=class_weight, validation_data=(X_val, y_val), verbose = 0)



def evaluate_model(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital=[]):
    (X_train, y_train), (X_val, y_val), (_, _), (_, _), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type)
    (_, _), (_, _), (X_test, y_test), (_, _), _ = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_eval', print_bool=print_bool, data_string = model_type)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

    if len(hospital) != 0:
        bool_list = np.isin(X_test[4], hospital)
        X_test = [X_test[i][bool_list] for i in range(len(X_test))]
        y_test = y_test[bool_list]

    #Change Trained testing to a null embedding
    for i in range(len(X_test)):
        for j in range(X_test[i].shape[0]):
            if X_test[i][j] >= metadata[i]:
                X_test[i][j] = metadata[i]-1

    K.clear_session()
    with tf.device('cpu:0'):
        model = model_fn(num_classes=num_classes, metadata=metadata, conv_layers=FIXED_PARAMETERS['conv_layers'], conv_nodes=FIXED_PARAMETERS['conv_nodes'],
                         embedding_dim=FIXED_PARAMETERS['embed_dim'], dropout_rate=FIXED_PARAMETERS['dropout'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights_path = './weights/sen_%s%s_v2.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type)
    model.load_weights(weights_path)

    preds = model.predict(X_test, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    cm = confusion_matrix(y_true, preds, labels= [i for i in range(num_classes)])
    if print_bool: print(cm)
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    auc = roc_auc_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)
    if print_bool: print("AUC: ", auc)
    if print_bool: print("F1: ", f1)

    return preds, y_true


def eval_train_model(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital=[]):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_sen, y_sen), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type, print_bool=True)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

    if len(hospital) != 0:
        bool_list = np.isin(X_train[4], hospital)
        X_train = [X_train[i][bool_list] for i in range(len(X_train))]
        y_train = y_train[bool_list]

    #Change Trained testing to a null embedding
    for i in range(len(X_train)):
        for j in range(X_train[i].shape[0]):
            if X_train[i][j] >= metadata[i]:
                X_train[i][j] = metadata[i]-1

    K.clear_session()
    with tf.device('cpu:0'):
        model = model_fn(num_classes=num_classes, metadata=metadata, conv_layers=FIXED_PARAMETERS['conv_layers'], conv_nodes=FIXED_PARAMETERS['conv_nodes'],
                         embedding_dim=FIXED_PARAMETERS['embed_dim'], dropout_rate=FIXED_PARAMETERS['dropout'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights_path = './weights/sen_%s%s_v2.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type)
    model.load_weights(weights_path)

    preds = model.predict(X_train, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_train, axis=-1)

    cm = confusion_matrix(y_true, preds, labels= [i for i in range(num_classes)])
    if print_bool: print(cm)
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    auc = roc_auc_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)
    if print_bool: print("AUC: ", auc)
    if print_bool: print("F1: ", f1)

    return preds, y_true

def eval_val_model(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital=[]):
    (X_train, y_train), (X_val, y_val), (_, _), (_, _), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

    if len(hospital) != 0:
        bool_list = np.isin(X_train[4], hospital)
        X_train = [X_train[i][bool_list] for i in range(len(X_train))]
        y_train = y_train[bool_list]

    if len(hospital) != 0:
        bool_list = np.isin(X_val[4], hospital)
        X_val = [X_val[i][bool_list] for i in range(len(X_val))]
        y_val = y_val[bool_list]

    #Change Trained testing to a null embedding
    for i in range(len(X_val)):
        for j in range(X_val[i].shape[0]):
            if X_val[i][j] >= metadata[i]:
                X_val[i][j] = metadata[i]-1

    K.clear_session()
    with tf.device('cpu:0'):
        model = model_fn(num_classes=num_classes, metadata=metadata, conv_layers=FIXED_PARAMETERS['conv_layers'], conv_nodes=FIXED_PARAMETERS['conv_nodes'],
                         embedding_dim=FIXED_PARAMETERS['embed_dim'], dropout_rate=FIXED_PARAMETERS['dropout'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights_path = './weights/sen_%s%s_v2.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type)
    model.load_weights(weights_path)

    preds = model.predict(X_val, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_val, axis=-1)

    cm = confusion_matrix(y_true, preds, labels= [i for i in range(num_classes)])
    if print_bool: print(cm)
    class0_recall = cm[0][0]/(cm[0][0]+cm[0][1])
    class1_recall = cm[1][1]/(cm[1][0]+cm[1][1])
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    auc = roc_auc_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)
    if print_bool: print("AUC: ", auc)
    if print_bool: print("F1: ", f1)

    return avg_recall, acc, cm


def eval_sen_model(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital=[]):
    (_, _), (_, _), (_, _), (X_sen, y_sen), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type, print_bool=True)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

    if len(hospital) != 0:
        bool_list = np.isin(X_sen[4], hospital)
        X_sen = [X_sen[i][bool_list] for i in range(len(X_sen))]
        y_sen = y_sen[bool_list]

    y_sen2 = np.argmax(y_sen, axis=-1)
    sen_labels, sen_counts = np.unique(y_sen2, return_counts=True)
    if print_bool: print("  Class Label Counts: ", sen_counts, " CPC1/2 Rate: ", 1-sen_counts[0]/sum(sen_counts))


    #Change Trained testing to a null embedding
    for i in range(len(X_sen)):
        for j in range(X_sen[i].shape[0]):
            if X_sen[i][j] >= metadata[i]:
                X_sen[i][j] = metadata[i]-1

    K.clear_session()
    with tf.device('cpu:0'):
        model = model_fn(num_classes=num_classes, metadata=metadata, conv_layers=FIXED_PARAMETERS['conv_layers'], conv_nodes=FIXED_PARAMETERS['conv_nodes'],
                         embedding_dim=FIXED_PARAMETERS['embed_dim'], dropout_rate=FIXED_PARAMETERS['dropout'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights_path = './weights/sen_%s%s_v2.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type)
    model.load_weights(weights_path)

    preds = model.predict(X_sen, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_sen, axis=-1)

    cm = confusion_matrix(y_true, preds, labels= [i for i in range(num_classes)])
    if print_bool: print(cm)
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    # auc = roc_auc_score(y_true, preds)
    # f1 = f1_score(y_true, preds)

    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)
    # if print_bool: print("AUC: ", auc)
    # if print_bool: print("F1: ", f1)

    return preds, y_true



