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
import copy

import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score

from utils.data_hospitals_utils import change_problem, change_meta, prepare_dataset_merged_keras_combined, subset_inputs, subset_inputs_all
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

def train_model(model_fn, FIXED_PARAMETERS, even_check=False, model_type = '', hospital_list_train=[], hospital_list_test=[]):
    (X_train, y_train), (X_val, y_val), (_, _), (_, _), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list='agency_list_train', data_string=model_type, print_bool=True, hospital_list = hospital_list_train)
    num_classes = len(np.unique(np.argmax(y_train, axis=-1)))

    K.clear_session()
    with tf.device(FIXED_PARAMETERS['device']):
        print(FIXED_PARAMETERS['device'])
        model = model_fn(num_classes=num_classes, metadata=metadata, conv_layers=FIXED_PARAMETERS['conv_layers'], conv_nodes=FIXED_PARAMETERS['conv_nodes'],
                         embedding_dim=FIXED_PARAMETERS['embed_dim'], dropout_rate=FIXED_PARAMETERS['dropout'])

    optimizer = Adam(FIXED_PARAMETERS['learning_rate'])
    model.compile(optimizer, loss=focal_loss(gamma=FIXED_PARAMETERS['gamma'], alpha=FIXED_PARAMETERS['alpha']), metrics=['accuracy'])

    weights_path = './weights/%s%s_%s.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type, '-'.join([str(s) for s in hospital_list_train]))
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



def evaluate_model(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital_list_train=[], hospital_list_test=[]):
    (X_train, y_train), (X_val, y_val), (_, _), (_, _), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type, hospital_list = hospital_list_train)
    (_, _), (_, _), (X_test, y_test), (_, _), _ = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_eval', print_bool=print_bool, data_string = model_type, hospital_list = hospital_list_test)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

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
    weights_path = './weights/%s%s_%s.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type, '-'.join([str(s) for s in hospital_list_train]))
    model.load_weights(weights_path)

    preds = model.predict(X_test, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    cm = confusion_matrix(y_true, preds, labels= [i for i in range(num_classes)])
    if print_bool: print(cm)
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)

    # df = pd.DataFrame()
    # df['actual'] = list(y_true)
    # df['pred'] = list(preds)
    # df.to_csv('%s.csv' % model_type)

    return avg_recall, acc, cm


def eval_val_model(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital_list_train=[], hospital_list_test=[]):
    (X_train, y_train), (X_val, y_val), (_, _), (_, _), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type,hospital_list = hospital_list_train)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

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
    weights_path = './weights/%s%s_%s.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type, '-'.join([str(s) for s in hospital_list_train]))
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

    if print_bool: print("Class0 Recall: ", class0_recall)
    if print_bool: print("Class1 Recall: ", class1_recall)
    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)

    return avg_recall, acc, cm


def eval_sen_model(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital_list_train=[], hospital_list_test=[]):
    (X_train, y_train), (X_val, y_val), (_, _), (_, _), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type, hospital_list = hospital_list_train)
    (_, _), (_, _), (_, _), (X_sen, y_sen), _ = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_eval', print_bool=print_bool, data_string = model_type, hospital_list = hospital_list_test)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

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
    weights_path = './weights/%s%s_%s.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type, '-'.join([str(s) for s in hospital_list_train]))
    try:
        model.load_weights(weights_path)
    except:
        pass

    preds = model.predict(X_sen, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_sen, axis=-1)

    cm = confusion_matrix(y_true, preds, labels= [i for i in range(num_classes)])
    if print_bool: print(cm)
    class0_recall = cm[0][0]/(cm[0][0]+cm[0][1])
    class1_recall = cm[1][1]/(cm[1][0]+cm[1][1])
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    if print_bool: print("Class0 Recall: ", class0_recall)
    if print_bool: print("Class1 Recall: ", class1_recall)
    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)
    return avg_recall, acc, cm


def sensitivity_analysis_ca(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital_list_train=[], hospital_list_test=[]):
    (X_train, y_train), (X_val, y_val), (_, _), (_, _), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type, hospital_list = hospital_list_train)
    (_, _), (_, _), (_, _), (X_sen, y_sen), _ = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_eval', print_bool=print_bool, data_string = model_type, hospital_list = hospital_list_test)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

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
    weights_path = './weights/%s%s_%s.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type, '-'.join([str(s) for s in hospital_list_train]))
    model.load_weights(weights_path)

    preds = model.predict(X_sen, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_sen, axis=-1)

    cm = confusion_matrix(y_true, preds, labels= [i for i in range(num_classes)])
    if print_bool: print(cm)
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)
    return preds, y_true


def sensitivity_analysis_cs(model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital=[]):
    (_, _), (_, _), (_, _), (X_sen, y_sen), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type, print_bool=True)
    (_, _), (_, _), (_, _), (X_sen_cpc, y_sen_cpc), metadata2 = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = 'CPC1_2', print_bool=True)

    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

    if len(hospital) != 0:
        bool_list = np.isin(X_sen[4], hospital)
        X_sen = [X_sen[i][bool_list] for i in range(len(X_sen))]
        y_sen = y_sen[bool_list]

        X_sen_cpc = [X_sen_cpc[i][bool_list] for i in range(len(X_sen_cpc))]
        y_sen_cpc = y_sen_cpc[bool_list]

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
    preds_original = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_sen, axis=-1)


    df = pd.DataFrame()
    y_true_cpc = np.argmax(y_sen_cpc, axis=-1)
    df['Actual_CPC'] = y_true_cpc


    df['Actual_CS'] = y_true
    df['Original_CA'] = X_sen[-1]
    df['Original_CS_Pred'] = preds_original

    X_sen_cpc = X_sen + [preds_original]

    X_sen_new = copy.deepcopy(X_sen)
    new_ca = np.asarray([int(not i) for i in X_sen_new[-1]])
    X_sen_new[-1] = new_ca

    preds = model.predict(X_sen_new, FIXED_PARAMETERS['batchsize'])
    preds_new = np.argmax(preds, axis=-1)

    df['New_CA'] = X_sen_new[-1]
    df['New_CS_Pred'] = preds_new

    X_sen_new_cpc = X_sen_new + [preds_new]


    for i in range(len(X_sen_cpc)):
        for j in range(X_sen_cpc[0].shape[0]):
            if X_sen_cpc[i][j] >= metadata2[i]:
                X_sen_cpc[i][j] = metadata2[i]-1

    for i in range(len(X_sen_new_cpc)):
        for j in range(X_sen_new_cpc[0].shape[0]):
            if X_sen_new_cpc[i][j] >= metadata2[i]:
                X_sen_new_cpc[i][j] = metadata2[i]-1


    FIXED_PARAMETERS = {'data_path': './data/split', 'agency_list_eval': '223', 'agency_list_train': '223', 'mt': 'CPC1_2', 'train_bool': 1, 'train_sen_bool': 1, 'print_bool': 1, 'version': 'v1', 'epochs': 150, 'batchsize': 128, 'learning_rate': 0.002, 'dropout': 0.9, 'embed_dim': 10, 'conv_layers': 3, 'conv_nodes': 64, 'gamma': 2, 'alpha': 0.7, 'device': '/cpu:0'}
    K.clear_session()
    with tf.device('cpu:0'):
        model = model_fn(num_classes=num_classes, metadata=metadata2, conv_layers=FIXED_PARAMETERS['conv_layers'], conv_nodes=FIXED_PARAMETERS['conv_nodes'],
                         embedding_dim=FIXED_PARAMETERS['embed_dim'], dropout_rate=FIXED_PARAMETERS['dropout'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights_path = './weights/sen_%s%s_v2.h5' % (FIXED_PARAMETERS['agency_list_train'], 'CPC1_2')
    model.load_weights(weights_path)


    preds_cpc = model.predict(X_sen_cpc, FIXED_PARAMETERS['batchsize'])
    preds_cpc = np.argmax(preds_cpc, axis=-1)
    df['Original_CPC_Pred'] = preds_cpc


    preds_cpc_new = model.predict(X_sen_new_cpc, FIXED_PARAMETERS['batchsize'])
    preds_cpc_new = np.argmax(preds_cpc_new, axis=-1)
    df['New_CPC_Pred'] = preds_cpc_new


    df.to_csv('sa2.csv')

    return preds, y_true


def sensitivity_analysis_cpc(pred_ca, pred_cs, model_fn, FIXED_PARAMETERS, print_bool=True, model_type='', hospital_list_train=[], hospital_list_test=[]):
    (X_train, y_train), (X_val, y_val), (_, _), (_, _), metadata = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_train', data_string = model_type, hospital_list = hospital_list_train)
    (_, _), (_, _), (_, _), (X_sen, y_sen), _ = prepare_dataset_merged_keras_combined(FIXED_PARAMETERS, agency_list = 'agency_list_eval', print_bool=print_bool, data_string = model_type, hospital_list = hospital_list_test)
    num_classes = 2# len(np.unique(np.argmax(y_test, axis=-1)))

    X_sen[-2] = pred_ca
    X_sen[-1] = pred_cs

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
    weights_path = './weights/%s%s.h5' % (FIXED_PARAMETERS['agency_list_train'], model_type)
    model.load_weights(weights_path)

    preds = model.predict(X_sen, FIXED_PARAMETERS['batchsize'])
    preds = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_sen, axis=-1)

    cm = confusion_matrix(y_true, preds, labels= [i for i in range(num_classes)])
    if print_bool: print(cm)
    avg_recall = 0.5*(cm[0][0]/(cm[0][0]+cm[0][1])) + 0.5*(cm[1][1]/(cm[1][0]+cm[1][1]))
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    if print_bool: print("Avg Recall: ", avg_recall)
    if print_bool: print("ACC: ", acc)
    return preds, y_true