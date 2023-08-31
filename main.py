import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Load Parameters
import utils.parameters as params
FIXED_PARAMETERS = params.load_parameters()

#Set Seeds for Reproducability
os.environ['PYTHONHASHSEED']=str(int(FIXED_PARAMETERS['version'][1:]))
np.random.seed(int(FIXED_PARAMETERS['version'][1:]))
rn.seed(int(FIXED_PARAMETERS['version'][1:]))
tf.random.set_seed(int(FIXED_PARAMETERS['version'][1:]))
from keras import backend as K

import pandas as pd
from utils.keras_utils import train_model, evaluate_model, eval_val_model, eval_sen_model
from utils.model_utils import build_embedding_model

agencies = ['223']

if __name__ == '__main__':
    print(FIXED_PARAMETERS)
    mt = FIXED_PARAMETERS['mt'] #CPC1_2'
    a1 = '223'
    a2 = '223'
    FIXED_PARAMETERS['agency_list_train'] = a1
    FIXED_PARAMETERS['agency_list_eval'] = a2
    print(a1, " ", a2)

    train_model(build_embedding_model, FIXED_PARAMETERS, model_type=mt)
    eval_val_model(build_embedding_model, FIXED_PARAMETERS, print_bool=True, model_type=mt)
    evaluate_model(build_embedding_model, FIXED_PARAMETERS, print_bool=True, model_type=mt)
    eval_sen_model(build_embedding_model, FIXED_PARAMETERS, print_bool=True, model_type=mt)




