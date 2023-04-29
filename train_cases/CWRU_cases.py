from utils.CWRU import load_CWRU
from utils.tools import one_hot_inverse
from NN_system.main_system import train_main_system
from NN_system.ML_embedding import FaceNetOneShotRecognitor
import tensorflow as tf
import numpy as np

def train_table_10_11_12(opt):
    X_train, X_test, y_train, y_test = load_CWRU(opt)
    y_train = one_hot_inverse(y_train)
    y_test = one_hot_inverse(y_test)
    print(f"\nCWRU case: {opt.CWRU_case}")
    print('\n' + f'Shape of original training data: {X_train.shape, y_train.shape}')
    print(f'Shape of original test data: {X_test.shape, y_test.shape}' + '\n')
    model = train_main_system(X_train, y_train, X_test, y_test, opt)
    emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, opt)
    X_train_embed, X_test_embed = emb_sys.get_emb()
    
    for each_ML in opt.ML_method:
        tf.keras.backend.clear_session()
        print("-"*10 + each_ML + "-"*10)
        emb_sys.predict(X_test_embed, X_train_embed, ML_method=each_ML, use_mean_var=False)
        print("-"*20, '\n')