import argparse
import numpy as np
from utils.PU import load_PU_table
from utils.tools import scaler_fit, ML_models, scale_test, load_PU_data
from utils.extraction_features import extracted_feature_of_signal, handcrafted_features

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--ML_method', default='SVM' , type=str, help='SVM, RF, KNN, LGBM')
    parser.add_argument('--scaler', default='MinMaxScaler', type=str, help='MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer')
    parser.add_argument('--type_data', type=str, default='vibration', help='vibration, MCS1, MCS2')
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/Khoa/data/PU_data', help='direction of data')
    parser.add_argument('--Ex_feature', type=str, default='time', help='time, fre, time_fre')
    parser.add_argument('--PU_table_8', default=True, type=bool)
    parser.add_argument('--PU_table_10', default=False, type=bool)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def train(opt):
    X_train, y_train, X_test, y_test = load_PU_table(opt)
    print('\n' + f'Shape of original training data: {X_train.shape, y_train.shape}')
    print(f'Shape of original test data: {X_test.shape, y_test.shape}' + '\n')

    # Extracting data ---------------------------------
    if opt.Ex_feature == 'time':
        X_train = extracted_feature_of_signal(X_train)
        X_test = extracted_feature_of_signal(X_test)
    if opt.Ex_feature == 'fre':
        X_train = handcrafted_features(X_train)
        X_test = handcrafted_features(X_test)
    if opt.Ex_feature == 'time_fre':
        X_train = np.concatenate((extracted_feature_of_signal(X_train), handcrafted_features(X_train)), axis=-1)
        X_test = np.concatenate((extracted_feature_of_signal(X_test), handcrafted_features(X_test)), axis=-1)

    print('\n' + f'Shape of present training data: {X_train.shape, y_train.shape}')
    print(f'Shape of present test data: {X_test.shape, y_test.shape}' + '\n')

    # Scale data----------------------------------------
    if opt.scaler != 'None':
        X_train, scale = scaler_fit(X_train, opt)
        X_test = scale_test(X_test, scale)

    # ML model------------------------------------------
    ML_models(X_train, y_train, X_test, y_test, opt)


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
    
