import argparse
from load_data.PU import load_PU_table
from utils.tools import scaler_fit, ML_models, scale_test
from utils.extraction_features import extracted_feature_of_signal, AudioFeatureExtractor

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--ML_method', default='SVM' , type=str, help='SVM, RF, KNN, LGBM')
    parser.add_argument('--scaler', default='MinMaxScaler', type=str, help='MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer')
    parser.add_argument('--type_PU_data', type=str, default='vibration', help='vibration, MCS1, MCS2')
    parser.add_argument('--PU_table_8', default=True, type=bool)
    parser.add_argument('--PU_table_10', default=True, type=bool)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def train(opt):
    X_train, y_train, X_test, y_test = load_PU_table(opt)
    print('\n' + f'Shape of training data: {X_train.shape, y_train.shape}')
    print('\n' + f'Shape of test data: {X_test.shape, y_test.shape}')

    # Extracting data ---------------------------------
    X_train = extracted_feature_of_signal(X_train)
    X_test = extracted_feature_of_signal(X_test)

    # Scale data----------------------------------------
    X_train, scale = scaler_fit(X_train, opt)
    X_test = scale_test(X_test, scale)

    # ML model------------------------------------------
    ML_models(X_train, y_train, X_test, y_test opt)


if __name__ == '__main__':
    opt = parse_opt()
    
