import argparse
import numpy as np
from utils.PU import load_PU_table
from utils.tools import scaler_fit, ML_models, scale_test
from utils.extraction_features import extracted_feature_of_signal, handcrafted_features
from NN_system.ML_embedding import FaceNetOneShotRecognitor
from NN_system.model import train_model

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--ML_method', default='SVM' , type=str, help='SVM, RF, KNN, LGBM, euclidean, cosine')
    parser.add_argument('--scaler', default='MinMaxScaler', type=str, help='MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer')
    parser.add_argument('--type_data', type=str, default='vibration', help='vibration, MCS1, MCS2')
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/Khoa/data/PU_data', help='direction of data')
    parser.add_argument('--weights_path', type=str, default='/content/drive/MyDrive/Khoa/vibration_project/Classification/results', help='direction of data')
    parser.add_argument('--img_outdir', type=str, default='/content/drive/MyDrive/Khoa/vibration_project/Classification/results/images', help='direction of data')
    parser.add_argument('--load_weight', default=False, type=bool)
    parser.add_argument('--Ex_feature', type=str, default='fre', help='time, fre, time_fre')
    parser.add_argument('--PU_table_8', default=True, type=bool)
    parser.add_argument('--PU_table_10', default=False, type=bool)

    # Parameters--------
    parser.add_argument('--alpha', default=0.4, type=int)
    parser.add_argument('--lambda_', default=0.3, type=int)
    parser.add_argument('--embedding_size', default=256, type=int)

    # Mode-------
    parser.add_argument('--table', type=str, default='table7', help='table6, table7')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def train_table6(opt):
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

def train_table7(opt):
    X_train, y_train, X_test, y_test = load_PU_table(opt)
    print('\n' + f'Shape of original training data: {X_train.shape, y_train.shape}')
    print(f'Shape of original test data: {X_test.shape, y_test.shape}' + '\n')

    model = train_model(X_train, y_train, X_test, y_test, opt)
    emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, opt)
    X_train_embed, X_test_embed, y_soft_pred = emb_sys.get_emb()
    emb_sys.predict(X_test_embed, X_train_embed, ML_method=opt.ML_method, use_mean_var=False)
    from TSNE_plot import tsne_plot
    tsne_plot(opt.img_outdir, 'original', X_train_embed[:, :opt.embedding_size], X_test_embed[:, :opt.embedding_size], y_train, y_test)
    tsne_plot(opt.img_outdir, 'extracted', X_train_embed[:, opt.embedding_size: ], X_test_embed[:, opt.embedding_size: ], y_train, y_test)


if __name__ == '__main__':
    opt = parse_opt()
    print('*'*10 + f'RUN: {opt.table}' + '*'*10)
    if opt.table == 'table6':
        train_table6(opt)
    if opt.table == 'table7':
        train_table7(opt) 
