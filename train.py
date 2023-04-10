import argparse
from train_cases.PU_t8 import train_table_6, train_table_7

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--ML_method', default=['RF', 'KNN', 'LGBM', 'euclidean', 'cosine', 'SVM']  , type=list, help='SVM, RF, KNN, LGBM, euclidean, cosine')
    parser.add_argument('--scaler', default="PowerTransformer", type=str, help='MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer')
    parser.add_argument('--type_data', type=str, default='vibration', help='vibration, MCS1, MCS2')
    parser.add_argument('--data_dir', type=str, default='./data/PU_data', help='direction of data')
    parser.add_argument('--weights_path', type=str, default='./results', help='direction of data')
    parser.add_argument('--img_outdir', type=str, default='./results/images', help='direction of data')
    parser.add_argument('--load_weights', default=True, type=bool)
    parser.add_argument('--Ex_feature', type=str, default='fre', help='time, fre, time_fre')
    parser.add_argument('--PU_table_8', default=False, type=bool)
    parser.add_argument('--PU_table_10', default=True, type=bool)

    # Parameters--------
    parser.add_argument('--alpha', default=0.4, type=int)
    parser.add_argument('--lambda_', default=0.3, type=int)
    parser.add_argument('--embedding_size', default=256, type=int)
    parser.add_argument('--e_input_shape', default=6270, type=int, help='11, 6270, 6281') 
    parser.add_argument('--input_shape', default=250604, type=int)  
    parser.add_argument('--num_classes', default=3, type=int) 
    parser.add_argument('--batch_size', default=16, type=int) 
    parser.add_argument('--epochs', default=30, type=int) 
    parser.add_argument('--train_mode', default=True, type=bool)
    parser.add_argument('--get_SDLM_extract', default=False, type=bool)
    parser.add_argument('--TSNE_plot', default=False, type=bool) # get_SDLM_extract
    
    # Mode-------
    parser.add_argument('--table', type=str, default='table7', help='table6, table7')
    parser.add_argument('--model', type=str, default='main_model', help='main_model, SDLM, S_SDLM, U_SDLM')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    print('*'*10 + f' RUN: {opt.table} ' + '*'*10)
    if opt.table == 'table6':
        train_table_6(opt)
    if opt.table == 'table7':
        train_table_7(opt)
        