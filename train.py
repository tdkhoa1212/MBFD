import argparse
from train_cases.PU_t8 import train_table_6, train_table_7
from train_cases.PU_t10 import train_table_9
from train_cases.CWRU_cases import train_table_10_11_12

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--ML_method', default=['RF', 'KNN', 'euclidean', 'cosine', 'SVM']  , type=list, help='SVM, RF, KNN, euclidean, cosine')
    parser.add_argument('--scaler', default="PowerTransformer", type=str, help='MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer')
    parser.add_argument('--type_data', type=str, default='vibration', help='vibration, MCS1, MCS2')
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/Khoa/data_new/data/CWRU_data/', help='direction of data')
    parser.add_argument('--weights_path', type=str, default='/content/drive/MyDrive/Khoa/results/', help='direction of data')
    parser.add_argument('--img_outdir', type=str, default='/content/drive/MyDrive/Khoa/results/images/', help='direction of data')
    parser.add_argument('--load_weights', default=False, type=bool)
    parser.add_argument('--Ex_feature', type=str, default='fre', help='time, fre, time_fre')

    # Data------------------------------------
    parser.add_argument('--PU_table_8', default=False, type=bool)
    parser.add_argument('--PU_table_10', default=False, type=bool)
    parser.add_argument('--CWRU_case', default="1", type=str, help="1, 2, 3, 4")
    parser.add_argument('--path_saved_data', type=str, default='/content/drive/MyDrive/Khoa/results/saved_data/', help='direction of data')

    # Parameters--------
    parser.add_argument('--alpha', default=0.4, type=int)
    parser.add_argument('--lambda_', default=0.3, type=int)
    parser.add_argument('--embedding_size', default=256, type=int)
    parser.add_argument('--e_input_shape', default=15, type=int, help='11, 16, 6270, 6400') 
    parser.add_argument('--input_shape', default=400, type=int, help='255990, 400')  
    parser.add_argument('--num_classes', default=45, type=int) 
    parser.add_argument('--batch_size', default=16, type=int) 
    parser.add_argument('--epochs', default=300, type=int) 
    parser.add_argument('--train_mode', default=True, type=bool)
    parser.add_argument('--get_SDLM_extract', default=False, type=bool)
    parser.add_argument('--TSNE_plot', default=False, type=bool) # get_SDLM_extract
    
    # Mode-------
    parser.add_argument('--table', type=str, default='table_10_11_12', help='table_6, table_7, table_9')
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
    if opt.table == 'table_9':
        train_table_9(opt)
    if opt.table == 'table_10_11_12':
        train_table_10_11_12(opt)
        