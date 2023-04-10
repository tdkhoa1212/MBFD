from NN_system.ML_embedding import FaceNetOneShotRecognitor
from utils.PU import Healthy, Outer_ring_damage, Inner_ring_damage 
from utils.tools import invert_one_hot, load_table_10_spe,\
                        recall_m, precision_m, f1_m, to_one_hot, handcrafted_features, scaler_transform, plot_confusion

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from itertools import combinations
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine, euclidean
from utils.angular_grad import AngularGrad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(all="ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
opt = parse_opt()
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

def train_table_9(path_saved_data):
    print('\t\t\t Loading labels...')
    Healthy_label           = np.array([0]*len(Healthy))
    Outer_ring_damage_label = np.array([1]*len(Outer_ring_damage))
    Inner_ring_damage_label = np.array([2]*len(Inner_ring_damage))

    ###################################### LOAD DATA ######################################
    if opt.PU_data_table_10_case_0:
        Healthy, Healthy_label = load_table_10_spe(Healthy, Healthy_label)
        Outer_ring_damage, Outer_ring_damage_label = load_table_10_spe(Outer_ring_damage, Outer_ring_damage_label)
        Inner_ring_damage, Inner_ring_damage_label = load_table_10_spe(Inner_ring_damage, Inner_ring_damage_label)
        if os.path.exists(join(path_saved_data, '/Healthy_10.npy')):
            Healthy = np.load(join(path_saved_data, '/Healthy_10.npy'), mmap_mode="r")  
            Outer_ring_damage = np.load(join(path_saved_data, '/Outer_ring_damage_10.npy'), mmap_mode="r")
            Inner_ring_damage = np.load(join(path_saved_data, '/Inner_ring_damage_10.npy'), mmap_mode="r")
        else: 
            Healthy = scaler_transform(Healthy, PowerTransformer)
            Outer_ring_damage = scaler_transform(Outer_ring_damage, PowerTransformer)
            Inner_ring_damage = scaler_transform(Inner_ring_damage, PowerTransformer)

            with open(join(path_saved_data, '/Healthy_10.npy'), 'wb') as f:
                np.save(f, Healthy)
            with open(join(path_saved_data, '/Outer_ring_damage_10.npy'), 'wb') as f:
                np.save(f, Outer_ring_damage)
            with open(join(path_saved_data, '/Inner_ring_damage_10.npy'), 'wb') as f:
                np.save(f, Inner_ring_damage)

        ###################################### PROCESS ######################################
        np.random.seed(0)
        Healthy, Healthy_label = shuffle(Healthy, Healthy_label, random_state=0)
        Outer_ring_damage, Outer_ring_damage_label = shuffle(Outer_ring_damage, Outer_ring_damage_label, random_state=0)
        Inner_ring_damage, Inner_ring_damage_label = shuffle(Inner_ring_damage, Inner_ring_damage_label, random_state=0)

        print(color.GREEN + '\n\n\t *************START*************\n\n' + color.END)
        emb_accuracy_SVM = []
        emb_accuracy_RF = []
        emb_accuracy_KNN = []
        emb_accuracy_LGBM = []
        emb_accuracy_euclidean = []
        emb_accuracy_cosine = []

        emb_accuracy_ensemble = []

        #------------------------------------------Case 0: shuffle------------------------------------------------
        if opt.PU_data_table_10_case_0:
            for i in range(5):
                distance_Healthy = int(0.6*len(Healthy))
                start_Healthy    = int(0.2*i*len(Healthy))
                X_train_Healthy = Healthy[start_Healthy: start_Healthy+distance_Healthy]
                if len(X_train_Healthy) < distance_Healthy:
                    break
                y_train_Healthy = Healthy_label[start_Healthy: start_Healthy+distance_Healthy]
                print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
                
                distance_Outer_ring_damage = int(0.6*len(Outer_ring_damage))
                start_Outer_ring_damage    = int(0.2*i*len(Outer_ring_damage))
                X_train_Outer_ring_damage, y_train_Outer_ring_damage = Outer_ring_damage[start_Outer_ring_damage: start_Outer_ring_damage+distance_Outer_ring_damage], Outer_ring_damage_label[start_Outer_ring_damage: start_Outer_ring_damage + distance_Outer_ring_damage]
                print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
                
                distance_Inner_ring_damage = int(0.6*len(Inner_ring_damage))
                start_Inner_ring_damage    = int(0.2*i*len(Inner_ring_damage))
                X_train_Inner_ring_damage, y_train_Inner_ring_damage = Inner_ring_damage[start_Inner_ring_damage: start_Inner_ring_damage + distance_Inner_ring_damage], Inner_ring_damage_label[start_Inner_ring_damage: start_Inner_ring_damage + distance_Inner_ring_damage]
                print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
                
                X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
                y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))
                print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
                
                print('\n'+ '-'*100)

                h = [a for a in range(len(Healthy)) if a not in range(start_Healthy, start_Healthy+distance_Healthy)]
                
                X_test_Healthy = Healthy[h]
                y_test_Healthy = Healthy_label[h]
                print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
                
                k = [a for a in range(len(Outer_ring_damage)) if a not in range(start_Outer_ring_damage, start_Outer_ring_damage+distance_Outer_ring_damage)]
                X_test_Outer_ring_damage = Outer_ring_damage[k]
                y_test_Outer_ring_damage = Outer_ring_damage_label[k]
                print(f'\n Shape of the Outer ring damage test data and label: {X_test_Outer_ring_damage.shape}, {y_test_Outer_ring_damage.shape}')
                
                l = [a for a in range(len(Inner_ring_damage)) if a not in range(start_Inner_ring_damage, start_Inner_ring_damage+distance_Inner_ring_damage)]
                X_test_Inner_ring_damage = Inner_ring_damage[l]
                y_test_Inner_ring_damage = Inner_ring_damage_label[l]
                print(f'\n Shape of the Inner ring damage test data and label: {X_test_Inner_ring_damage.shape}, {y_test_Inner_ring_damage.shape}')
                
                X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
                y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
                print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
                print('\n'+ '-'*100)
                
                # train_embs, test_embs = train_new_triplet_center(opt, X_train, y_train, X_test, y_test, CNN_C_trip, i) 
                emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, opt)
                X_train_embed, X_test_embed = emb_sys.get_emb()
                this_acc = []

                y_pred_all = []
                l = 0
                for each_ML in ['SVM', 'RF', 'KNN', 'LGBM', 'euclidean', 'cosine']:
                    acc = accuracy_score(y_test, y_pred)
                    acc = emb_sys.predict(X_test_embed, X_train_embed, ML_method=each_ML, use_mean_var=False, get_acc=True)
                    if each_ML == 'SVM':
                        emb_accuracy_SVM.append(acc)
                    elif each_ML == 'RF':
                        emb_accuracy_RF.append(acc)
                    elif each_ML == 'KNN':
                        emb_accuracy_KNN.append(acc)
                    elif each_ML == 'LGBM':
                        emb_accuracy_LGBM.append(acc)
                    elif each_ML == 'euclidean':
                        emb_accuracy_euclidean.append(acc)
                    elif each_ML == 'cosine':
                        emb_accuracy_cosine.append(acc)

                    print(f'\n--------------Test accuracy: {acc} with the {each_ML} method--------------')

                y_pred_all = y_pred_all.astype(np.float32) / l
                y_pred_all = np.argmax(y_pred_all, axis=1)
                acc_all = accuracy_score(y_test, y_pred_all)
                emb_accuracy_ensemble.append(acc_all)

                print(f'\n --------------Ensemble: {acc_all}--------------')
                print(color.GREEN + f'\n\t\t********* FINISHING ROUND {i} *********\n\n\n' + color.END)

    #------------------------------------------Case 1: no shuffle------------------------------------------------
    if opt.PU_data_table_10_case_1:
        comb = combinations([0, 1, 2, 3, 4], 3)
        
        # Print the obtained combinations
        for idx, i in enumerate(comb):
            i = list(i)
            # tf.keras.backend.clear_session()
            # gc.collect()
            if os.path.exists(join(path_saved_data, f'/X_train_table10_{i}.npy')):
                X_train = np.load(join(path_saved_data, f'/X_train_table10_{i}.npy'), mmap_mode="r")
                X_train_scaled = np.load(join(path_saved_data, f'/X_train_scaled_table10_{i}.npy'), mmap_mode="r")
                y_train = np.load(join(path_saved_data, f'/y_train_table10_{i}.npy'), mmap_mode="r")
            
                X_test = np.load(join(path_saved_data, f'/X_test_table10_{i}.npy'), mmap_mode="r")
                X_test_scaled = np.load(join(path_saved_data, f'/X_test_scaled_table10_{i}.npy'), mmap_mode="r")
                y_test = np.load(join(path_saved_data, f'/y_test_scaled_table10_{i}.npy'), mmap_mode="r")
            else:
                X_train_Healthy = Healthy[list(i)]
                y_train_Healthy = Healthy_label[list(i)]
                X_train_Healthy, y_train_Healthy = load_table_10_spe(X_train_Healthy, y_train_Healthy)
                X_train_Healthy_scaled = scaler_transform(X_train_Healthy, PowerTransformer)
                print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
                
                X_train_Outer_ring_damage, y_train_Outer_ring_damage = Outer_ring_damage[list(i)], Outer_ring_damage_label[list(i)]
                X_train_Outer_ring_damage, y_train_Outer_ring_damage = load_table_10_spe(X_train_Outer_ring_damage, y_train_Outer_ring_damage)
                X_train_Outer_ring_damage_scaled = scaler_transform(X_train_Outer_ring_damage, PowerTransformer)
                print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
            
            X_train_Inner_ring_damage, y_train_Inner_ring_damage = Inner_ring_damage[list(i)], Inner_ring_damage_label[list(i)]
            X_train_Inner_ring_damage, y_train_Inner_ring_damage = load_table_10_spe(X_train_Inner_ring_damage, y_train_Inner_ring_damage)
            X_train_Inner_ring_damage_scaled = scaler_transform(X_train_Inner_ring_damage, PowerTransformer)
            print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
            
            X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
            X_train_scaled = np.concatenate((X_train_Healthy_scaled, X_train_Outer_ring_damage_scaled, X_train_Inner_ring_damage_scaled))
            y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))
            with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_table10_{i}.npy', 'wb') as f:
                np.save(f, X_train)
            with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_scaled_table10_{i}.npy', 'wb') as f:
                np.save(f, X_train_scaled)
            with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train_table10_{i}.npy', 'wb') as f:
                np.save(f, y_train)
            
            print('\n'+ '-'*100)

            h = [a for a in range(len(Healthy)) if a not in list(i)]
            X_test_Healthy = Healthy[h]
            y_test_Healthy = Healthy_label[h]
            X_test_Healthy, y_test_Healthy = load_table_10_spe(X_test_Healthy, y_test_Healthy)
            X_test_Healthy_scaled = scaler_transform(X_test_Healthy, PowerTransformer)
            print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
            
            k = [a for a in range(len(Outer_ring_damage)) if a not in list(i)]
            X_test_Outer_ring_damage = Outer_ring_damage[k]
            y_test_Outer_ring_damage = Outer_ring_damage_label[k]
            X_test_Outer_ring_damage, y_test_Outer_ring_damage = load_table_10_spe(X_test_Outer_ring_damage, y_test_Outer_ring_damage)
            X_test_Outer_ring_damage_scaled = scaler_transform(X_test_Outer_ring_damage, PowerTransformer)  
            
            l = [a for a in range(len(Inner_ring_damage)) if a not in list(i)]
            X_test_Inner_ring_damage = Inner_ring_damage[l]
            y_test_Inner_ring_damage = Inner_ring_damage_label[l]
            X_test_Inner_ring_damage, y_test_Inner_ring_damage = load_table_10_spe(X_test_Inner_ring_damage, y_test_Inner_ring_damage)
            X_test_Inner_ring_damage_scaled = scaler_transform(X_test_Inner_ring_damage, PowerTransformer)
                    
            X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
            X_test_scaled = np.concatenate((X_test_Healthy_scaled, X_test_Outer_ring_damage_scaled, X_test_Inner_ring_damage_scaled))
            y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
            
            with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_table10_{i}.npy', 'wb') as f:
                np.save(f, X_test)
            with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_scaled_table10_{i}.npy', 'wb') as f:
                np.save(f, X_test_scaled)
            with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_test_scaled_table10_{i}.npy', 'wb') as f:
                np.save(f, y_test)

            print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
            print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
            print('\n'+ '-'*100)

            if opt.faceNet:
            print('\n Train phase...')
            # X_train = handcrafted_features(X_train)
            # X_test  = handcrafted_features(X_test)
            print(f'\n Length the handcrafted feature vector: {X_train.shape}')
            train_embs, test_embs, y_test_solf, y_train, outdir = train_new_triplet_center(opt, X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, CNN_C_trip, idx) 
            
            print('\n Saving embedding phase...')   
            this_acc = []

            y_pred_all = []
            count = 0
            for each_ML in ['SVM', 'RandomForestClassifier', 'LogisticRegression', 'GaussianNB', 'euclidean', 'cosine', 'KNN', 'BT']:
                model = FaceNetOneShotRecognitor(opt, X_train, y_train, X_test, y_test) 
                y_pred = model.predict(test_embs=test_embs, train_embs=train_embs, ML_method=each_ML, use_mean=False)
                y_pred_inv = np.argmax(y_pred, axis=1)
                acc = accuracy_score(y_test, y_pred_inv)
                plot_confusion(y_test, y_pred_inv, outdir, each_ML)
            
                if each_ML not in ['euclidean', 'cosine']:
                if y_pred_all == []:
                    y_pred_all = y_pred
                else:
                    y_pred_all += y_pred
                count += 1

                if each_ML == 'SVM':
                    emb_accuracy_SVM.append(acc)
                if each_ML == 'RandomForestClassifier':
                    emb_accuracy_RandomForestClassifier.append(acc)
                if each_ML == 'LogisticRegression':
                    emb_accuracy_LogisticRegression.append(acc)
                if each_ML == 'GaussianNB':
                    emb_accuracy_GaussianNB.append(acc)
                if each_ML == 'KNN':
                    emb_accuracy_KNN.append(acc)
                if each_ML == 'BT':
                    emb_accuracy_BT.append(acc)
                if each_ML == 'euclidean':
                    emb_accuracy_euclidean.append(acc)
                if each_ML == 'cosine':
                    emb_accuracy_cosine.append(acc)

                print(f'\n-------------- 1.Test accuracy: {acc} with the {each_ML} method--------------')

            y_pred_all = y_pred_all.astype(np.float32) / count
            y_pred_all = np.argmax(y_pred_all, axis=1)
            acc_all = accuracy_score(y_test, y_pred_all)
            emb_accuracy_ensemble.append(acc_all)
            print(f'\n --------------Ensemble: {acc_all}--------------')
        
        print(color.GREEN + f'\n\t\t********* FINISHING ROUND {idx} *********\n\n\n' + color.END)

    print(color.CYAN + 'FINISH!\n' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_SVM)} with SVM' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_RandomForestClassifier)} with RandomForestClassifier' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_LogisticRegression)} with LogisticRegression' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_GaussianNB)} with GaussianNB' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_KNN)} with KNN' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_BT)} with BT' + color.END)

    # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_SVM_no_emb)} with no embedding  SVM' + color.END)
    # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_RandomForestClassifier_no_emb)} with no embedding RandomForestClassifier' + color.END)
    # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_LogisticRegression_no_emb)} with no embedding LogisticRegression' + color.END)
    # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_GaussianNB_no_emb)} with no embedding GaussianNB' + color.END)
    # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_KNN_no_emb)} with no embedding KNN' + color.END)
    # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_BT_no_emb)} withno embedding  BT' + color.END)

    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_euclidean)} with euclidean' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_cosine)} with cosine' + color.END)

    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_ensemble)} with ensemble' + color.END)