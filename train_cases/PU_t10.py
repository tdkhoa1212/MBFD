from NN_system.ML_embedding import FaceNetOneShotRecognitor
from utils.PU import load_PU_data_10
from utils.tools import load_table_10_spe,\
                        plot_confusion, scaler_fit

from itertools import combinations
from sklearn.utils import shuffle
from NN_system.main_system import train_main_system
import tensorflow as tf
import numpy as np
import os
from os.path import join
from sklearn.metrics import accuracy_score
import warnings
from NN_system.main_system import train_main_system

warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(all="ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
def get_list(data, lis):
    all_ = [data[i] for i in lis]
    return all_
    
def train_table_9(opt):
    case_0 = False 
    case_1 = True 

    print('\nLoading labels...\n')
    Healthy, Outer_ring_damage, Inner_ring_damage = load_PU_data_10(opt)
    Healthy_label           = np.array([0]*len(Healthy))
    Outer_ring_damage_label = np.array([1]*len(Outer_ring_damage))
    Inner_ring_damage_label = np.array([2]*len(Inner_ring_damage))

    # ###################################### LOAD DATA ######################################
    # Healthy, Healthy_label = load_table_10_spe(Healthy, Healthy_label)
    # Outer_ring_damage, Outer_ring_damage_label = load_table_10_spe(Outer_ring_damage, Outer_ring_damage_label)
    # Inner_ring_damage, Inner_ring_damage_label = load_table_10_spe(Inner_ring_damage, Inner_ring_damage_label)
    # if os.path.exists(join(opt.path_saved_data, '/Healthy_10.npy')):
    #     Healthy = np.load(join(opt.path_saved_data, '/Healthy_10.npy'), mmap_mode="r")  
    #     Outer_ring_damage = np.load(join(opt.path_saved_data, '/Outer_ring_damage_10.npy'), mmap_mode="r")
    #     Inner_ring_damage = np.load(join(opt.path_saved_data, '/Inner_ring_damage_10.npy'), mmap_mode="r")
    # else: 
    #     with open(join(opt.path_saved_data, '/Healthy_10.npy'), 'wb') as f:
    #         np.save(f, Healthy)
    #     with open(join(opt.path_saved_data, '/Outer_ring_damage_10.npy'), 'wb') as f:
    #         np.save(f, Outer_ring_damage)
    #     with open(join(opt.path_saved_data, '/Inner_ring_damage_10.npy'), 'wb') as f:
    #         np.save(f, Inner_ring_damage)

    # ###################################### PROCESS ######################################
    # np.random.seed(0)
    # Healthy, Healthy_label = shuffle(Healthy, Healthy_label, random_state=0)
    # Outer_ring_damage, Outer_ring_damage_label = shuffle(Outer_ring_damage, Outer_ring_damage_label, random_state=0)
    # Inner_ring_damage, Inner_ring_damage_label = shuffle(Inner_ring_damage, Inner_ring_damage_label, random_state=0)

    print(color.GREEN + '\n\n\t *************START*************\n\n' + color.END)
    emb_accuracy_SVM = []
    emb_accuracy_RF = []
    emb_accuracy_KNN = []
    emb_accuracy_LGBM = []
    emb_accuracy_euclidean = []
    emb_accuracy_cosine = []

    emb_accuracy_ensemble = []

    #------------------------------------------Case 0: shuffle------------------------------------------------
    if case_0:
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

            print("\n" + "#"*20 + ' TRAINING PHASE ' + "#"*20 + "\n")
            model, scale_1, scale_2 = train_main_system(X_train, y_train, X_test, y_test, opt)
            emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, scale_1, scale_2, opt)
            X_train_embed, X_test_embed = emb_sys.get_emb()

            y_pred_all = []
            l = 0
            for each_ML in ['SVM', 'RF', 'KNN', 'LGBM', 'euclidean', 'cosine']:
                l += 1
                y_pred = emb_sys.predict(X_test_embed, X_train_embed, ML_method=each_ML, use_mean_var=False, get_pred=True)
                y_pred_all.append(y_pred)
                acc = accuracy_score(y_pred, y_test)
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

            print(color.GREEN + f'\n\t\t********* FINISHING ROUND {idx} *********\n\n\n' + color.END)

    #------------------------------------------Case 1: similar to the original paper------------------------------------------------
    if case_1:
        comb = combinations([0, 1, 2, 3, 4], 3)
        
        # Print the obtained combinations
        for idx, i in enumerate(comb):
            i = list(i)
            tf.keras.backend.clear_session()
            # gc.collect()
            if os.path.exists(join(opt.path_saved_data, f'X_train_table10_{i}.npy')):
                X_train = np.load(join(opt.path_saved_data, f'X_train_table10_{i}.npy'), mmap_mode="r")
                y_train = np.load(join(opt.path_saved_data, f'y_train_table10_{i}.npy'), mmap_mode="r")
            
                X_test = np.load(join(opt.path_saved_data, f'X_test_table10_{i}.npy'), mmap_mode="r")
                y_test = np.load(join(opt.path_saved_data, f'y_test_scaled_table10_{i}.npy'), mmap_mode="r")
            else:
                X_train_Healthy, y_train_Healthy = get_list(Healthy, i), get_list(Healthy_label, i) 
                X_train_Healthy, y_train_Healthy = load_table_10_spe(X_train_Healthy, y_train_Healthy)
                print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
                
                X_train_Outer_ring_damage, y_train_Outer_ring_damage = get_list(Outer_ring_damage, i), get_list(Outer_ring_damage_label, i) 
                X_train_Outer_ring_damage, y_train_Outer_ring_damage = load_table_10_spe(X_train_Outer_ring_damage, y_train_Outer_ring_damage)
                print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
            
                X_train_Inner_ring_damage, y_train_Inner_ring_damage = get_list(Inner_ring_damage, i), get_list(Inner_ring_damage_label, i)  
                X_train_Inner_ring_damage, y_train_Inner_ring_damage = load_table_10_spe(X_train_Inner_ring_damage, y_train_Inner_ring_damage)
                print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
                
                X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
                y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))

                with open(join(opt.path_saved_data,  f'X_train_table10_{i}.npy'), 'wb') as f:
                    np.save(f, X_train)

                with open(join(opt.path_saved_data, f'y_train_table10_{i}.npy'), 'wb') as f:
                    np.save(f, y_train)
                
            print('\n'+ '-'*100 + '\n')

            h = [a for a in range(len(Healthy)) if a not in list(i)]
            X_test_Healthy = get_list(Healthy, h)
            y_test_Healthy = get_list(Healthy_label, h)
            X_test_Healthy, y_test_Healthy = load_table_10_spe(X_test_Healthy, y_test_Healthy)
            print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
            
            k = [a for a in range(len(Outer_ring_damage)) if a not in list(i)]
            X_test_Outer_ring_damage = get_list(Outer_ring_damage, k)
            y_test_Outer_ring_damage = get_list(Outer_ring_damage_label, k)
            X_test_Outer_ring_damage, y_test_Outer_ring_damage = load_table_10_spe(X_test_Outer_ring_damage, y_test_Outer_ring_damage)
            print(f'\n Shape of the outer ring damage test data and label: {X_test_Outer_ring_damage.shape}, {y_test_Outer_ring_damage.shape}')

            l = [a for a in range(len(Inner_ring_damage)) if a not in list(i)]
            X_test_Inner_ring_damage = get_list(Inner_ring_damage, l)
            y_test_Inner_ring_damage = get_list(Inner_ring_damage_label, l)
            X_test_Inner_ring_damage, y_test_Inner_ring_damage = load_table_10_spe(X_test_Inner_ring_damage, y_test_Inner_ring_damage)
            print(f'\n Shape of the inter ring damage test data and label: {X_test_Inner_ring_damage.shape}, {y_test_Inner_ring_damage.shape}')

            X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
            y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
            
            with open(join(opt.path_saved_data, f'X_test_table10_{i}.npy'), 'wb') as f:
                np.save(f, X_test)
            with open(join(opt.path_saved_data, f'y_test_scaled_table10_{i}.npy'), 'wb') as f:
                np.save(f, y_test)

            print('\n'+ '-'*100)
            print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
            print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
            print('\n'+ '-'*100)

            print("\n" + "#"*20 + ' TRAINING PHASE ' + "#"*20 + "\n")
            model, scale_1, scale_2 = train_main_system(X_train, y_train, X_test, y_test, opt)
            emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, scale_1, scale_2, opt)
            X_train_embed, X_test_embed = emb_sys.get_emb()

            y_pred_all = []
            l = 0
            for each_ML in ['SVM', 'RF', 'KNN', 'LGBM', 'euclidean', 'cosine']:
                l += 1
                tf.keras.backend.clear_session()
                y_pred = emb_sys.predict(X_test_embed, X_train_embed, ML_method=each_ML, use_mean_var=False, get_pred=True)
                
                y_pred_all.append(y_pred)
                acc = accuracy_score(y_pred, y_test)
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

            print(color.GREEN + f'\n\t\t********* FINISHING ROUND {idx} *********\n\n\n' + color.END)

    print(color.CYAN + 'FINISH!\n' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_SVM)} with SVM' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_RF)} with RF' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_KNN)} with KNN' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_LGBM)} with LGBM' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_euclidean)} with euclidean' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_cosine)} with cosine' + color.END)
    print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_ensemble)} with ensemble' + color.END)