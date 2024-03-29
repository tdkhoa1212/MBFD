import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import os
import scipy.io
import tensorflow as tf
from os.path import join
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------Scaler methods----------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# ----------------------------------------------------ML Models----------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def scale_test(test_signals, scale):
  return scale.transform(test_signals)

def TSNE_plot(x, y, title, save_path):
  tsne = TSNE(n_components=2, verbose=1, random_state=123)
  z = tsne.fit_transform(x) 
  df = pd.DataFrame()
  df["y"] = y
  df["comp-1"] = z[:,0]
  df["comp-2"] = z[:,1]

  sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                  palette=sns.color_palette("hls", 3),
                  data=df).set(title=title) 
  plt.savefig(save_path)


def scaler_tripdata(a, p, n, opt):
    length = len(a)
    all_ = np.concatenate((a, p, n), axis=0)
    all_, scale = scaler_fit(all_, opt)
    return  all_[:length, :], all_[length: length*2, :], all_[length*2: , :], scale


def scaler_fit(train_signals, opt):
    '''
    train_signals: a matrix 
    scaler_method: name of scaler method
    '''
    if opt.scaler == 'MinMaxScaler':
      scaler = MinMaxScaler()
    if opt.scaler == 'MaxAbsScaler':
      scaler = MaxAbsScaler()
    if opt.scaler == 'StandardScaler':
      scaler = StandardScaler()
    if opt.scaler == 'RobustScaler':
      scaler = RobustScaler()
    if opt.scaler == 'Normalizer':
      scaler = Normalizer()
    if opt.scaler == 'QuantileTransformer':
      scaler = QuantileTransformer()
    if opt.scaler == 'PowerTransformer':
      scaler = PowerTransformer()
    print('\n' + 10*'-' + f'{opt.scaler}' + 10*'-' + '\n')
    scale = scaler.fit(train_signals)
    train_data = scale.transform(train_signals)
    return train_data, scale

def predict_batch(model, data, batch):
  pred = []
  for i in range(0, len(data), batch):
    data_batch = data[i: i+batch]
    if pred == []:
      pred = model.predict(data_batch)
    else:
      pred = np.concatenate((pred, model.predict(data_batch)))
  return pred


def ML_models(X_train, y_train, X_test, y_test, ML_method, get_pred=False):
    '''
    X_train, X_test: matrices
    y_train, y_test: matrices (onehot)
    '''
    if ML_method == 'SVM': # SVM, RandomForestClassifier, LogisticRegression, GaussianNB
      model = SVC(kernel='rbf', probability=True)
    if ML_method == 'RF':
      model = RandomForestClassifier()
    if ML_method == 'KNN':     
      model = KNeighborsClassifier()
    
    model.fit(X_train, y_train)
    y_test_pred = predict_batch(model, X_test, 20)
    if get_pred:
      return y_test_pred
    print(f"\nTEST ACCURACY: {accuracy_score(y_test, y_test_pred)}\n")

def load_PU_data(path, opt, pre_min_l=None):
  data = []
  all_data = []
  min_l = 0
  for name in os.listdir(path):
    if name.split('.')[-1] == 'mat':
      path_signal = join(path, name)
      file_name = path_signal.split('/')[-1]
      name = file_name.split('.')[0]
      signal = scipy.io.loadmat(path_signal)[name]
      
      if opt.type_data == 'vibration':
        signal = signal[0][0][2][0][6][2]  
      if opt.type_data == 'MCS1':
        signal = signal[0][0][2][0][1][2]
      if opt.type_data == 'MCS2':
        signal = signal[0][0][2][0][2][2]
      signal = signal.reshape(-1, )
           
      if min_l == 0:
        min_l = int(signal.shape[0])
      elif min_l > signal.shape[0]:
        min_l = int(signal.shape[0])   
      all_data.append(signal)
  
  if pre_min_l != None:
    min_l = pre_min_l
  for i in all_data:
    each_data = i[:min_l].tolist()
    data.append(each_data)
  return np.array(data)

def load_table_10(path):
  all_data = []
  for name in os.listdir(path):
    each_path = os.path.join(path, name)
    each_data = np.expand_dims(np.load(each_path)[:255900], axis=0)
    if all_data == []:
      all_data = each_data
    else:
      all_data = np.concatenate((all_data, each_data))
      
  return np.expand_dims(all_data, axis=0)

def load_table_10_spe(data, label):
  new_data = []
  new_label = []
  for idx, each_data in enumerate(data):
    each_data = np.squeeze(each_data)
    if new_data == []:
      new_data = each_data
    else:
      new_data = np.concatenate((new_data, each_data))
    
    each_label = label[idx]
    each_label = [each_label]*len(each_data)
    if new_label == []:
      new_label = each_label
    else:
      new_label = np.concatenate((new_label, each_label))
  return np.array(new_data), np.array(new_label)

def one_hot_convert(label, n_class):
  new_label = np.zeros((len(label), n_class))
  for idx, val in enumerate(label):
    new_label[idx, val] = 1
  return new_label

def one_hot_inverse(label):
  label_1D = []
  for i in label:
    label_1D.append(np.argmax(i))
  return np.array(label_1D)
    
def plot_confusion(y_test, y_pred_inv, outdir, each_ML):
   commands = ['Healthy', 'OR Damage', 'IR Damage']
   confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_inv)

   plt.figure(figsize=(10, 8))
   sns.heatmap(confusion_mtx,
             xticklabels=commands,
             yticklabels=commands,
             annot=True, fmt='g')
   plt.xlabel('Prediction')
   plt.ylabel('Label')
   plt.savefig(os.path.join(outdir, each_ML))
   plt.show()

def divide_sample(x=None, window_length=400, hop_length=200):
  '''
  The shape of x must be (n_sample, )
  '''
  if len(x.shape) > 1:
    x = x.reshape(-1, )
  a = []
  window = 0
  all_hop_length = 0
  num_window = (x.shape[0]-(window_length-hop_length))//hop_length
  while window < num_window:
    if len(x[all_hop_length: all_hop_length+window_length])==window_length:
      a.append(x[all_hop_length: all_hop_length+window_length])
    all_hop_length += hop_length
    window += 1
  return np.array(a)

def concatenate_data(x=None, window_length=400, hop_length=200, hand_fea=True, SNdb=10):
  X_train_all = []
  X_test = []
  for idx, i in enumerate(x):
    if len(x[i]) > 80:
      if X_train_all == []:
        data = x[i].reshape(-1, 1)
        X_train_all, X_test = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
      else:
        data = x[i].reshape(-1, 1)
        each_X_train_all, each_X_test = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)

        X_train_all = np.concatenate((X_train_all, each_X_train_all), axis=0)
        X_test = np.concatenate((X_test, each_X_test), axis=0)
  
  X_train_all = divide_sample(X_train_all, window_length, hop_length)
  X_test = divide_sample(X_test, window_length, hop_length)
  
  return X_train_all, X_test

def one_hot(pos, num_class):
    num = np.zeros((1, num_class))
    num[0, pos] = 1
    return num

def convert_one_hot(x, state=True):
    if state == False:
      index = None
      x = np.squeeze(x)

      for idx, i in enumerate(x):
        if i == 1:
          index = idx
      return [index]
    else:
      return x.tolist()