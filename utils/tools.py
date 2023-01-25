import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import os
import scipy.io
from os.path import join
from sklearn.metrics import accuracy_score

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

def scale_test(test_signals, scale):
  return scale.transform(test_signals)

def TSNE_plot(x, y, title, save_path):
  tsne = TSNE(n_components=2, verbose=1, random_state=123)
  x = tsne.fit_transform(x) 
  df = pd.DataFrame()
  df["y"] = y
  df["comp-1"] = z[:,0]
  df["comp-2"] = z[:,1]

  sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                  palette=sns.color_palette("hls", 3),
                  data=df).set(title=title) 
  sns.savefig(save_path)

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

def ML_models(X_train, y_train, X_test, y_test, opt):
    '''
    X_train, X_test: matrices
    y_train, y_test: matrices (onehot)
    '''
    if opt.ML_method == 'SVM': # SVM, RandomForestClassifier, LogisticRegression, GaussianNB
      model = SVC(kernel='rbf', probability=True)
    if opt.ML_method == 'RF':
      model = RandomForestClassifier()
    if opt.ML_method == 'KNN':     
      model = KNeighborsClassifier()
    if opt.ML_method == 'LGBM':
      model = LGBMClassifier()
    
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    print(f"\nTEST ACCURACY: {accuracy_score(y_test, y_test_pred)}\n")

def load_PU_data(path, opt):
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

def one_hot(label):
  n_class = np.max(label) + 1
  new_label = np.zeros((len(label), n_class))
  for idx, val in enumerate(label):
    new_label[idx, val] = 1
  return new_label
    
    
    