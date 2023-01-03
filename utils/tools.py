import numpy as np
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

def ML_models(X_train, y_train, X_test, y_test opt):
    '''
    X_train, X_test: matrices
    y_train, y_test: matrices (onehot)
    '''
    if opt.ML_method == 'SVM': # SVM, RandomForestClassifier, LogisticRegression, GaussianNB
      model = SVC(kernel='rbf', probability=True)
    if opt.ML_method == 'RF':
      model = RandomForestClassifier(n_estimators= 300, max_features = "sqrt", n_jobs = -1, random_state = 38)
    if opt.ML_method == 'KNN':     
      model = KNeighborsClassifier(random_state=1)
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
      path_signal = path + '/' + name
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

    
    
    