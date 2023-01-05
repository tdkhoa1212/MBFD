from scipy.spatial.distance import cosine, euclidean
from train import parse_opt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from utils.tools import ML_models

opt = parse_opt()

class FaceNetOneShotRecognitor(object):
    def __init__(self, opt, X_train, y_train, X_test, y_test, model):
        self.opt = opt
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test   = X_test, y_test
        self.model = model
    
    def emb_test(self, input_data):
        pd = []
        if len(input_data) == 1: # Only one test input data
            pd = np.array(self.model(input_data))
        elif len(input_data) > 1: # Multi test input data
            for start in tqdm(range(0, len(input_data), self.opt.batch_size)):
                embeddings = self.model(input_data[start: start+self.opt.batch_size])
                if pd == []:
                    pd = embeddings
                else:
                    pd = np.concatenate((pd, embeddings))
        return pd
      
    def predict(self, test_embs, train_embs, ML_method=None, emb=True, use_mean_var=False):
        print('\n Test embs shape: ', test_embs.shape)
        print('Train embs shape: ', train_embs.shape)
        
        if ML_method == 'Euclidean' or ML_method == 'Cosine':
            print(f'Classification method: {ML_method}')
            test_pred = []

            if use_mean_var:
                class_names = np.unique(self.y_train_all)
                train_emb_all = []
                for i in class_names:
                    mean_class = np.mean(train_embs[self.y_train_all==i], axis=0)
                    var_class  = np.var(train_embs[self.y_train_all==i], axis=0)
                    emb_class  = np.concatenate((mean_class, var_class))
                    train_emb_all.append(emb_class)

            num = test_embs.shape[0]  # number of test data
            for i in range(num): 
                # each test data is compared with each training data in training loop
                distances = []
                if use_mean_var:
                    for train_emb in train_emb_all:
                        # testing embbed---------------------------------------
                        test_emb = test_embs[i]
                        test_mean = np.mean(test_emb, axis=0)
                        test_var = np.var(test_emb, axis=0)

                        # training embbed----------------------------------------
                        half = train_emb.shape[0]//2
                        train_mean, train_var = train_emb[ :half], train_emb[half: ]
        
                      # emb_test_each = np.concatenate((test_emb, test_var))
                        if ML_method == 'euclidean':
                            distances.append(euclidean(train_mean, test_mean) + euclidean(train_var, test_var)) # append one value
                        elif ML_method == 'cosine':
                            distances.append(cosine(train_mean, test_mean) + cosine(train_var, test_var))

                    test_pred.append(np.argsort(distances)[0])
                else:
                    for j in range(train_embs.shape[0]):
                        if ML_method == 'euclidean':
                            distances.append(euclidean(test_embs[i].reshape(-1), train_embs[j])) # append one value
                        elif ML_method == 'cosine':
                            distances.append(cosine(test_embs[i].reshape(-1), train_embs[j]))
                    test_pred.append(np.argsort(distances)[0])   

            print(f"\nTEST ACCURACY: {accuracy_score(test_pred, self.y_test)}\n")   
        else:
            ML_models(self.X_train, self.y_train, self.X_test, self.y_test, self.opt)