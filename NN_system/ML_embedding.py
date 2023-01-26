from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from utils.tools import ML_models, scaler_fit, scale_test
from utils.extraction_features import extracted_feature_of_signal, handcrafted_features


class FaceNetOneShotRecognitor(object):
    def __init__(self, X_train, y_train, X_test, y_test, model, opt):
        self.opt = opt
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test   = X_test, y_test
        self.model = model

    def get_emb(self):
        # Scale data----------------------------------------
        X_tra_e = self.X_train
        X_te_e = self.X_test
        if self.opt.scaler != None:
            self.X_train, scale = scaler_fit(self.X_train, self.opt)
            self.X_test = scale_test(self.X_test, scale)

        if self.opt.model == 'main_model':
            # Extract data--------------------------------------
            if self.opt.Ex_feature == 'time':
                X_train_e = extracted_feature_of_signal(X_tra_e)
                X_test_e = extracted_feature_of_signal(X_te_e)
            if self.opt.Ex_feature == 'fre':
                X_train_e = handcrafted_features(X_tra_e)
                X_test_e = handcrafted_features(X_te_e)
            if self.opt.Ex_feature == 'time_fre':
                X_train_time_e = extracted_feature_of_signal(X_tra_e)
                X_train_fre_e = handcrafted_features(X_tra_e)
                X_train_e = np.concatenate((X_train_time_e, X_train_fre_e), axis=-1)

                X_test_time_e = extracted_feature_of_signal(X_te_e)
                X_test_fre_e = handcrafted_features(X_te_e)
                X_test_e = np.concatenate((X_test_time_e, X_test_fre_e), axis=-1)
            
            if self.opt.scaler != None:
                X_train_e, scale = scaler_fit(X_train_e, self.opt)
                X_test_e = scale_test(X_test_e, scale)
            
            _, X_train_embed = self.model.predict([self.X_train, X_train_e])
            _soft_pred, X_test_embed = self.model.predict([self.X_test, X_test_e])
        
        if self.opt.model == 'S_SDLM':
            _, X_train_embed = self.model.predict([self.X_train])
            _soft_pred, X_test_embed = self.model.predict([self.X_test])

        return X_train_embed, X_test_embed
      
    def predict(self, test_embs, train_embs, ML_method=None, use_mean_var=False):
        print('\n Test embs shape: ', test_embs.shape)
        print('Train embs shape: ', train_embs.shape)
        
        if ML_method == 'euclidean' or ML_method == 'cosine':
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
                else:
                    for j in range(train_embs.shape[0]):
                        if ML_method == 'euclidean':
                            distances.append(euclidean(test_embs[i].reshape(-1), train_embs[j])) # append one value
                        elif ML_method == 'cosine':
                            distances.append(cosine(test_embs[i].reshape(-1), train_embs[j]))
                test_pred.append(self.y_train[np.argsort(distances)[0]])   

            print(f"\nTEST ACCURACY: {accuracy_score(test_pred, self.y_test)}\n")   
        else:
            ML_models(self.X_train, self.y_train, self.X_test, self.y_test, self.opt)