import numpy as np
from Networks.SDLM_model import SDLM
from utils.PU import load_PU_table
from utils.tools import scaler_fit, ML_models, scale_test, one_hot, TSNE_plot, one_hot_inverse
from utils.extraction_features import extracted_feature_of_signal, handcrafted_features
from NN_system.ML_embedding import FaceNetOneShotRecognitor
from NN_system.main_system import train_main_system, train_S_SDLM_system, train_U_SDLM_system
from Networks.SDLM_model import SDLM
from os.path import join
from tensorflow.keras.models import Model
import tensorflow as tf

def train_table_6(opt):
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

def train_table_7(opt):
    X_train, y_train, X_test, y_test = load_PU_table(opt)
    print('\n' + f'Shape of original training data: {X_train.shape, y_train.shape}')
    print(f'Shape of original test data: {X_test.shape, y_test.shape}' + '\n')
    model, scale_1, scale_2 = train_main_system(X_train, y_train, X_test, y_test, opt)
    emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, scale_1, scale_2, opt)
    X_train_embed, X_test_embed = emb_sys.get_emb()
    
    if opt.model == 'main_model':
        for each_ML in opt.ML_method:
            tf.keras.backend.clear_session()
            print("-"*10 + each_ML + "-"*10)
            emb_sys.predict(X_test_embed, X_train_embed, ML_method=each_ML, use_mean_var=False)
            if opt.TSNE_plot:
                TSNE_plot(X_test_embed[:, :opt.embedding_size], y_test, f'Test data with {opt.scaler} and {each_ML}', join(opt.weights_path, 'images/', f'{opt.model}_{opt.scaler}_{each_ML}.png'))
            print("-"*20, '\n')

    if opt.model == 'SDLM':
        y_test = one_hot(y_test)
        y_train = one_hot(y_train)
        model = SDLM(opt)
        model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc']) # loss='mse'
        model.summary()
        weight_path = join(opt.weights_path, opt.model)

        if opt.load_weights:
          model.load_weights(weight_path)
        else:
          history = model.fit(X_train, y_train,
                              epochs     = opt.epochs,
                              batch_size = opt.batch_size,
                              validation_data=(X_test, y_test),)
          print("\nSave weights sucessful!")
          print(f"Path {weight_path}\n")
          model.save(weight_path)
        if opt.get_SDLM_extract:
            model = Model(inputs=model.inputs,
                          outputs=model.get_layer(name="Un_output").output)
            y_train = one_hot_inverse(y_train)
            y_test = one_hot_inverse(y_test)
            emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, opt)
            X_train_embed, X_test_embed = emb_sys.get_emb()
            emb_sys.predict(X_test_embed, X_train_embed, ML_method=opt.ML_method, use_mean_var=False)
        else:
            _, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print('\n' + '*'*20 + f' Test accuracy: {test_acc}' + '*'*20)
    
    if opt.model == 'S_SDLM':
        model = train_S_SDLM_system(X_train, y_train, X_test, y_test, opt)
        emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, opt)
        X_train_embed, X_test_embed = emb_sys.get_emb()
        emb_sys.predict(X_test_embed, X_train_embed, ML_method=opt.ML_method, use_mean_var=False)
        if opt.TSNE_plot:
            TSNE_plot(X_test_embed, y_test, f'Test data with {opt.scaler}', join(opt.weights_path, 'image/', f'{opt.model}_{opt.scaler}.png'))

    if opt.model == 'U_SDLM':
        model = train_U_SDLM_system(X_train, y_train, X_test, y_test, opt)
        emb_sys = FaceNetOneShotRecognitor(X_train, y_train, X_test, y_test, model, opt)
        X_train_embed, X_test_embed = emb_sys.get_emb()
        emb_sys.predict(X_test_embed, X_train_embed, ML_method=opt.ML_method, use_mean_var=False)
        if opt.TSNE_plot:
            TSNE_plot(X_test_embed, y_test, f'Test data with {opt.scaler}', join(opt.weights_path, 'image/', f'{opt.model}_{opt.scaler}.png'))
