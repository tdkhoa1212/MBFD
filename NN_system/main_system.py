from tensorflow.keras.layers import concatenate, Input
from Networks.center_model import center
from Networks.S_SDLM_model import S_SDLM
from Networks.U_SDLM_model import U_SDLM
from utils.triplet import new_triplet_loss, generate_triplet, triplet_loss
from utils.extraction_features import extracted_feature_of_signal, handcrafted_features
from utils.tools import one_hot, scaler_fit, scale_test
from os.path import isdir, join
from tensorflow.keras.models import Model
from tensorflow.saved_model import save
import numpy as np
import tensorflow as tf
from utils.angular_grad import AngularGrad

def train_main_system(X_train, y_train, X_test, y_test, opt):   
    # Expand 1 channel for data ------------------------------
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Extract model ---------------------------------------------------------
    e_i_1 = Input((opt.e_input_shape, ), name='extracting_input_1')
    e_o_1  = U_SDLM(e_i_1, opt)
    e_model_1 = Model(inputs=[e_i_1], outputs=[e_o_1])
    e_y_1 = e_model_1([e_i_1])

    e_i_2 = Input((opt.e_input_shape, ), name='extracting_input_2')
    e_o_2  = U_SDLM(e_i_2, opt)
    e_model_2 = Model(inputs=[e_i_2], outputs=[e_o_2])
    e_y_2 = e_model_2([e_i_2])
    
    e_i_3 = Input((opt.e_input_shape, ), name='extracting_input_3')
    e_o_3  = U_SDLM(e_i_3, opt)
    e_model_3 = Model(inputs=[e_i_3], outputs=[e_o_3])
    e_y_3 = e_model_3([e_i_3])

     # Center model----------------------------------------------------------
    c_i     = Input((1,), name='center_input')
    c_o     = center(c_i, opt)
    c_model = Model(inputs=[c_i], outputs=[c_o])
    c_y     = c_model([c_i])

    # Triplet model----------------------------------------------------------
    t_i = Input(shape=(opt.input_shape, 1), name='Triplet_model')
    softmax, logits = S_SDLM(t_i, opt)
    t_model = Model(inputs=[t_i], outputs=[softmax, logits])
  
    a_i = Input((opt.input_shape, 1), name='anchor_input')
    p_i = Input((opt.input_shape, 1), name='positive_input')
    n_i = Input((opt.input_shape, 1), name='negative_input')
    
    soft_a, logits_a = t_model([a_i])
    soft_p, logits_p = t_model([p_i])
    soft_n, logits_n = t_model([n_i])

    m_logit  = concatenate([logits_a, e_y_1, logits_p, e_y_2, logits_n, e_y_3, c_y], axis=-1, name='merged_logit_output')
    m_soft = concatenate([soft_a, soft_p, soft_n], axis=-1, name='merged_soft_ouput')
    
    loss_weights = [1, 0.01]
    path = join(opt.weights_path, "main_SDLM")

    if opt.train_mode:
        # ------------------------------------- GENERATE DATA ---------------------------------------------------------
        # Data of main branch
        X_train, y_train = generate_triplet(X_train, y_train)  #(anchors, positive, negative)
        X_train_e =  X_train
        
        if opt.scaler != None:
            X_train, scale_1 = scaler_fit(X_train, opt)
            X_test = scale_test(X_test, scale)

        a_data = X_train[:, 0].reshape(-1, opt.input_shape, 1)
        p_data = X_train[:, 1].reshape(-1, opt.input_shape, 1)
        n_data = X_train[:, 2].reshape(-1, opt.input_shape, 1)

        a_data_e = X_train_e[:, 0]
        p_data_e = X_train_e[:, 1]
        n_data_e = X_train_e[:, 2]

        if opt.Ex_feature == 'time':
            e_a_data = extracted_feature_of_signal(a_data_e)
            e_p_data = extracted_feature_of_signal(p_data_e)
            e_n_data = extracted_feature_of_signal(n_data_e)


        if opt.Ex_feature == 'fre':
            e_a_data = handcrafted_features(a_data_e)
            e_p_data = handcrafted_features(p_data_e)
            e_n_data = handcrafted_features(n_data_e)

        if opt.Ex_feature == 'time_fre':
            a_time   = extracted_feature_of_signal(a_data_e)
            p_time   = extracted_feature_of_signal(p_data_e)
            n_time   = extracted_feature_of_signal(n_data_e)

            a_fre   = handcrafted_features(a_data_e)
            p_fre   = handcrafted_features(p_data_e)
            n_fre   = handcrafted_features(n_data_e)

            e_a_data = np.concatenate((a_time, a_fre), axis=-1)
            e_p_data = np.concatenate((p_time, p_fre), axis=-1)
            e_n_data = np.concatenate((n_time, n_fre), axis=-1)

        if opt.scaler != None:
            length = len(e_a_data)
            all_ = np.concatenate((e_a_data, e_p_data, e_n_data), axis=0)
            all_, scale_2 = scaler_fit(all_, opt)
            e_a_data, e_p_data, e_n_data = all_[:length, :], all_[length: length*2, :], all_[length*2: , :]
        #-----------------------------------------------

        a_label = one_hot(y_train[:, 0])
        p_label = one_hot(y_train[:, 1])
        n_label = one_hot(y_train[:, 2])
        c_data   = y_train[:, 1]

        t_soft = np.concatenate((a_label, p_label, n_label), -1)

        model = Model(inputs=[a_i, e_i_1, p_i, e_i_2, n_i, e_i_3, c_i], outputs=[m_soft, m_logit])
        model.summary()
        if opt.load_weights:
            if isdir(path):
                model.load_weights(path)
                print(f'\n Load weight : {path}')
            else:
                print('\n No weight file.')

        model.compile(loss=["categorical_crossentropy",
                    new_triplet_loss],
                    optimizer=tf.keras.optimizers.experimental.RMSprop(), 
                    metrics=["accuracy"], 
                    loss_weights=loss_weights)

        # Note:
        # y=[t_soft, c_data] c_data is just for afternative position for blank position
        # only use t_soft for softmax head in training process
        model.fit(x = [a_data, e_a_data, p_data, e_p_data, n_data, e_n_data, c_data], 
                  y = [t_soft, c_data],
                  batch_size=opt.batch_size, 
                  epochs=opt.epochs, 
                #   callbacks=[callback], 
                  shuffle=True)

        save(model, path)

    # ------------------------------------- TEST MODEL ---------------------------------------------------------
    model = Model(inputs=[a_i, e_i_1], outputs=[soft_a, logits_a])
    model.load_weights(path)
    return model

def train_S_SDLM_system(X_train, y_train, X_test, y_test, opt):
    if opt.scaler != None:
      X_train, scale = scaler_fit(X_train, opt)
      X_test = scale_test(X_test, scale)

    # Expand 1 channel for data ------------------------------
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Triplet model----------------------------------------------------------
    t_i = Input(shape=(opt.input_shape, 1), name='Triplet_model')
    softmax, logits = S_SDLM(t_i, opt)
    t_model = Model(inputs=[t_i], outputs=[softmax, logits])
  
    a_i = Input((opt.input_shape, 1), name='anchor_input')
    p_i = Input((opt.input_shape, 1), name='positive_input')
    n_i = Input((opt.input_shape, 1), name='negative_input')

    soft_a, logits_a = t_model([a_i])
    soft_p, logits_p = t_model([p_i])
    soft_n, logits_n = t_model([n_i])

    m_logit = concatenate([logits_a, logits_p, logits_n], axis=-1, name='merged_logit_output')
    m_soft  = concatenate([soft_a, soft_p, soft_n], axis=-1, name='merged_soft_ouput')
    
    loss_weights = [1, 0.01]
    path = join(opt.weights_path, "S_SDLM")

    if opt.train_mode:
        # ------------------------------------- GENERATE DATA ---------------------------------------------------------
        # Data of main branch
        X_train, y_train = generate_triplet(X_train, y_train)  #(anchors, positive, negative)
        a_data = X_train[:, 0].reshape(-1, opt.input_shape, 1)
        p_data = X_train[:, 1].reshape(-1, opt.input_shape, 1)
        n_data = X_train[:, 2].reshape(-1, opt.input_shape, 1)

        a_label = one_hot(y_train[:, 0])
        p_label = one_hot(y_train[:, 1])
        n_label = one_hot(y_train[:, 2])

        t_soft = np.concatenate((a_label, p_label, n_label), -1)

        model = Model(inputs=[a_i, p_i, n_i], outputs=[m_soft, m_logit])
        model.summary()
        
        if opt.load_weights:
            if isdir(path):
                model.load_weights(path)
                print(f'\n Load weight : {path}')
            else:
                print('\n No weight file.')

        model.compile(loss         = ["categorical_crossentropy", triplet_loss],
                    optimizer    = tf.keras.optimizers.experimental.RMSprop(), 
                    metrics      = ["accuracy"], 
                    loss_weights = loss_weights)

        # Note:
        # y=[t_soft, c_data] c_data is just for afternative position for blank position
        # only use t_soft for softmax head in training process
        model.fit(x=[a_data, p_data, n_data], y=[t_soft, t_soft],
                batch_size=opt.batch_size, 
                epochs=opt.epochs, 
                # callbacks=[callback], 
                shuffle=True)

        save(model, path)

    # ------------------------------------- TEST MODEL ---------------------------------------------------------
    model = Model(inputs=[a_i], outputs=[soft_a, logits_a])
    model.load_weights(path)
    return model

def train_U_SDLM_system(X_train, y_train, X_test, y_test, opt):
    # Expand 1 channel for data ------------------------------
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Extract model ---------------------------------------------------------
    e_i_1 = Input((opt.e_input_shape, ), name='extracting_input_1')
    e_o_1  = U_SDLM(e_i_1, opt)
    e_model_1 = Model(inputs=[e_i_1], outputs=[e_o_1])
    e_y_1 = e_model_1([e_i_1])

    e_i_2 = Input((opt.e_input_shape, ), name='extracting_input_2')
    e_o_2  = U_SDLM(e_i_2, opt)
    e_model_2 = Model(inputs=[e_i_2], outputs=[e_o_2])
    e_y_2 = e_model_2([e_i_2])
    
    e_i_3 = Input((opt.e_input_shape, ), name='extracting_input_3')
    e_o_3  = U_SDLM(e_i_3, opt)
    e_model_3 = Model(inputs=[e_i_3], outputs=[e_o_3])
    e_y_3 = e_model_3([e_i_3])

    m_logit  = concatenate([e_y_1, e_y_2, e_y_3], axis=-1, name='merged_logit_output')

    path = join(opt.weights_path, "U_SDLM")

    if opt.train_mode:
        # ------------------------------------- GENERATE DATA ---------------------------------------------------------
        # Data of main branch
        X_train, y_train = generate_triplet(X_train, y_train)  #(anchors, positive, negative)
        a_data = X_train[:, 0].reshape(-1, opt.input_shape, 1)
        p_data = X_train[:, 1].reshape(-1, opt.input_shape, 1)
        n_data = X_train[:, 2].reshape(-1, opt.input_shape, 1)

        # Data of extract branch
        if opt.Ex_feature == 'time':
            e_a_data = extracted_feature_of_signal(np.squeeze(a_data))
            e_p_data = extracted_feature_of_signal(np.squeeze(p_data))
            e_n_data = extracted_feature_of_signal(np.squeeze(n_data))

        if opt.Ex_feature == 'fre':
            e_a_data = handcrafted_features(np.squeeze(a_data))
            e_p_data = handcrafted_features(np.squeeze(p_data))
            e_n_data = handcrafted_features(np.squeeze(n_data))

        if opt.Ex_feature == 'time_fre':
            a_time   = extracted_feature_of_signal(np.squeeze(a_data))
            p_time   = extracted_feature_of_signal(np.squeeze(p_data))
            n_time   = extracted_feature_of_signal(np.squeeze(n_data))

            a_fre   = handcrafted_features(np.squeeze(a_data))
            p_fre   = handcrafted_features(np.squeeze(p_data))
            n_fre   = handcrafted_features(np.squeeze(n_data))

            e_a_data = np.concatenate((a_time, a_fre), axis=-1)
            e_p_data = np.concatenate((p_time, p_fre), axis=-1)
            e_n_data = np.concatenate((n_time, n_fre), axis=-1)
            
        if opt.scaler != None:
            length = len(e_a_data)
            all_ = np.concatenate((e_a_data, e_p_data, e_n_data), axis=0)
            all_, scale_2 = scaler_fit(all_, opt)
            e_a_data, e_p_data, e_n_data = all_[:length, :], all_[length: length*2, :], all_[length*2: , :]

        model = Model(inputs=[e_i_1, e_i_2, e_i_3], outputs= m_logit)
        model.summary()
        if opt.load_weights:
            if isdir(path):
                model.load_weights(path)
                print(f'\n Load weight : {path}')
            else:
                print('\n No weight file.')

        model.compile(loss=triplet_loss,
                    optimizer=tf.keras.optimizers.experimental.RMSprop(), 
                    metrics=["accuracy"])

        # Note:
        # y=[t_soft, c_data] c_data is just for afternative position for blank position
        # only use t_soft for softmax head in training process
        model.fit(x = [e_a_data, e_p_data, e_n_data], 
                  y = e_a_data,
                  batch_size=opt.batch_size, 
                  epochs=opt.epochs, 
                #   callbacks=[callback], 
                  shuffle=True)

        save(model, path)

    # ------------------------------------- TEST MODEL ---------------------------------------------------------
    model = Model(inputs=[e_i_1], outputs=e_o_1)
    model.load_weights(path)
    return model
    
    