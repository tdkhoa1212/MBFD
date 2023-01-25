from tensorflow.keras.layers import concatenate, Input
from Networks.S_SDLM_model import S_SDLM
from utils.triplet import triplet_loss, generate_triplet
from utils.tools import one_hot
from os.path import isdir, join
from tensorflow.keras.models import Model
from tensorflow.saved_model import save
import numpy as np
import tensorflow as tf
from utils.angular_grad import AngularGrad

def train_S_SDLM_system(X_train, y_train, X_test, y_test, opt):
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

    if opt.train_model:
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
        print(t_soft.shape, a_data.shape, p_data.shape, n_data.shape)

        model = Model(inputs=[a_i, p_i, n_i], outputs=[m_soft, m_logit])
        model.summary()
        path = join(opt.weights_path, "S_SDLM")
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

    
    
    