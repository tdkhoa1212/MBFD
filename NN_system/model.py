from tensorflow.keras.layers import concatenate, Input
from Networks.center_model import center
from Networks.SDLM_model import SDLM
from Networks.U_SDLM_model import U_SDLM
from utils.triplet import new_triplet_loss, generate_triplet
from utils.extraction_features import extracted_feature_of_signal, handcrafted_features
from utils.tools import one_hot
from os.path import isdir, join
from tensorflow.keras.models import Model
from tensorflow.saved_model import save
import numpy as np
import AngularGrad

def train_model(X_train, y_train, X_test, y_test, opt):
    # Expand 1 channel for data ------------------------------
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Extract model ---------------------------------------------------------
    e_i_1 = Input((11, ), name='extracting_input_1')
    e_o_1  = U_SDLM(e_i_1, opt)
    e_model_1 = Model(inputs=[e_i_1], outputs=[e_o_1])
    e_y_1 = e_model_1([e_i_1])

    e_i_2 = Input((11, ), name='extracting_input_2')
    e_o_2  = U_SDLM(e_i_2, opt)
    e_model_2 = Model(inputs=[e_i_2], outputs=[e_o_2])
    e_y_2 = e_model_2([e_i_2])
    
    e_i_3 = Input((11, ), name='extracting_input_3')
    e_o_3  = U_SDLM(e_i_3, opt)
    e_model_3 = Model(inputs=[e_i_3], outputs=[e_o_3])
    e_y_3 = e_model_3([e_i_3])
    e_model_1.summary()

     # Center model----------------------------------------------------------
    c_i   = Input((1,), name='center_input')
    c_o = center(c_i, opt)
    c_model = Model(inputs=[c_i], outputs=[c_o])
    c_y = c_model([c_i])

    # Triplet model----------------------------------------------------------
    t_i = Input(shape=(opt.input_shape, 1))
    softmax, logits = SDLM(opt, t_i)
    t_model = Model(inputs=[t_i], outputs=[softmax, logits])
    t_model.summary()
  
    a_i   = Input((opt.input_shape, 1,), name='anchor_input')
    p_i = Input((opt.input_shape, 1,), name='positive_input')
    n_i = Input((opt.input_shape, 1,), name='negative_input')
    

    soft_a, logits_a = t_model([a_i])
    soft_p, logits_p       = t_model([p_i])
    soft_n, logits_n       = t_model([n_i])

    m_logit  = concatenate([logits_a, e_y_1, logits_p, e_y_2, logits_n, e_y_3, c_y], axis=-1, name='merged_logit_output')
    m_soft = concatenate([soft_a, soft_p, soft_n], axis=-1, name='merged_soft_ouput')
    
    loss_weights = [1, 0.01]

    # ------------------------------------- GENERATE DATA ---------------------------------------------------------
    # Data of main branch
    X_train, y_train = generate_triplet(X_train, y_train)  #(anchors, positive, negative)
    a_data = X_train[:, 0].reshape(-1, opt.input_shape, 1)
    p_data = X_train[:, 1].reshape(-1, opt.input_shape, 1)
    n_data = X_train[:, 2].reshape(-1, opt.input_shape, 1)

    # Data of extract branch
    if opt.Ex_feature == 'time':
        e_a_data = extracted_feature_of_signal(np.squeeze(a_data))
        e_p_data = extracted_feature_of_signal(np.aqueeze(p_data))
        e_n_data = extracted_feature_of_signal(np.aqueeze(n_data))

    if opt.Ex_feature == 'fre':
        e_a_data = handcrafted_features(np.squeeze(a_data))
        e_p_data = handcrafted_features(np.aqueeze(p_data))
        e_n_data = handcrafted_features(np.aqueeze(n_data))

    if opt.Ex_feature == 'time_fre':
        a_time   = extracted_feature_of_signal(np.squeeze(a_data))
        p_time   = extracted_feature_of_signal(np.aqueeze(p_data))
        n_time   = extracted_feature_of_signal(np.aqueeze(n_data))

        a_fre   = handcrafted_features(np.squeeze(a_data))
        p_fre   = handcrafted_features(np.aqueeze(p_data))
        n_fre   = handcrafted_features(np.aqueeze(n_data))

        e_a_data = np.concatenate((a_time, a_fre), axis=-1)
        e_p_data = np.concatenate((p_time, p_fre), axis=-1)
        e_n_data = np.concatenate((n_time, n_fre), axis=-1)

    a_label = one_hot(y_train[:, 0])
    p_label = one_hot(y_train[:, 1])
    n_label = one_hot(y_train[:, 2])
    c_data   = y_train[:, 1]

    t_soft = np.concatenate((a_label, p_label, n_label), -1)

    model = Model(inputs=[a_i, e_i_1, p_i, e_i_2, n_i, e_i_3, c_i], outputs=[m_soft, m_logit])
    path = join(opt.weights_path, "S_SDLM")
    if opt.load_weights:
        if isdir(path):
            model.load_weights(path)
            print(f'\n Load weight : {path}')
        else:
            print('\n No weight file.')

    model.compile(loss=["categorical_crossentropy",
                  new_triplet_loss],
                  optimizer=AngularGrad(), 
                  metrics=["accuracy"], 
                  loss_weights=loss_weights)

    # Note:
    # y=[t_soft, c_data] c_data is just for afternative position for blank position
    # only use t_soft for softmax head in training process
    model.fit(x=[a_data, e_a_data, p_data, e_p_data, n_data, e_n_data, c_data], y=[t_soft, c_data],
              batch_size=opt.batch_size, 
              epochs=opt.epoch, 
              # callbacks=[callback], 
              shuffle=True)

    save(model, path)

    # ------------------------------------- TEST MODEL ---------------------------------------------------------
    logits_e_a = concatenate([logits_a, e_y_1], axis=-1, name='logit anchor head')
    model = Model(inputs=[a_i, e_i_1], outputs=[soft_a, logits_e_a])
    model.load_weights(path)
    return model

    
    
    from TSNE_plot import tsne_plot
    tsne_plot(opt.img_outdir, 'original', X_train_embed[:, :opt.embedding_size], X_test_embed[:, :opt.embedding_size], y_train, y_test)
    tsne_plot(opt.img_outdir, 'extracted', X_train_embed[:, opt.embedding_size: ], X_test_embed[:, opt.embedding_size: ], y_train, y_test)