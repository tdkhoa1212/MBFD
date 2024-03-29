import os
import tensorflow as tf
from tensorflow_addons.layers import MultiHeadAttention
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers, layers
from tensorflow.keras.layers import Activation, BatchNormalization, \
                                    Conv1D, Dense, GlobalAveragePooling1D, \
                                    MaxPooling1D, Lambda, concatenate, Lambda

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def TransformerLayer(x=None, c=48, num_heads=4):
    q   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    k   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    v   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    ma  = MultiHeadAttention(head_size=c, num_heads=num_heads)([q, k, v]) 
    return ma

# For m34 Residual, use RepeatVector. Or tensorflow backend.repeat
def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)

    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def S_SDLM(input_, num_classes, opt):
    '''
    The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
    https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
    '''
    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),)(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=None)(x)

    if opt.table == 'table_10_11_12':
        for i in range(2):
            x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i)
        x = MaxPooling1D(pool_size=4, strides=None)(x)
        for i in range(2):
            x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i)
        x = MaxPooling1D(pool_size=4, strides=None)(x)
    else:
        for i in range(4):
            x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i)
            x = MaxPooling1D(pool_size=4, strides=None)(x)
            
        for i in range(3):
            x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i)  
        x = MaxPooling1D(pool_size=4, strides=None)(x)
      
    x = GlobalAveragePooling1D()(x)

    # Two heads of transformer layers----------------------------
    x1 = TransformerLayer(x=x, c=96)
    x2 = TransformerLayer(x=x, c=96)
    x = concatenate([x1, x2], axis=-1)

    x = Dense(opt.embedding_size)(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x) # no first
    logit = x 
    # logit = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    softmax = Dense(num_classes, activation='softmax')(x)
    
    return softmax, logit