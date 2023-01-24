from tensorflow.keras.layers import Conv1D, Activation, \
                                    Dense, BatchNormalization, \
                                    Lambda, MaxPooling1D, GlobalAveragePooling1D, \
                                    Input, concatenate, Dropout
from tensorflow_addons.layers import MultiHeadAttention
import keras.backend as K
from keras import layers, regularizers
from keras.models import Model

def TransformerLayer(x=None, c=48, num_heads=12):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    x   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    x = Dropout(0.2)(x)   
    ma  = MultiHeadAttention(head_size=c, num_heads=num_heads)([x, x, x])
    x = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(ma) 
    x = Dropout(0.2)(x)                       
    x = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x) 
    x = Dropout(0.2)(x)
    return x
    
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

    # up-sample from the activation maps.
    # otherwise it's a mismatch. Recommendation of the authors.
    # here we x2 the number of filters.
    # See that as duplicating everything and concatenate them.
    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def SDLM(opt):
    '''
    The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
    https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
    '''
    inputs = Input(shape=[opt.input_shape, 1])
    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=None)(x)

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
    x = Dense(opt.num_classes, activation='softmax')(x)

    m = Model(inputs, x, name='SDLM_model')
    return m