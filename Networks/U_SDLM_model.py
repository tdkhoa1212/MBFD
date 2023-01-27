import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, Lambda, Dense
from tensorflow.keras import regularizers


def U_SDLM(input, opt):
  x = Dense(opt.embedding_size*4,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5))(input)
  x = Dense(opt.embedding_size*2,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5))(x)
  x = Dense(opt.embedding_size//2,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5))(x)
  x = Dense(opt.embedding_size*2,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5))(x)
  x = Dense(opt.embedding_size*4,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5))(x)
  x = Dropout(rate=0.5)(x)
  x = Dense(opt.embedding_size)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Lambda(lambda  x: K.l2_normalize(x, axis=1))(x)
  return x

