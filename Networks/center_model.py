from tensorflow.keras.layers import BatchNormalization, Dense, Lambda
import keras.backend as K

def center(input, opt):
    x = Dense(opt.e_input_shape*2)(input)
    x = BatchNormalization()(x)
    x = Lambda(lambda  x: K.l2_normalize(x, axis=1))(x)
    return x