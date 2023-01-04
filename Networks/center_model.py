from tensorflow.keras.layers import BatchNormalization, Dense

def center(input, opt):
    x = Dense(opt.embedding_size*2)(input)
    x = BatchNormalization()(x)
    return x