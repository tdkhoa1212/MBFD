# triplet loss
import tensorflow.keras.backend as K
from itertools import permutations
import random
import tensorflow as tf
import numpy as np


def generate_triplet(x, y,  ap_pairs=16, an_pairs=16):
    data_xy = tuple([x, y])
    trainsize = 1

    triplet_train_pairs = []
    y_triplet_pairs = []
    #triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):
        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        # pair = min(len(list(permutations(same_class_idx, 2))), len(diff_class_idx))
        A_P_pairs = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)  # Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        #Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            y_Anchor = data_xy[1][ap[0]]

            Positive = data_xy[0][ap[1]]
            y_Pos = data_xy[1][ap[1]]

            for n in Neg_idx:
                Negative = data_xy[0][n]
                y_Neg = data_xy[1][n]
                
                triplet_train_pairs.append([Anchor, Positive, Negative])
                y_triplet_pairs.append([y_Anchor, y_Pos, y_Neg])
                # test
    return np.array(triplet_train_pairs), np.array(y_triplet_pairs)

    
def new_triplet_loss(y_true, y_pred, alpha=0.2, lambda_=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    total_lenght = y_pred.shape.as_list()[-1]

    anchor   = y_pred[:, 0: opt.embedding_size]
    anchor_extract   = y_pred[:, opt.embedding_size: opt.embedding_size*2]

    positive = y_pred[:, opt.embedding_size*2: opt.embedding_size*3]
    positive_extract = y_pred[:, opt.embedding_size*3: opt.embedding_size*4]

    negative = y_pred[:, opt.embedding_size*4: opt.embedding_size*5]
    negative_extract = y_pred[:, opt.embedding_size*5: opt.embedding_size*6]

    y_center = y_pred[:, opt.embedding_size*6: opt.embedding_size*7]
    y_center_extract = y_pred[:, opt.embedding_size*7:]
    
    # Extract branch loss----------------------------------
    pos_extract_dist  = K.sum(K.square(anchor_extract - positive_extract), axis=1)
    neg_extract_dist  = K.sum(K.square(anchor_extract - negative_extract), axis=1)
    out_extract_l2    = K.sum(K.square(anchor_extract - y_center_extract), axis=1) 
    loss_extract  = K.maximum(lambda_*(pos_extract_dist + out_extract_l2) + lambda_ - neg_extract_dist, 0.0)

    # Main branch loss----------------------------------
    pos_dist          = K.sum(K.square(anchor - positive), axis=1)
    neg_dist          = K.sum(K.square(anchor - negative), axis=1)
    out_l2     = K.sum(K.square(anchor - y_center), axis=1) 
    loss       = K.maximum(lambda_*(pos_dist + out_l2) + lambda_ - neg_dist, 0.0)
    
    # Final loss------------------------
    f_loss = loss + alpha*loss_extract
    return f_loss

def magnitudes(x):
  '''
  https://www.omnicalculator.com/math/angle-between-two-vectors
  '''
  return tf.math.sqrt(tf.math.reduce_sum(x**2))

def product(x, y):
  return tf.math.reduce_sum(x*y)

def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    total_each_lenght = y_pred.shape.as_list()[-1]//3

    anchor   = y_pred[:, :total_each_lenght]
    positive = y_pred[:, total_each_lenght: total_each_lenght*2]
    negative = y_pred[:, total_each_lenght*2: ]

    # distance between the anchor and the positive
    pos_dist          = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist          = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss       = K.maximum(basic_loss, 0.0)

    return loss


def triplet_center_loss(y_true, y_pred, n_classes= 3, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    shape = y_pred.shape[1]//2
    pre_logits, center = y_pred[:, :shape], y_pred[:, shape:]
    y_pred = K.sum(K.square(pre_logits - center), axis=1)

    # repeat y_true for n_classes and == np.arange(n_classes)
    # repeat also y_pred and apply mask
    # obtain min for each column min vector for each class

    classes = tf.range(0, n_classes,dtype=tf.float32)
    y_pred_r = tf.reshape(y_pred, (tf.shape(y_pred)[0], 1))
    y_pred_r = tf.keras.backend.repeat(y_pred_r, n_classes)

    y_true_r = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
    y_true_r = tf.keras.backend.repeat(y_true_r, n_classes)

    mask = tf.equal(y_true_r[:, :, 0], classes)

    #mask2 = tf.ones((tf.shape(y_true_r)[0], tf.shape(y_true_r)[1]))  # todo inf

    # use tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32) #+ (mask2 * tf.cast(tf.logical_not(mask), tf.float32))*tf.constant(float(2**10))
    masked = tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    minimums = tf.math.reduce_min(masked, axis=1)

    loss = K.max(y_pred - minimums +alpha ,0)

    return loss