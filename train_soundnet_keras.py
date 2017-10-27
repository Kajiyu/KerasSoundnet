import os, sys
import glob
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import random

import keras
from keras.layers.core import Dense, Flatten
from keras.layers import Input, BatchNormalization, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras.activations import relu 
from keras.models import Model as KModel
import keras.backend as K
from keras.losses import kullback_leibler_divergence



def keras_conv_2d(prev_layer, in_ch, out_ch, k_h=1,
                 k_w=1, d_h=1, d_w=1,p_h=0, p_w=0, pad='valid',
                 name_scope='conv1', weight_dict=None, eps=1e-5, bn_act=True):
    if pad=='valid':
        padded_input = ZeroPadding2D((p_h, p_w))(prev_layer)
    else:
        padded_input = prev_layer
    
    if weight_dict is not None:
        weights = weight_dict[name_scope]
    
    conv = Conv2D(out_ch, (k_h,k_w),
               strides=(d_h, d_w))
    # Need to pass input through so the layer knows its shape. 
    convOut = conv(padded_input)
    if weight_dict is not None:
        conv.set_weights([weights['weights'], weights['biases']])

    # Break if we don't need to add activation or BatchNorm. 
    if not bn_act:
        return convOut
    
    bn = BatchNormalization(epsilon=eps)
    bnOut = bn(convOut)
    
    if weight_dict is not None:
        bn.set_weights([weights[k] for k in ['gamma','beta','mean','var']])
    act = Activation('relu')
    rOut = act(bnOut)
    
    return rOut


def keras_maxpool(prev, k_h=1, k_w=1, d_h=1, d_w=1):
    return MaxPooling2D(pool_size=(k_h,k_w), strides=(d_h,d_w))(prev)


def create_sn_places_only(param_G=None, n_class=401):
    inp = Input(shape=(None, 1, 1))
    x1 = keras_conv_2d(inp, 1, 16, k_h=64, d_h=2, p_h=32, name_scope='conv1', weight_dict=param_G)
    x2 = keras_maxpool(x1, k_h=8, d_h=8)
    x3 = keras_conv_2d(x2, 16, 32, k_h=32, d_h=2, p_h=16, name_scope='conv2', weight_dict=param_G)
    x4 = keras_maxpool(x3, k_h=8, d_h=8)
    x5 = keras_conv_2d(x4, 32, 64, k_h=16, d_h=2, p_h=8, name_scope='conv3',weight_dict=param_G)
    x6 = keras_conv_2d(x5, 64, 128, k_h=8, d_h=2, p_h=4, name_scope='conv4',weight_dict=param_G)
    x7 = keras_conv_2d(x6, 128, 256, k_h=4, d_h=2, p_h=2, name_scope='conv5',weight_dict=param_G)
    x8 = keras_maxpool(x7, k_h=4, d_h=4)
    x9 = keras_conv_2d(x8, 256, 512, k_h=4, d_h=2, p_h=2, name_scope='conv6',weight_dict=param_G)
    x = keras_conv_2d(x9, 512, 1024, k_h=4, d_h=2, p_h=2, name_scope='conv7',weight_dict=param_G)
    if n_class == 401:
        places = keras_conv_2d(x, 1024, n_class, k_h=8, d_h=2,name_scope='conv8_2',weight_dict=param_G,bn_act=False)
    else:
        places = keras_conv_2d(x, 1024, n_class, k_h=8, d_h=2,name_scope='conv8_2',bn_act=False)
    places = Activation('softmax')(places)
    placesModel = KModel(inputs=inp, outputs=places)
    return placesModel



def kld(p, q):
    print q.shape
    p = p
    q = q
    q = np.mean(q, axis=1)
    q = np.reshape(q, (1, q.shape[0], 1, q.shape[-1]))
    print p.shape, q.shape
    return np.sum(p * np.log(p / q), axis=(p.ndim - 1))



def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=-1)



def custom_kld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.mean(K.clip(y_pred, K.epsilon(), 1), axis=1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def batch_train(model, max_count = 500):
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i*100 + j*10 + k > max_count:
                    break
                img_features = np.load("/data/frames_npy/" + str(i) + str(j) + str(k) + ".npy")
                print img_features.shape
                img_npys = glob.glob("/data/frames/videos/" + str(i) + "/" + str(j) + "/" +  str(k) + "/*.mp4")
                img_npys.sort()
                xx = None
                batch_local_idx = 0
                for ldx, img_npy_path in enumerate(img_npys):
                    img_name = img_npy_path.split("/")[-1]
                    mp3_path = "/data/mp3/videos/" + str(i) + "/" + str(j) + "/" +  str(k) +"/"+ img_name + ".mp3"
                    x, sr = librosa.load(mp3_path)
                    npy_wav = x
                    if x.shape[0] < 220050:
                        t = int(220050.0 / float(x.shape[0])) + 1
                        for hh in range(t):
                            npy_wav = np.concatenate((npy_wav, x))
                    npy_wav = npy_wav[:220050].reshape((1, 220050))
                    if xx is None:
                        xx = npy_wav
                    else:
                        xx = np.concatenate((xx, npy_wav), axis=0)
                    if (ldx + 1) % 100 == 0:
                        print xx.shape
                        xx = np.reshape(xx, (xx.shape[0], xx.shape[1], 1, 1))
                        tmp_loss_val = model.train_on_batch(xx, img_features[batch_local_idx*100:(batch_local_idx+1)*100])
                        batch_local_idx = batch_local_idx + 1
                        print tmp_loss_val
                        xx = None
                print xx.shape
                xx = np.reshape(xx, (xx.shape[0], xx.shape[1], 1, 1))
                tmp_loss_val = model.train_on_batch(xx, img_features[batch_local_idx*100:])
                print tmp_loss_val
                print str(i)+str(j)+str(k)+" all batch finished!"
                model.save_weights('./model/places365_only.hdf5')


NB_EPOCH = 200
MACHINE_STATE = '/gpu:1'

if __name__ == '__main__':
    G_name = './model/sound8.npy'
    param_G = np.load(G_name, encoding='latin1').item()
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
        #    visible_device_list="1",
            allow_growth=True
        ),log_device_placement=True
    )
    sess = tf.Session(config=config)
    K.set_session(sess)
    with tf.device(MACHINE_STATE):
        model = create_sn_places_only(param_G=param_G, n_class=365)
        model.summary()
        model.compile(loss=custom_kld, optimizer='adam')
    for n_epoch in range(NB_EPOCH):
        print n_epoch, "epoch ::::: start!"
        batch_train(model)
