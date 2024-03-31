# -*- coding: utf-8 -*-
# @Time    : 2023/9/13 15:40
# @Author  : 孙昊
# @File    : model.py

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Flatten, Dense, LSTM, GRU
from transformer import TransformerEncoder,PositionalEmbedding
from keras.layers import Bidirectional,MultiHeadAttention
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout

def mymodel(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    input_1 = Input(name='inputs_1', shape=[length, ], dtype='float32')
    input_2 = Input(name='inputs_2', shape=[length, 21, ], dtype='float32')
    input_3 = Input(name='inputs_3', shape=[length, 1024, ], dtype='float32')
    input_4 = Input(name='inputs_4', shape=[length, 1280, ], dtype='float32')
    input_5 = Input(name='inputs_5', shape=[length, 566, ], dtype='float32')

    x_b = Embedding(input_dim=21, input_length=length, output_dim=ed,)(input_1)

    # ********************** seq ***************************************
    x_1 = PositionalEmbedding(sequence_length=length, input_dim=21, output_dim=ed)(input_1)
    a = TransformerEncoder(ed, 16, 8)(x_1)
    a_1 = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(a)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a_1)
    x_1 = Dropout(dp)(apool)
    x_1 = Bidirectional(LSTM(50, return_sequences=True))(x_1)

    # ********************** one-hot ***********************************
    x_2 = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(input_2)
    x_2 = MaxPooling1D(pool_size=ps, strides=1, padding='same')(x_2)
    x_2 = Dropout(dp)(x_2)
    x_2 = Bidirectional(LSTM(50, return_sequences=True))(x_2)

    # ********************** prot_t5 ***********************************
    x_3 = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(input_3)
    x_3 = MaxPooling1D(pool_size=ps, strides=1, padding='same')(x_3)
    x_3 = Dropout(dp)(x_3)
    x_3 = Bidirectional(LSTM(50, return_sequences=True))(x_3)

    # ********************** esm ***********************************
    x_4 = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(input_4)
    x_4 = MaxPooling1D(pool_size=ps, strides=1, padding='same')(x_4)
    x_4 = Dropout(dp)(x_4)
    x_4 = Bidirectional(LSTM(50, return_sequences=True))(x_4)

    # ********************** aaindex1 ***********************************
    x_5 = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(input_5)
    x_5 = MaxPooling1D(pool_size=ps, strides=1, padding='same')(x_5)
    x_5 = Dropout(dp)(x_5)
    x_5 = Bidirectional(LSTM(50, return_sequences=True))(x_5)

    x = Concatenate(axis=-1)([x_1,x_2,x_3,x_4,x_5, x_b])
    x = Dropout(dp)(x)
    x = MultiHeadAttention(num_heads=8, key_dim=ed)(x, x, x)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = Flatten()(x)

    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)
    outputs = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)
    model = Model(inputs=[input_1,input_2,input_3,input_4,input_5], outputs=outputs)
    adam = Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss=tf.losses.MSE, metrics=[tf.keras.losses.MSE])
    model.summary()
    return model
