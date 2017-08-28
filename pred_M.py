# from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, Input
from keras.layers import LSTM, Bidirectional, GRU, Dropout, merge
from keras.layers.core import *
from keras.models import *
from keras import optimizers
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

import ast
import numpy as np
from sklearn.utils import shuffle
from keras import backend as K
import collections
import find_map
import sys
import os

EMBEDDING_DIM = 300
context_size = 4
vocabulary_size = 49208

f1 = open('/home/ml/jliu164/code/data/importance_input/X_test_model.txt', 'r')

def build_recurrent_model(we_file = True):
    # embedding_layer = Embedding(vocabulary_size, 300, trainable=True)
    if we_file:
        embedding_matrix = np.load('we.npy')
        print("Embedding matrix loaded")
    else:
        embedding_matrix = np.zeros((vocabulary_size+3,EMBEDDING_DIM))
        with open("/home/ml/jliu164/code/data/we_file.json","rb") as d:
            We = json.load(d)
            unk_vec = We["UNK"]
        for k,v in vocab.items():
            embedding_matrix[v] = We.get(k,unk_vec)
        print("Embedding matrix built")
        np.save('we.npy',embedding_matrix)

    embedding_layer = Embedding(vocabulary_size+3,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=context_size+2,
                            trainable=False)
    inputs = Input(shape=(context_size+2,))
    # print(inputs.shape)
    embedding_sequences = embedding_layer(inputs)

    lstm_units = 128
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding_sequences)

    # ATTENTION PART STARTS HERE
    a = Permute((2, 1))(lstm_out)
    b = Reshape((2*lstm_units, context_size + 2))(a)
    c = Dense(context_size + 2, activation='tanh')(b)
    d = Activation('softmax')(c)
    e = Dropout(0.5)(d)
    a_probs = Permute((2, 1))(e)
    attention_mul = merge([lstm_out, a_probs], name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


model = build_recurrent_model()
file=h5py.File('models/importance/attn_bilstm_'+str(context_size)+'.h5py','r')
weight = []
for i in range(len(file.keys())):
    weight.append(file['weight'+str(i)][:])
model.set_weights(weight)
print(model.summary())
print("model_built")


data_X = []

print("Load input...")
for lines in f1:
    data_X.append(ast.literal_eval(lines.rstrip('\n'))) 
data_X = np.array(data_X)


target_tk = 6
target_pt = 7
h_middle = int(context_size/2)
data_X_ = data_X[..., target_tk-h_middle:target_pt+h_middle+1]
print(data_X_.shape)
batch_size = 256
yp = model.predict(data_X_, batch_size=batch_size, verbose=1)

fpred ='pred/pred_M' + str(context_size) + '_test_model.npy'
print(yp.shape)
np.save(fpred, yp)


# for mdl in models:
#     # f50 = open('pred/' + model + '_acc.txt', 'w')
#     f100 = open(mdl + '_map.txt', 'w')

#     print('******************************************')
#     print(mdl)
#     # data_X, data_Y = shuffle(data_X, data_Y, random_state=0)
#     context_size = 0
#     while context_size <= 12:
#         target_tk = 6
#         target_pt = 7
#         h_middle = int(context_size/2)

#         data_X = np.array(data_X)
#         print(data_X.shape)
#         print("target_tk: %s, h_middle:%s,target_pt: %s"%(target_tk,h_middle,target_pt))
#         data_X_ = data_X[..., target_tk-h_middle:target_pt+h_middle+1]
#         target_vec_train = data_X[..., target_pt]
#         target_vec_train = target_vec_train.reshape((target_vec_train.shape)[0], 1)

#         # print test_X_.shape
#         # print target_vec_test.shape
#         data_X_ = np.append(data_X_, target_vec_train, axis = 1)
#         # test_X_ = np.append(test_X_, target_vec_test, axis=1)

#         ##############################
#         #Preprocessing
#         ##############################


#         n_epochs = 15
#         learning_rate = 0.00001

#         n_steps = context_size + 2
#         n_classes = 2
#         n_hidden = 64
#         n_input = 300
#         training_iters = 100000
#         display_rate = 50
#         mid = 10
#         h_context = context_size/2
#         vocabulary_size = 437858
#         # vocabulary_size = 50000

#         batch_size = 256
#         n_input = 300

#         print('Load model...')
#         model = build_recurrent_model()
#         print("model built")
#         file=h5py.File('models/importance/'+mdl+"_" + str(context_size) + '.h5py','r')
#         weight = []
#         for i in range(len(file.keys())):
#             weight.append(file['weight'+str(i)][:])
#         model.set_weights(weight)
#         print(model.summary())

#         # score, acc = model.evaluate(data_X_, data_Y, batch_size=batch_size)
#         yp = model.predict(data_X_, batch_size=batch_size, verbose=1)

#         # fpred = open('pred/pred_' + mdl + '.txt', 'w')

#         # print (np.array(yp)).shape

#         for vals in yp:
#             fpred.write(str(vals[0]) + '\n')
#         fpred.flush()
#         # print('Test score:', score)
#         # print('Test accuracy:', acc)
#         print('')
#         mapr = find_map.find_map(mdl)
#         # f50.write(str(acc) + '\n')
#         f100.write(str(mapr) + '\n')
#         # f50.flush()
#         f100.flush()
#         context_size += 4
#         K.clear_session()
