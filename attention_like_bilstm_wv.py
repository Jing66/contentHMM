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

import collections
import ast
import numpy as np
from sklearn.utils import shuffle
import find_map

np.random.seed(42)
f1 = open('bal_data1/train_X.txt', 'r')
f2 = open('bal_data1/train_Y.txt', 'r')
f11 = open('bal_data1/test_X.txt', 'r')
f12 = open('bal_data1/test_Y.txt', 'r')

train_X = []
train_Y = []
test_X = []
test_Y = []

# train_size = 999971
# test_size = 229700
# train_size = 300000
# test_size = 52000

context_size = 0

k = 0
train_vecs = []
test_vecs = []

# val_size = 50000

# test_size = k - train_size
# test_size = k
# test_size = 900
# f3.seek(0)
# k  = 0
# for lines in f3:
#     if int(lines.rstrip('\n')) >= 12000:
#         # train_size = k
#         break
#     k += 1
# test_size = test_size - train_size

# for k in range(1, train_size + test_size + 1):
#
#     if k%1000==0:
#         print k

print 'Loading Training Data'
i = 0
for lines in f1:
    train_X.append(ast.literal_eval(lines.rstrip('\n')))
    # i += 1
    # if i == 10000:
    #     break


print 'Loaded Training Data'

i = 0
print 'Loading Testing Data'
for lines in f11:
    test_X.append(ast.literal_eval(lines.rstrip('\n')))
    # i += 1
    # if i == 10000:
    #     break

print 'Loaded Testing Data'
i = 0
for lines in f2:
    lines = lines.rstrip('\n')
    train_Y.append(int(lines))
    # i += 1
    # if i == 10000:
    #     break

i = 0
for lines in f12:
    lines = lines.rstrip('\n')
    test_Y.append(int(lines))
    # i += 1
    # if i == 10000:
    #     break

f50 = open('pred/attn_bilstm_acc.txt', 'w')
f100 = open('pred/attn_bilstm_map.txt', 'w')

# data_X, data_Y = shuffle(data_X, data_Y, random_state=0)

vocabulary_size = 300000

def build_recurrent_model():
    embedding_layer = Embedding(vocabulary_size, 300, trainable=True)
    inputs = Input(shape=(context_size+2,))
    embedding_sequences = embedding_layer(inputs)

    lstm_units = 128
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding_sequences)

    # ATTENTION PART STARTS HERE
    a = Permute((2, 1))(lstm_out)
    a = Reshape((2*lstm_units, context_size + 2))(a)
    a = Dense(context_size + 2, activation='tanh')(a)
    a = Activation('softmax')(a)
    a = Dropout(0.5)(a)
    a_probs = Permute((2, 1))(a)
    attention_mul = merge([lstm_out, a_probs], name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model
    # embedding_layer = Embedding(vocabulary_size, 300, trainable=True)
    # inputs = Input(shape=(context_size+2,))
    # embedding_sequences = embedding_layer(inputs)
    #
    # # ATTENTION PART STARTS HERE
    # a = Permute((2, 1))(embedding_sequences)
    # # a = Reshape((input_dim, context_size + 2))(a)
    # a = Dense(context_size + 2, activation='softmax')(a)
    # # a = Reshape((context_size+2,))(a)  # this is the vector!
    # # a = RepeatVector(input_dim)(a)
    # a_probs = Permute((2, 1))(a)
    # attention_mul = merge([embedding_sequences, a_probs], name='attention_mul', mode='mul')
    # # attention_mul = layers.multiply([inputs, a_probs])
    # # ATTENTION PART FINISHES HERE
    #
    # lstm_units = 32
    # attention_mul = Bidirectional(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))(attention_mul)
    # output = Dense(1, activation='sigmoid')(attention_mul)
    # model = Model(input=[inputs], output=output)
    # return model

while context_size <= 40:
    target_tk = 20
    target_pt = 21
    h_middle = context_size/2

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    num_samples = (train_X.shape)[0]
    train_Y = train_Y[:num_samples]

    train_X_ = train_X[..., target_tk-h_middle:target_pt+h_middle+1]
    test_X_ = test_X[..., target_tk-h_middle:target_pt+h_middle+1]

    # target_vec_train = train_X[..., target_pt]
    # target_vec_train = target_vec_train.reshape((target_vec_train.shape)[0], 1)
    #
    #
    #
    # target_vec_test = test_X[..., target_pt]
    # target_vec_test = target_vec_test.reshape((target_vec_test.shape)[0], 1)

    # print test_X_.shape
    # print target_vec_test.shape
    # train_X_ = np.append(train_X_, target_vec_train, axis = 1)
    # test_X_ = np.append(test_X_, target_vec_test, axis=1)
    # val_X = np.array(val_X)
    # val_Y = np.array(val_Y)

    # vec_test = vecs[train_size:train_size + test_size]

    # print vec_test
    print train_X_.shape
    print train_Y.shape
    print test_X_.shape
    print test_Y.shape
    # val_X = np.array(val_X)
    # val_Y = np.array(val_Y)


    ##############################
    #Preprocessing
    ##############################


    n_epochs = 15
    learning_rate = 0.00001

    n_steps = context_size + 2
    n_classes = 2
    n_hidden = 64
    n_input = 300
    training_iters = 100000
    display_rate = 50
    mid = 10
    h_context = context_size/2

    # vocabulary_size = 50000

    batch_size = 256
    n_input = 300
    # target_token = np.random.uniform(-0.25, 0.25, 300)
    # target_token = tf.get_variable("t_token", [batch_size ,n_input], dtype=tf.float32)

    print('Build model...')
    # model = Sequential()

    # model.add(Embedding(vocabulary_size, 300, trainable=True))
    # sequence_input = Input(shape=(context_size + 2,), dtype='int32')
    # embedding_sequences = embedding_layer(sequence_input)
    # l_lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2)
    # # model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    # l_att = AttLayer()(l_lstm)
    # preds = Dense(1, activation='softmax')(l_att)
    # # model.add(Dense(1, activation='relu'))
    # model = Model(sequence_input, preds)

    model = build_recurrent_model()
    print model.summary()
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print('Train...')
    model.fit(train_X_, train_Y, batch_size=batch_size, epochs=2,
              validation_split = 0.1)

    score, acc = model.evaluate(test_X_, test_Y, batch_size=batch_size)
    yp = model.predict(test_X_, batch_size=batch_size, verbose=1)

    model.save('models/attn_bilstm_' + str(context_size) + '.h5')
    fpred = open('pred/pred_attn_bilstm.txt', 'w')

    print (np.array(yp)).shape

    for vals in yp:
        fpred.write(str(vals[0]) + '\n')
    fpred.flush()
    print('Test score:', score)
    print('Test accuracy:', acc)
    print ''
    mapr = find_map.find_map('attn_bilstm')
    f50.write(str(acc) + '\n')
    f100.write(str(mapr) + '\n')
    f50.flush()
    f100.flush()
    context_size += 4
    K.clear_session()
