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

vocab_path = "/home/ml/jliu164/code/data/word2idx.json"
_p = "/home/ml/jliu164/code/data/importance_input/"
we_file = _p+"we_file.npy"

EMBEDDING_DIM = 300
np.random.seed(42)
f1 = open(_p+"X.txt", 'r')
f2 = open(_p+"Y.txt", 'r')
# f11 = open(_p+"testX.txt", 'r')


train_X = []
train_Y = []
test_X = []
# test_Y = []


context_size = 12

k = 0
train_vecs = []
test_vecs = []


i = 0
for lines in f1:
    train_X.append(ast.literal_eval(lines.rstrip('\n')))

# i = 0
# for lines in f11:
#     test_X.append(ast.literal_eval(lines.rstrip('\n')))

i = 0
train_Y = ast.literal_eval(f2.read())

assert len(train_X) == len(train_Y), (len(train_X),len(train_Y))
print("All data loaded.")


# data_X, data_Y = shuffle(data_X, data_Y, random_state=0)

with open(vocab_path) as f:
        vocab = json.load(f)
print("In total %s word to index mapping"%(len(vocab)))
vocabulary_size = len(vocab)


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


while context_size <= 12:
    target_tk = 6
    target_pt = 7
    h_middle = context_size/2

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    # test_X = np.array(test_X)

    num_samples = (train_X.shape)[0]
    train_Y = train_Y[:num_samples]

    train_X_ = train_X[..., int(target_tk-h_middle):int(target_pt+h_middle+1)]
    # test_X_ = test_X[..., target_tk-h_middle:target_pt+h_middle+1]

    print(train_X_.shape)
    print(train_Y.shape)
    # print(test_X_.shape)
   
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
    # h_context = context_size/2

    batch_size = 256
    n_input = 300
    
    print('Build model...')
    model = build_recurrent_model()
    print(model.summary())
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print('Train...')
    model.fit(train_X_, train_Y, batch_size=batch_size, epochs=2,
              validation_split = 0.1)

    # score, acc = model.evaluate(test_X_, test_Y, batch_size=batch_size)
    # yp = model.predict(test_X_, batch_size=batch_size, verbose=1)

    # model.save('models/importance/attn_bilstm_' + str(context_size) + '.h5')
    file = h5py.File('models/importance/attn_bilstm_' + str(context_size) + '.h5py','w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight'+str(i),data=weight[i])
    file.close()
    # fpred = open('pred/pred_attn_bilstm_tmp.txt', 'w')

    # print (np.array(yp)).shape

    # for vals in yp:
    #     fpred.write(str(vals[0]) + '\n')
    # fpred.flush()
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    # print('')
    # mapr = find_map.find_map('attn_bilstm')
    # f50.write(str(acc) + '\n')
    # f100.write(str(mapr) + '\n')
    # f50.flush()
    # f100.flush()

    context_size += 4
    K.clear_session()
    print(">>>> One Mode Built!\n")


