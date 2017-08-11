from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM
from keras import optimizers
from keras.models import load_model

import ast
import numpy as np
from sklearn.utils import shuffle
from keras import backend as K
import collections
import find_map


f1 = open('data/dataset_X_c.txt', 'r')


data_X = []
context_size = 0

k = 0
vecs = []


i = 0
for lines in f1:
    data_X.append(ast.literal_eval(lines.rstrip('\n')))
    if i%1000000 == 0:
        print 'Processed vectors:' + str(i)
    # if i==200000:
    #     break
    i += 1



models = ['lstm', 'gru','bi_lstm', 'bi_gru', 'attn_bilstm']
# models = ['lstm']
for mdl in models:
    # f50 = open('pred/' + model + '_acc.txt', 'w')
    f100 = open('pred/' + mdl + '_map.txt', 'w')

    print '******************************************'
    print mdl
    # data_X, data_Y = shuffle(data_X, data_Y, random_state=0)
    context_size = 0
    while context_size <= 40:
        target_tk = 20
        target_pt = 21
        h_middle = context_size/2

        data_X = np.array(data_X)
        data_Y = np.array(data_Y)
        print data_X.shape

        data_X_ = data_X[..., target_tk-h_middle:target_pt+h_middle+1]
        target_vec_train = data_X[..., target_pt]
        target_vec_train = target_vec_train.reshape((target_vec_train.shape)[0], 1)


        # print test_X_.shape
        # print target_vec_test.shape
        data_X_ = np.append(data_X_, target_vec_train, axis = 1)
        # test_X_ = np.append(test_X_, target_vec_test, axis=1)

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
        vocabulary_size = 437858
        # vocabulary_size = 50000

        batch_size = 256
        n_input = 300
        # target_token = np.random.uniform(-0.25, 0.25, 300)
        # target_token = tf.get_variable("t_token", [batch_size ,n_input], dtype=tf.float32)

        print('Load model...')
        model = load_model('models/' + mdl + '_' + str(context_size) + '.h5')
        print model.summary()

        # score, acc = model.evaluate(data_X_, data_Y, batch_size=batch_size)
        yp = model.predict(data_X_, batch_size=batch_size, verbose=1)

        fpred = open('pred/pred_' + mdl + '.txt', 'w')

        print (np.array(yp)).shape

        for vals in yp:
            fpred.write(str(vals[0]) + '\n')
        fpred.flush()
        # print('Test score:', score)
        # print('Test accuracy:', acc)
        print ''
        mapr = find_map.find_map(mdl)
        # f50.write(str(acc) + '\n')
        f100.write(str(mapr) + '\n')
        # f50.flush()
        f100.flush()
        context_size += 4
        K.clear_session()
