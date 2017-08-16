import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from sklearn.utils import shuffle
np.random.seed(7)

# from ffnn_binclassif import load_data
data_path = "/home/ml/jliu164/code/data/"


def init_weight(M1, M2):
  return np.random.randn(M1, M2) * np.sqrt(2.0 / M1)


class HiddenLayer(object):
    def __init__(self, M1, M2, f):
        self.M1 = M1
        self.M2 = M2
        self.f = f
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W)
        self.b = theano.shared(b)
        self.params = [self.W, self.b]

    def forward(self, X):
        if self.f == T.nnet.relu:
            return self.f(X.dot(self.W) + self.b, alpha=0.1)
        return self.f(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, cand_len,activation=T.nnet.relu, learning_rate=1e-3, mu=0.0, reg=0, epochs=100, batch_sz = 32,print_period=100, show_fig=True):
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)

        # initialize hidden layers
        N, D = X.shape
        print("N:",N)
        self.layers = []
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, activation)
            self.layers.append(h)
            M1 = M2
        
        # final layer
        h = HiddenLayer(M1, 1, T.nnet.sigmoid)
        self.layers.append(h)

        # collect params for later use
        self.params = []
        for h in self.layers:
            self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]

        # set up theano functions and variables
        thX = T.matrix('X')
        thY = T.ivector('Y')
        thLength = T.ivector("len")
        yp = self.forward(thX).reshape((-1,1)) # Real number

        rcost = reg*T.mean([(p*p).sum() for p in self.params])

        
        delta = 1 # error margin
        def stack_train(local_len, idx,yphat,true_y): #order: seq,prior,non-seq
            yp_ = yphat[idx:idx+local_len]
            y_ = true_y[idx:idx+local_len]
           
            # cost_ = -T.mean(T.log(yp_[T.arange(y_.shape[0]), y_]))
            # yi_scores = yp_[T.arange(yp_.shape[0]),y_] # http://stackoverflow.com/a/23435843/459241 
            yi_scores = y_
            margins = T.maximum(0, yp_ - yi_scores + delta)
            margins = T.set_subtensor(margins[T.arange(yp_.shape[0]),y_] , 0)
            cost_ = -T.mean(T.sum(margins))

            idx_ = T.cast((idx+local_len),'int8')
            return idx_,cost_# ,pred

        
        [_,costs],_ = theano.scan(
            fn = stack_train,
            outputs_info = [0,None],
            sequences = thLength,
            non_sequences = [yp,thY],
            n_steps = thLength.shape[0]
        )


        def stack_pred(local_len, idx,yp,): #order: seq,prior,non-seq
            yp_ = yp[idx:idx+local_len]            
            pred = T.argmax(yp_)
            idx_ = T.cast((idx+local_len),'int8')
            return idx_,pred
        
        [_,preds],_ = theano.scan(
            fn = stack_pred,
            outputs_info = [0,None],
            sequences = thLength,
            non_sequences = [yp],
            n_steps = thLength.shape[0]
        )

        cost = T.sum(costs) #+ rcost
        
        prediction = preds

        grads = T.grad(cost, self.params)

        # momentum only
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        train_op = theano.function(
            inputs=[thX, thY, thLength],
            outputs=[cost, prediction],
            updates=updates,
            allow_input_downcast=True
        )

        self.predict_op = theano.function(
            inputs=[thX, thLength],
            outputs=prediction,
            allow_input_downcast=True
        )


        n_batches = len(cand_len) // batch_sz # a set of [source + summary]+[10 candidates]
        costs = []
        for i in range(epochs):
            if n_batches > 1:
                # shuffle outside batches
                X,Y,p = shuffle_batch(X,Y,cand_len)

            idx = 0
            for j in range(n_batches):
                p_ = p[j*batch_sz:j*batch_sz+batch_sz]
                cl = cand_len[p_]
                l = np.sum(cand_len[p_])
                Xbatch = X[idx:idx+l]
                Ybatch = Y[idx:idx+l]
                idx += l
                print("X_batch.shape",Xbatch.shape)
                print("Y_batch.shape",Ybatch.shape)

                c, p = train_op(Xbatch, Ybatch,cl)
                # p = predict_op(thX,cl)
                # print("prediction",p)
                costs.append(c)
                if (j+1) % print_period == 0:
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c)
        
        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        out = X
        for h in self.layers:
            out = h.forward(out)
        return out

    def score(self, X, Y):
        P = self.predict_op(X)
        return np.mean(Y == P)

    def predict(self, X):
        return self.predict_op(X)



def shuffle_batch(X,Y,cand_len):
    p = np.random.permutation(len(cand_len))
    print("p",p)
    cl_incr = np.add.accumulate(cand_len)
    X_ = np.zeros_like(X)
    Y_ = np.zeros_like(Y)
    idx = 0
    for i in range(len(p)):
        # print("p[i]",p[i])
        # print("slicing index",cl_incr[p[i]-1],cl_incr[p[i]])
        x = X[cl_incr[p[i]-1:p[i]]] if p[i]>0 else X[:cl_incr[p[i]]]
        y = Y[cl_incr[p[i]-1:p[i]]] if p[i]>0 else Y[:cl_incr[p[i]]]
        # x,y = shuffle(x,y)
        X_[idx:idx+cand_len[p[i]]] = x
        Y_[idx:idx+cand_len[p[i]]] = y
        idx += cand_len[p[i]]
    return X_,Y_,p


def load_data():
    X = np.load(data_path+"model_input/FFNN/X_rdn1.npy")
    Y = np.load(data_path+"model_input/FFNN/Y_rdn1.npy")
    return X,Y


def test():
    X = np.zeros(50)
    Y = np.zeros(50)
    X[:10] = 1
    X[10:15] = 3
    X[15:28] = 4
    X[28:44] = 2
    X[44:48] = 10
    X[48:] = 7
    cl = np.array([10,5,13,16,4,2])
    cl_incr = np.add.accumulate(cl)
    assert np.sum(cl) == len(X)
    X,Y ,p = shuffle_batch(X,Y,cl)
    print(X)
    batch_sz = 2
    n_batches = len(cl) // batch_sz
    print("n_batches",n_batches)
    idx = 0
    for j in range(n_batches):
        p_ = p[j*batch_sz:j*batch_sz+batch_sz]
        l = np.sum(cl[p_])

        x = X[idx:idx+l]
        print(p_)
        print(l)
        print(x)
        idx += l


if __name__ == '__main__':
    X,Y = load_data()
    # Y = Y.reshape((-1,1))
    print("X.shape",X.shape)
    print("Y.shape",Y.shape)
    cand_len = np.load(data_path+"model_input/FFNN/candidate_length1.npy")
    model = ANN(np.random.random_integers(low= 200, high=2500,size=2))
    model.fit(X,Y,cand_len)

    # test()