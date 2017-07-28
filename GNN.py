import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from HMM_feature import load_data as _load_featured_data

NUM_CAND = 10
_D_cand = 306 # dim(Ycandidate[i])
_D_Cont = 300 # dim(Context[t])
_PAD = 0 # pad for candidate

def _init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

# generative model
class GNN(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, reg=1.0,fn=T.tanh, epochs=500, margin = 1, show_fig=False):
        # Yc is candidate set, a list of Ycand 
        assert len(X)==len(Y)

        self.fn = fn
        N = len(X) # number of documents-summary pairs
        Dx = X[0].shape[1] # X is of size (N , T(n)x , Dx)
        Dy = Y[0].shape[1] # Y is of size (N , T(n)y , Dy)
        Dcand = _D_cand # Ycand is of size (NUM_CAND x Dcand)
        # Tx = max([X[i].shape[0] for i in range(N)]) # longest Tx
        # Ty = max([Y[i].shape[0] for i in range(N)]) # longest Ty
        Tx = X[0].shape[0]
        Ty = Y[0].shape[0]
        K = _D_Cont # dimension of context vector
        
        ######################### Initialize weights ######################
        Wxc1 = _init_weight(NUM_CAND, Tx) 
        Wxc2 = _init_weight(Dx, K)
        bxc = np.zeros((NUM_CAND,K)) # X => context
        Wyc10 = _init_weight(NUM_CAND, Dy) 
        Wyc20 = _init_weight(1, K) # Y=> context
    
        Wcands = np.random.uniform(-1,1,size=Dcand) # Ycand => score
        Wcs = np.random.uniform(-1,1,size=K)
        bcs = np.zeros(NUM_CAND) # context => score

        h0 = np.zeros(NUM_CAND)
      
        self.Wxc1 = theano.shared(Wxc1)
        self.Wxc2 = theano.shared(Wxc2)
        self.Wyc10 = theano.shared(Wyc10)
        self.Wyc20 = theano.shared(Wyc20) # initial values for recurrence
        self.bxc = theano.shared(bxc)
        self.Wcs = theano.shared(Wys)
        self.bcs = theano.shared(bys)
        ####################### End Initialize weights ######################

        self.params = [self.Wy1x, self.Wys, self.WYs,self.Wxcand, self.bxcand,self.bys,self.bYs]

        thX = T.fmatrix('X') # input one doc: (Tx, Dx)
        thY = T.fmatrix('Y') # output summary sentences: (Ty, Dy)
        thYcand = T.fmatrix('Ycand') # represent the candidates. size (Tx, NUM_CAND, Dcand)
        
        ############## Recurrence ###############
        def recurrence(Y_cand, Wyc1_t, Wyc2_t, Y_t1, X_i):  #order:sequences, prior result(s), non-sequences
            # Y_cand:(NUM_CAND, D_cand). Y_t1:(Dy, ). X_i:(Tx, Dx).
            C_t = self.fn((self.Wxc1.dot(X_i)).dot(self.Wxc2) + (Wyc1_t.dot(Y_t1.T)).dot(Wyc2_t)) # (NUM_CAND, K)
            score = self.Wcs.dot(C_t)+self.Wcands.dot(Y_cand) # size (NUM_CAND,)
            pred = T.argmax(score) # (Dy,)
            return Wyc1_t, Wyc2_t, pred, score

        init_pred = T.zeros(NUM_CAND)
        [Wyc1, Wyc2, preds, scores], _ = theano.scan(
            fn=recurrence,
            outputs_info = [self.Wyc10, self.Wyc20, init_pred, None], # Initialization of Wyc, pred, score
            sequences=[thYcand],
            non_sequences = [thX,thY],
            n_steps=thX.shape[0], # Tx
        )
        ############# End Recurrence #############
        for 
        c = T.switch( thY,
                -T.mean((margin+score-T.sum(scores*(Y_i>0)))*(Y_i==0)), # hinge loss
                0.0
                )
        
        cost = T.sum(c) # cost: sum over all sentences
        dgrads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ] # use momentum
        
        self.predict_op = theano.function(
            inputs=[thX, None, thYcand],
            outputs=[preds],
            allow_input_downcast=True,
        )
        self.train_op = theano.function(
            inputs=[thX, thY, thYcand],
            outputs=[cost, preds],
            updates=updates
        )


        costs = []
        for i in range(epoch):
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                Y_cand = _get_cand(X[j]) #(T(n)x, NUM_CAND, Dcand)
                try:
                    c, p = self.train_op(X[j], Y[j], Y_cand)
                except Exception as e:
                    PYX, pred = self.predict_op(X[j])
                    print "input_sequence len:", len(Y)
                    print "PYX.shape:",PYX.shape
                    print "pred.shape:", pred.shape
                    raise e
                cost += c
            costs.append(cost)
            n_total += len(Y[j])
                for pj, xj in zip(p, Y[j]):
                    if pj == xj:
                        n_correct += 1
                    if j % 200 == 0:
                        sys.stdout.write("j/N: %d/%d correct rate so far: %f\r" % (j,N, float(n_correct)/n_total))
                        sys.stdout.flush()

        if show_fig:
            plt.plot(costs)
        plt.show()


def _get_cand(X_n):
    # for a document X_n, generate a set of candidate for each sentence.
    # Input X_i: (T(n)x,Dx). Output Ycand: (T(n), NUM_CAND, Dcand)
    n = X_n.shape[0] # T(n)x
    Ycand = np.zeros((n, NUM_CAND, Dcand))
    for i in range(n-NUM_CAND):
        Y_cand_j = X_n[i:i+NUM_CAND] # (NUM_CAND, Dx) 
        Ycand[i] = Y_cand_j

    for i in range(max(0,n-NUM_CAND),n):
        # needs padding
        n_pad = n-i
        Y_pad = np.zeros((n_pad, Dcand))
        Y_pad.fill(_PAD)
        Y_cand_j = X_n[i:]
        Ycand[i] = np.vstack((Y_cand_j,Y_pad))
    return Y_cand


def _load_data(conts, sums, context_sz=0):
    # Input: list of documents, each list of sentences, each list of words
    # Output: X:(N, T(n)x,Dx). Y:(N, T(n)y, Dy)
    assert len(conts)==len(sums)
    N = len(conts)
    X_base,Y_base = _load_featured_data(context_sz) # X_base:(# total sentences, Dx)
    idx = 0
    X = np.zeros((N, max([len(c) for c in conts]), X_base.shape[1])) # shorter sentences are 0
    Y = np.zeros((N, max([len(c) for c in sums]), Y_base.shape[1]))
    for i in range(N):
        X[i] = X_base[idx:idx+len(conts[i]),...]
        Y[i] = Y_base[idx:idx+len(sums[i]),...]
        idx += len(conts[i])

    Y_feat = _feat_Y(Y) # add feat: pos(Yprev), cluster(Yprev)
    Y = np.hstack((Y,Y_feat))
    return X,Y

def _feat_Y(Y):
    pass


# TODO
def train():
    X,Y = get_data()



def testing():
    _load_data()



if __name__ == 'main':
    # train()
    testing()


