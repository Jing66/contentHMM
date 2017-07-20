import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

NUM_CAND = 10

def _init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

# generative model
class GNN(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, reg=1.0,fn=T.tanh, epochs=500, show_fig=False):
        
        assert len(X)==len(Y)

        self.fn = fn
        N = len(X) # number of documents
        n = X[0].shape[0] # number of sentences in each doc
        Dx = X[0].shape[1] # X is of size N x T(n) x Dx
        Dy = Y[0].shape[1] # Y is of size N x T'(n) x Dy
        Dcand = len(Y[0][0]) # dimension of feat(Yprev) WTF

        Wy1x = _init_weight(NUM_CAND, n) 
        Wxcand = _init_weight(Dx, Dcand) 
        bxcand = np.zeros((*Y[0].shape))# x -> candidate y: Wy1x * X[i] * Wxcand +b[i] = Y_cand[i]
        Wys = np.random.uniform(-1,1,size=Dcand) 
        bys = np.zeros(NUM_CAND) # candidate -> score: Ycand * Wys + bys = score
        WY1s = _init_weight(NUM_CAND, t) # WTF is t??
        WYs = np.zeros(Dy) 
        bYs = np.zeros(NUM_CAND) # Yprev -> score: WY1s * Yprev * WYs + b = score
  

        self.Wxy = theano.shared(Wxy)
        self.Wys = theano.shared(Wys)
        self.WYs = theano.shared(WYs)
        self.WYy = theano.shared(WYy)
