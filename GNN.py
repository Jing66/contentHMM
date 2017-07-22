import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

NUM_CAND = 10
_D_cand = 306 # dim(Ycandidate[i])
_D_Cont = 300 # dim(Context)

def _init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

# generative model
class GNN(object):
    def __init__(self, p_keep):
        self.droprate = p_keep
    
    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, reg=1.0,fn=T.tanh, epochs=500, margin = 1, show_fig=False):
        # Yc is candidate set, a list of Ycand 
        assert len(X)==len(Y)


        self.fn = fn
        N = len(X) # number of documents-summary pairs
        n = X[0].shape[0] # number of sentences in each doc => needs padding
        Dx = X[0].shape[1] # X is of size (N , T(n)x , Dx)
        Dy = Y[0].shape[1] # Y is of size (N , T(n)y , Dy)
        Dcand = _D_cand # Ycand is of size (NUM_CAND x Dcand)
        Tx = max([X[i].shape[0] for i in range(N)]) # longest Tx
        Ty = max([Y[i].shape[0] for i in range(N)]) # longest Ty
        K = _D_Cont # dimension of context vector
        
        ######################### Initialize weights ######################
        Wxc1 = _init_weight(NUM_CAND, Tx) 
        Wxc2 = _init_weight(Dx, K)
        bxc = np.zeros((NUM_CAND,K)) # X => context
        Wyc1 = _init_weight(NUM_CAND, Dy) 
        Wyc2 = _init_weight(1, K) # Y=> context
    
        Wcands = np.random.uniform(-1,1,size=Dcand) # Ycand => score
        Wcs = np.random.uniform(-1,1,size=K)
        bcs = np.zeros(NUM_CAND) # context => score

        h0 = np.zeros(NUM_CAND)
      
        self.Wxc1 = theano.shared(Wxc1)
        self.Wxc2 = theano.shared(Wxc2)
        self.Wyc1 = theano.shared(Wyc1)
        self.Wyc2 = theano.shared(Wyc2)
        self.bxc = theano.shared(bxc)
        self.Wcs = theano.shared(Wys)
        self.bcs = theano.shared(bys)

        self.h0 = theano.shared(h0)
        ####################### End Initialize weights ######################

        self.params = [self.Wy1x, self.Wys, self.WYs,self.Wxcand, self.bxcand,self.bys,self.bYs]

        thX = T.fmatrix('X') # input one doc: (Tx, Dx)
        thY = T.vector('Y') # output one summary sentence: (Dy,)
        thYcand = T.fmatrix('Ycand') # represent the candidates. size (NUM_CAND, Dcand)
        
        mask = self.rng.binomial(n=1, p=p, size=thX.shape) # dropouts
        ############## Recurrence ###############
        def recurrence(self, Y_t1, Y_cand, X_i): # Y_i not None when training, to calculate cost
            C_t = self.fn((self.Wxc1.dot(X_i)).dot(Wxc2) + (self.Wyc1.dot(Y_t1)).dot(Wyc2)) # (NUM_CAND, K)
            score = self.Wcs.dot(C_t)+self.Wcands.dot(Y_cand) # size (NUM_CAND,)
            pred = T.argmax(score)
            
            c = T.switch( thY,
                -T.mean((margin+score-T.sum(scores*(Y_i>0)))*(Y_i==0)), # hinge loss
                0.0
                )
            return pred, c
        
        [preds, c], _ = theano.scan(
            fn=recurrence,
            outputs_info = self.h0, # Initialization of Y_t0
            sequences=[thYcand],
            non_sequences = [thX],
            n_steps=thX.shape[0],
        )
        ############# End Recurrence #############
        
        cost = T.sum(c) # cost: sum over all sentences
        dgrads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ] # use momentum
        
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=[prediction],
            allow_input_downcast=True,
        )
        self.train_op = theano.function(
            inputs=[thX, thY, thYcand],
            outputs=[cost, prediction],
            updates=updates
        )


        costs = []
        for i in range(epoch):
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                Y_cand = _get_cand(X[j])
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


def _get_cand(X_i):
    # for a document X, generate a set of candidate for each sentence



def train():
    X,Y = get_data()


if __name__ == 'main':
    train()



