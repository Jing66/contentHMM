import ast
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.utils import shuffle
from HMM_feature import load_data as _load_featured_data
from baseGNN import _get_cand, load_data, _init_weight

data_path = "/home/ml/jliu164/code/data/"
_content_input = "/home/ml/jliu164/code/contentHMM_input/contents/"
_sum_input = "/home/ml/jliu164/code/contentHMM_input/summaries/"


D_cand = 306
_PAD = 0 # pad for candidate


# generative model
class GNN(object):
    def __init__(self,  D_cont, n_cand):
        self._D_Cont = D_cont # dim(Context[t])
        self._n_cand = n_cand
    
    def fit(self, X, Y,Lx,  learning_rate=10e-1, mu=0.99, reg=1.0,fn=T.tanh, epochs=500, margin = 1, show_fig=False):
        # Lx is a list of length of each source article. len == NW
        assert len(X)==len(Y), "Number of articles don't match!"

        N = len(X) # number of documents-summary pairs
        Dx = int(X[0].shape[1]) # X is of size (N , T(n)x , Dx)
        Dy = Y[0].shape[1] # Y is of size (N , T(n)y , Dy)
        Dy = int(Dx)
        print("Dx:%s, Dy:%s, Dcand:%s"%(Dx,Dy, D_cand))
        
        Tx = int(X[0].shape[0])

        K = self._D_Cont # dimension of context vector
        NUM_CAND = self._n_cand
        
        ######################### Initialize weights ######################
        Wxc = _init_weight(Dx,K)
        bc = np.zeros((Tx,K)) # X => context
        Wyc = _init_weight(Dy,K) # Yprev => context
        Wcc = _init_weight(K,K) # Context_t1 => Context
        
        Wcs1 = _init_weight(NUM_CAND,Tx)
        Wcs2 = _init_weight(K,1)
        Wcands = _init_weight(D_cand,1) # Ycand => score
        bs = np.zeros(NUM_CAND) # context => score

        Wsp = _init_weight(1,NUM_CAND)
        bidx = np.random.rand(1)
      
        self.Wxc = theano.shared(Wxc)
        self.Wyc = theano.shared(Wyc)
        self.bc = theano.shared(bc)
        self.Wcc = theano.shared(Wcc)
        self.Wcs1 = theano.shared(Wcs1)
        self.Wcs2 = theano.shared(Wcs2)
        self.bs = theano.shared(bs)
        self.Wcands = theano.shared(Wcands)
        self.Wsp = theano.shared(Wsp)
        self.bidx = theano.shared(bidx)

        ####################### End Initialize weights ######################

        self.params = [self.Wxc, self.Wyc,self.Wcc, self.bs,self.bc,self.Wcs1,self.Wcs2,self.Wcands, self.Wsp, self.bidx]

        thX = T.fmatrix('X') # input one doc: (Tx, Dx)
        thY = T.fmatrix('Y') # output summary sentences: (Tx, NUM_CAND)
        thYcand = T.ftensor3('Ycand') # represent the candidates. size (Tx, NUM_CAND, Dcand)
        thLx = T.iscalar("len") # T(n)x
        

        ############## Recurrence ###############
        def recurrence(Y_cand,  Yprev, C_t1, t, thY, X_i):  # order:sequences, prior result(s), non-sequences
            # Y_cand:(NUM_CAND, D_cand). Yprev:(Tx, Dy). C_t1:(Tx,K). t: timestep
            C_t = T.tanh(X_i.dot(self.Wxc)+Yprev.dot(self.Wyc)+C_t1.dot(self.Wcc)+self.bc) # size(Tx,K)
            score_t = (self.Wcs1.dot(C_t)).dot(self.Wcs2) + Y_cand.dot(self.Wcands) + self.bs # size (NUM_CAND,) # Dimension ERROR

            pred_idx = T.round(self.Wsp.dot(score_t)+self.bidx)
            pred_idx = T.cast(pred_idx,'int32')
            # pred = Y_cand[pred_idx]             
            pred = (T.switch(pred_idx <0, Yprev[t],  Y_cand[pred_idx])).reshape((D_cand,))[:Dy]

            Yprev = T.set_subtensor(Yprev[t], pred)

            correct_idx = T.argmax(thY[t])
            cost_t = T.switch(thY, 
                -T.mean(margin + score_t - 2*score_t[correct_idx]), 
                0) 

            cost_t = T.cast(cost_t, 'float32')
            return Yprev, C_t, t+1 ,cost_t

        init_pred = T.zeros((Tx, Dy),dtype='float32')
        C_0 = _init_weight(Tx,K)
        C_0.astype(np.float32)

        [Ypreds, _,_, cost], _ = theano.scan(
            fn=recurrence,
            outputs_info = [init_pred, C_0, np.int32(0),None], # Initialization of Wyc, pred, score
            sequences=[thYcand],
            non_sequences = [thY, thX],
            n_steps= thLx, # loop T(n)x times
        )
        ############# End Recurrence #############


        costs = T.sum(cost)
        grads = T.grad(costs, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ] # use momentum
        
        self.predict_op = theano.function(
            inputs=[thX, thY, thYcand, thLx],
            outputs=[Ypreds],
            allow_input_downcast=True,
            mode='DebugMode',
        )
        self.train_op = theano.function(
            inputs=[thX, thY, thYcand, thLx],
            outputs=[cost, Ypreds],
            allow_input_downcast=True,
            updates=updates
        )


        costs = []
        for i in range(epochs):
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                Y_cand = _get_cand(X[j],self._n_cand, n_sent = Lx[j]) #(T(n)x, NUM_CAND, Dcand)

                try:
                    print("**************************** Epoch "+str(j)+" **************************** ")
                    print(X[j].shape)
                    print(Y[j].shape)
                    print(Y_cand.shape)
                    c, p = self.train_op(X[j], Y[j], Y_cand, Lx[j])
                except Exception as e:
                    raise(e)
                    exit(0)
                    PYX, pred = self.predict_op(X[j], None, Y_cand, Lx[j])
                    print("input_sequence len:", len(Y))
                    print("PYX.shape:",PYX.shape)
                    print("pred.shape:", pred.shape)
                    
                cost += c
            costs.append(cost)
            n_total += len(X[j])
            
            for pj, xj in zip(p, Y[j]):
                if pj == xj:
                    n_correct += 1
                if j % 200 == 0:
                    sys.stdout.write("j/N: %d/%d correct rate so far: %f\r" % (j,N, float(n_correct)/n_total))
                    sys.stdout.flush()

        if show_fig:
            plt.plot(costs)
        plt.show()



def main():
    X,Y,Lx = load_data(saved = True, padding = True, base = True)
    model = GNN(400,10)
    model.fit(X,Y,Lx)
    

if __name__ == '__main__':
    main()
