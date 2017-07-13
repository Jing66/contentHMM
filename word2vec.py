import json
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.urils import shuffle

from utils import *

class W2V(object):
	def __init__(self, D, V, context_sz):
		self.D = D
		self.V = V
		self.context_sz = context_sz

	def _set_Pnw(self,X):
		word_freq = {}
		word_count = sum([len(x) for x in X])
		for x in X:
			if not word_freq.get(x):
				word_freq[x] = 1
			else:
				word_freq[x] = word_freq.get(x) + 1
		self.Pnw = np.zeros(self.V)
		for i in range(4,self.V): # skip SOS,EOS,UNK,SOD
			self.Pnw[i] = word_freq[i]/word_count
		self.Pnw = self.Pnw ** 0.75
		assert np.all(Pnw[4:] > 0)
		return self.Pnw

	def _get_neg_samples(context, num_neg_samples):
        saved = {}
        for context_idx in context:
            saved[context_idx] = self.Pnw[context_idx]
            print "saving -- context id:", context_idx, "value:", self.Pnw[context_idx]
            self.Pnw[context_idx] = 0
        neg_samples = np.random.choice(
            xrange(self.V),
            size=num_neg_samples,
            replace=False,
            p=self.Pnw / np.sum(self.Pnw),
        )
        # print "saved:", saved
        for j, pnwj in saved.iteritems():
            self.Pnw[j] = pnwj
        assert(np.all(self.Pnw[4:] > 0))
        return neg_samples


	def fit(self, X,num_neg_samples=10, learning_rate = 1e-4, reg = 0.1, epochs = 10,mu=0.99):
		N = len(X)
        V = self.V
        D = self.D
        self._set_Pnw(X)

        # initialize weights and momentum changes
        W1 = init_weights((V, D))
        W2 = init_weights((D, V))
        W1 = theano.shared(W1)
        W2 = theano.shared(W2)

        thInput = T.iscalar('input_word')
        thContext = T.ivector('context')
        thNegSamples = T.ivector('negative_samples')

        W1_subset = W1[thInput]
        W2_psubset = W2[:, thContext]
        W2_nsubset = W2[:, thNegSamples]

        p_activation = W1_subset.dot(W2_psubset)
        pos_pY = T.nnet.sigmoid(p_activation)
        n_activation = W1_subset.dot(W2_nsubset)
        neg_pY = T.nnet.sigmoid(-n_activation)

        cost = -T.log(pos_pY).sum() - T.log(neg_pY).sum()

        W1_grad = T.grad(cost, W1_subset)
        W2_pgrad = T.grad(cost, W2_psubset)
        W2_ngrad = T.grad(cost, W2_nsubset)

        W1_update = T.inc_subtensor(W1_subset, -learning_rate*W1_grad)
        W2_update = T.inc_subtensor(T.inc_subtensor(W2_psubset, -learning_rate*W2_pgrad)[:,thNegSamples], -learning_rate*W2_ngrad)
        updates = [(W1, W1_update), (W2, W2_update)]

        train_op = theano.function(
            inputs=[thInput, thContext, thNegSamples],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True,
        )

        sample_indices = range(N)
        costs = []
        cost_per_epoch = []
        for i in range(epochs):
        	sample_indices = shuffle(sample_indices)
        	cost_per_epoch_i = []

        	for it in sample_indices:
        		x = X[it] # one sentence
        		# too short to do 1 iteration, skip
        		if len(x) < 2 * self.context_sz + 1:
        			continue
        		cj = []
        		n = len(x)
        		########## try one random window per sentence ###########
        		jj = np.random.choice(n)
        		start = max(0, jj - self.context_sz)
        		end = min(n, jj + 1 + self.context_sz)
        		context = np.concatenate([x[start:jj], x[(jj+1):end]])
        		context = np.array(list(set(context)), dtype=np.int32)
        		neg_samples = self._get_negative_samples(context, num_neg_samples)

        		c = train_op(x[jj], context, neg_samples)
        		cj.append(c / (num_neg_samples + len(context)))
        		#########################################################
        		cj = np.mean(cj)
        		cost_per_epoch_i.append(cj)
        		costs.append(cj)
        		if it % 100 == 0:
        			sys.stdout.write("epoch: %d j: %d/ %d cost: %f\r" % (i, it, N, cj))
        			sys.stdout.flush()

        	epoch_cost = np.mean(cost_per_epoch_i)
        	cost_per_epoch.append(epoch_cost)
        	print "time to complete epoch %d:" % i, (datetime.now() - t0), "cost:", epoch_cost

        self.W1 = W1.get_value()
        self.W2 = W2.get_value()
        plt.plot(costs)
        plt.title("Theano costs")
        plt.show()

        plt.plot(cost_per_epoch)
        plt.title("Theano cost at each epoch")
        plt.show()



    def save(path):
    	weights = [self.W1, self.W2]
    	np.savez(path,*weights)


    def load(path):
    	npz = np.load(path)
    	W1 = npz['arr_0']
    	W2 = npz['arr_1']
    	return W1, W2

def main():
	path = "word2vec/input/tmp.pkl"
	sentences, word2idx = pickle.load(open(path))

	D = 100
	model = W2V(D, len(word2idx),8)
	model.fit(sentences)
	model.save("we_file.npz")

if __name__ == '__main__':
	main()




