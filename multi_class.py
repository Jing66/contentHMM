## use hinge loss, as a multi-class classification problem
import numpy as np
import tensorflow as tf
from all_feat import _length_indicator, FEAT2COL
from sklearn.utils import shuffle


np.random.seed(7)

data_path = "/home/ml/jliu164/code/data/model_input/"
PAD = 0

def make_X(ds = 1, topic = 'War Crimes and Criminals', savename = data_path+"FFNN_var/X_hinge"):
	### x = [source] + [summary] + [All Candidates]. y = one_hot of chosen sentence
	## load from the random sample file
	X_sep = np.load(data_path+"FFNN/X_rdn"+str(ds)+".npy")
	cand_len = np.load(data_path+"FFNN/candidate_length_rdn"+str(ds)+".npy")

	n_dim = 4364 # dimension of x
	X = np.zeros(n_dim) 
	idx = 0
	for cl in cand_len:
		## concate the following cl vectors
		outer_sep = FEAT2COL["sum_se"][1]
		x_outer = X_sep[idx][:outer_sep]
		X_ = X_sep[idx:idx+cl]
		for x in X_:
			x_ = x[outer_sep:] # unique part of each vector
			x_outer = np.concatenate((x_outer, x_))
		
		## PAD if no enough candidate X
		if cl<11:
			n_pad = n_dim - x_outer.shape[0]
			x_outer = np.concatenate((x_outer, np.zeros(n_pad)))

		X = np.vstack((X, x_outer))
		idx += cl

	X = X[1:]
	print("For hinge loss: X.shape", X.shape)
	if savename:
		np.save(savename+str(ds),X)


def make_Y(ds = 1, topic = 'War Crimes and Criminals', savename = data_path+"FFNN_var/Y_hinge"):
	### y = one-hot, pad = 0
	cand_len = np.load(data_path+"FFNN/candidate_length_rdn"+str(ds)+".npy")
	Y_sep = np.load(data_path+"FFNN/Y_rdn"+str(ds)+".npy")

	n_dim = np.max(cand_len)
	Y = np.zeros(n_dim)

	idx = 0
	for cl in cand_len:
		y_sep = Y_sep[idx:idx+cl]
		target = np.argwhere(y_sep == 1)
		# print(cl,y_sep,target)
		assert len(target)==1
		y = np.zeros(n_dim)
		y[target] = 1
		Y = np.vstack((Y, y))

		idx += cl
	Y = Y[1:]
	print("For hinge loss: Y.shape", Y.shape)
	if savename:
		np.save(savename+str(ds),Y)


def load_data(ds=1):
	X = np.load(data_path+"FFNN_var/X_hinge"+str(ds)+".npy")
	Y =  np.load(data_path+"FFNN_var/Y_hinge"+str(ds)+".npy")
	assert len(X) == len(Y)
	X,Y = shuffle(X,Y)
	return X,Y




def hinge(logits,label, delta=1):
	true_score = tf.reduce_sum(logits * label, 0)    
	L = tf.nn.relu((delta + logits -true_score) * (1-label))
	final_loss = tf.reduce_mean(tf.reduce_max(L, 0))
	return final_loss


class MyModel():
	def __init__(self, hiddens, fns = [tf.nn.relu, tf.nn.relu]):
		self.layers = []
		self.fns = fns
		self.hiddens = hiddens

	def fit(self, X,Y,epochs = 20, batch_sz = 32):
		X = X.astype(np.float32)
		Y = Y.astype(np.float32)
		mi = X.shape[1]
		for h,f in zip(self.hiddens, self.fns):
			layer = Layer(mi,h,f )
			mi = h
			self.layers.append(layer)
		
		tfX = tf.placeholder(tf.float32, shape = (None, X.shape[1]))
		tfY = tf.placeholder(tf.float32, shape = (None, Y.shape[1]))

		w = np.random.randn(mi,Y.shape[1])/np.sqrt(2.0/mi)
		b = np.zeros(Y.shape[1])
		self.W = tf.Variable(w.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		logits = self.forward(tfX)
		cost = hinge(logits, tfY)

		train_op =  tf.train.AdamOptimizer().minimize(cost)
		prediction = self.predict(tfX)
		
		init = tf.initialize_all_variables()
		n_batch = len(X)//batch_sz
		costs = []
		with tf.Session() as sess:
			sess.run(init)
			for i in range(epochs):
					for b in range(n_batch):
						X_b, Y_b = X[batch_sz*b: batch_sz*(b+1)],Y[batch_sz*b: batch_sz*(b+1)]
						c = sess.run(train_op, feed_dict={tfX: X_b,tfY: Y_b})
						costs.append(c)
		print(costs)


	def forward(self, X):
		inputs = X
		for l in self.layers:
			outputs = l.forward(inputs)
			inputs = outputs
		return tf.matmul(outputs, self.W)+self.b

	def predict(self, X):
		out = self.forward(X)
		return tf.argmax(out,1)

class Layer():
	def __init__(self, mi, mo, f):
		self.mi = mi
		self.mo = mo
		self.f = f

		W = np.random.randn(mi,mo)/np.sqrt(2.0/mi)
		b = np.zeros(mo)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

	def forward(self, X):
		return self.f(tf.matmul(X,self.W)+self.b)



def main():
	X,Y = load_data()
	# Y = np.array([np.where(r==1)[0][0] for r in Y]).astype(int) # convert from one hot to integer encoding
	X_dev, X_train = X[:int(0.1*len(X))], X[int(0.1*len(X)):]
	Y_dev, Y_train = Y[:int(0.1*len(X))], Y[int(0.1*len(X)):]
	h_sz1 = np.random.random_integers(low= 100, high=2000,size=5)
	h_sz2 = np.random.random_integers(low= 100, high=1500,size=5)
	
	for hiddens in zip(h_sz1,h_sz2):
		print("\n>> Trying hidden size %s "%(hiddens,))
		model = MyModel(hiddens)
		model.fit(X_train,Y_train)
		# score = model.evaluate(X_dev,Y_dev)
		# print(score)

if __name__ == '__main__':
	# make_X()
	# make_Y(savename = None)

	main()
