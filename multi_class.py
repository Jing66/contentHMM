## use hinge loss, as a multi-class classification problem
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support


np.random.seed(7)

data_path = "/home/ml/jliu164/code/data/model_input/"
PAD = 0

MULT_FEAT = {"src_dist":(0,10),"src_se":(10,310),"sum_all":(310,325),"sum_se":(325,625)}
CAND_DIM = 314+26
CAND_WE_OFFSET = 14

#############
#	Data 	#
#############
from all_feat import _length_indicator, FEAT2COL
def make_X(ds = 1, topic = 'War Crimes and Criminals', savename = data_path+"FFNN_var/X_hinge"):
	### x = [source] + [summary] + [All Candidates]. y = one_hot of chosen sentence
	## load from the random sample file
	X_sep = np.load(data_path+"FFNN/X_rdn"+str(ds)+".npy")
	cand_len = np.load(data_path+"FFNN/candidate_length_rdn"+str(ds)+".npy")

	n_dim = 4365 # dimension of x
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


def load_data(ds=1, rm = {}):
	X = np.load(data_path+"FFNN_var/X_hinge"+str(ds)+".npy")
	Y =  np.load(data_path+"FFNN_var/Y_hinge"+str(ds)+".npy")
	assert len(X) == len(Y)
	## remove some features

	X,Y = shuffle(X,Y)
	return X,Y



#############
#	Model 	#
#############

def hinge(logits,label, delta=1):
	### loss function
	true_score = tf.reduce_sum(logits * label, 0)    
	L = tf.nn.relu((delta + logits -true_score) * (1-label))
	final_loss = tf.reduce_mean(L)
	return final_loss


class MyModel():
	def __init__(self,dim_in,dim_out, hiddens, fns = [tf.nn.relu, tf.nn.relu], lr=0.001):
		self.layers = []
		self.fns = fns
		self.hiddens = hiddens

		mi = dim_in
		for h,f in zip(self.hiddens, self.fns):
			layer = Layer(mi,h,f )
			mi = h
			self.layers.append(layer)
		
		self.tfX = tf.placeholder(tf.float32, shape = (None, dim_in))
		self.tfY = tf.placeholder(tf.float32, shape = (None, dim_out))

		# w = np.random.randn(mi,dim_out)/np.sqrt(2.0/mi)
		# b = np.zeros(dim_out)
		# self.W = tf.Variable(w.astype(np.float32))
		# self.b = tf.Variable(b.astype(np.float32))
		self.W = tf.Variable(tf.truncated_normal([mi, dim_out]))
		self.b = tf.Variable(tf.constant(0.1, shape=[dim_out]))

		self.logits = self.forward(self.tfX)
		# self.cost = hinge(self.logits, self.tfY)
		self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.tfY, logits=self.logits)

		self.train_op =  tf.train.AdamOptimizer().minimize(self.cost)
		self.prediction = self.predict(self.tfX)
		
		self.init = tf.initialize_all_variables()
		

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


def acc_rate(yp,y):
	### yp: prediction, integer encoded. y: truth, one-hot encoded
	y_int = np.argmax(y, axis=-1)
	acc = np.mean(yp==y_int)
	print(acc)
	# precision, recall, f1, _ = precision_recall_fscore_support(y_int,yp, average='micro') # do f1 for each class/candidate?
	return acc
	

def train_eval(model, X,Y,X_dev,Y_dev,epochs=50, batch_sz = 32, print_every = 20):
	n_batch = len(X)//batch_sz
	costs = []
	with tf.Session() as sess:
		sess.run(model.init)
		for i in range(epochs):
				for b in range(n_batch):
					X_b, Y_b = X[batch_sz*b: batch_sz*(b+1)],Y[batch_sz*b: batch_sz*(b+1)]
					# print(X_b[:20],Y_b[:20])
					sess.run(model.train_op, feed_dict={model.tfX: X_b,model.tfY: Y_b})
					if b%print_every==0:
						c = sess.run(model.cost,feed_dict={model.tfX: X, model.tfY: Y})
						p = sess.run(model.prediction, feed_dict={model.tfX: X})
						acc = acc_rate(p,Y)
						costs.append(c)
						print("epoch: %s, n_batch: %s, cost: %s, accuracy: %s"%(i,b,c,acc))
		yp = sess.run(model.prediction, feed_dict={model.tfX:X_dev})
		acc = acc_rate(yp,Y_dev)
		print("On validation set accuracy", acc)
	# plt.plot(costs)
	# plt.show()


#############
#	Main 	#
#############

def main():
	X,Y = load_data(rm = {})
	# Y = np.array([np.where(r==1)[0][0] for r in Y]).astype(int) # convert from one hot to integer encoding
	X_dev, X_train = X[:int(0.1*len(X))], X[int(0.1*len(X)):]
	Y_dev, Y_train = Y[:int(0.1*len(X))], Y[int(0.1*len(X)):]
	
	hiddens = (1030, 727)
		
	model = MyModel(X.shape[1],Y.shape[1],hiddens)
	train_eval(model, X_train, Y_train, X_dev, Y_dev)


if __name__ == '__main__':
	# make_X()
	# make_Y(savename = None)

	main()

	# yp = np.array([3,4,0])
	# y = np.array([[0,0,1,0,0],[0,0,0,0,1],[1,0,0,0,0]])
	# acc_rate(yp,y)

