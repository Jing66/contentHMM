# Use hinge loss but not concatenate all candidates together
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from multi_class import Layer
from multi_class import hinge, acc_rate
from all_feat import feat_select
from utils import MyPCA, pca_we

N_CAND = 10
PAD = -1000
IGNORE = 0
EOS = 11
#############
#	Data 	#
#############
data_path = "/home/ml/jliu164/code/data/model_input/"
def make_Y(ds = 1, topic = 'War Crimes and Criminals', savename = data_path+"FFNN_var/Y_hinge_fixed_"):
	### y = one-hot, pad = 0
	cand_len = np.load(data_path+"FFNN/candidate_length_rdn"+str(ds)+".npy")
	p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	
	## convert Y to one-hot
	Y = np.zeros((sum([len(s) for s in selected_tot])+len(selected_tot), EOS))
	print(Y.shape)
	
	idx_cl = 0
	idx = 0
	for s in selected_tot:
		y = np.zeros((len(s)+1, EOS))
		if s[0] < N_CAND:
			indicies = [s[0]]
		else:
			indicies = [N_CAND-1]

		for j in range(len(s)-1):
			cur_idx = s[j]
			next_idx = s[j+1]
			
			if cur_idx+N_CAND < next_idx:
				indicies.append(EOS-2)
			else:
				indicies.append(max(0,next_idx - cur_idx-1))
		# last row
		indicies.append(EOS-1)
		print(s, indicies)

		indicies = np.array(indicies).astype(int)
		y[np.arange(len(s)+1),indicies] = 1
		Y[idx:idx+len(s)+1] = y
		idx += len(s)+1
		idx_cl +=1
	print("For hinge loss: Y.shape", Y.shape)
	if savename:
		np.save(savename+str(ds),Y)

def make_data(ds=1):
	### return/save X, Y(one-hot), pad all batches to NUM_CAND.
	X = np.load(data_path+"FFNN/X_rdn"+str(ds)+".npy") # max = 512, min=-644
	Y = np.load(data_path+"FFNN_var/Y_hinge_fixed_"+str(ds)+".npy")
	cand_len = np.load(data_path+"FFNN/candidate_length_rdn"+str(ds)+".npy")
	
	assert X.shape[0]==np.sum(cand_len), (X.shape[0],np.sum(cand_len))
	
	X_new = np.zeros(X.shape[1])
	Y_new = np.zeros(N_CAND+1)
	
	## shuffle batches by candidate size
	idx_incr = np.add.accumulate(cand_len) #[i]-[i+1] is where the candidate is
	count = 0
	for i in range(len(cand_len)):
		prev = idx_incr[i-1] if i>0 else 0
		cur = idx_incr[i]
		x = np.zeros((N_CAND+1,X.shape[1]))
		y = Y[count].reshape((-1,))
		# print("candidate length:%s, cur-prev: %s, prev:%s"%(cand_len[i],(cur-prev),prev))
		x[:(cur-prev)] = X[prev:cur]
		## if not enough size, pad x and set y to IGNORE
		n_pad = N_CAND+1 - (cur-prev)
		if n_pad:
			x_pad = np.full((n_pad, x.shape[1]),PAD)
			x[-n_pad:] = x_pad
			# y_pad = np.full(n_pad, IGNORE)
			# y[-n_pad:] = y_pad
		X_new = np.vstack((X_new, x))
		Y_new = np.vstack((Y_new, y.reshape((-1,N_CAND+1))))
		count+=1
	X_new = X_new[1:]
	Y_new = Y_new[1:]
	print("X.shape",X.shape,"Y.shape",Y.shape)
	assert X_new.shape[0]==Y_new.shape[0]*(N_CAND+1), (X_new.shape[0], Y_new.shape[0])
	
	np.save(data_path+"FFNN_var/X_hinge_fixed"+str(ds),X_new)
	np.save(data_path+"FFNN_var/Y_hinge_fixed"+str(ds),Y_new)
	return X_new, Y_new

def shuffle_data(X,Y):
	### shuffle the examples upon blocks of candidate sets
	X_new = np.zeros(X.shape)
	Y_new = np.zeros(Y.shape)

	shuffled_idx = np.random.permutation(Y.shape[0])
	count = 0
	for i, si in enumerate(shuffled_idx):
		x,y = X[si*(N_CAND+1):(si+1)*(N_CAND+1)], Y[si]
		# print("i = %s, si = %s, slice X: [%s - %s]"%(i,si,si*(N_CAND+1),(si+1)*(N_CAND+1)))
		X_new[count:count+(N_CAND+1)] = x
		Y_new[i] = y

	print("X.shape",X.shape,"Y.shape",Y.shape)
	assert X_new.shape[0]==Y_new.shape[0]*(N_CAND+1), (X_new.shape[0], Y_new.shape[0])
	return X,Y


#############
#	Model 	#
#############


# def hinge(logits, label,delta=1):
# 	### hinge loss for score (#candidate set, candidate size), label (#candidate set)
# 	true_score = tf.reduce_sum(logits * label, 0)
# 	L = tf.nn.relu((delta + logits -true_score) * (1-label))
# 	final_loss = tf.reduce_mean(L)
# 	return final_loss

######### Loss function shouldn't have loops ##############
# def hinge(score, label, cand_len):
# 	### compute for each candidate set, the hinge loss
# 	i0=tf.constant(0)
# 	cond = lambda i,acc: tf.less(i, tf.size(cand_len))
# 	def body(i, acc):
# 		cur = tf.gather(cand_len, [i])
# 		prev = tf.cond(i>0, lambda: tf.gather(cand_len,[i-1]), lambda: tf.constant(0, dtype=int32))
# 		size = cur-prev
# 		score_ = tf.slice(score, [prev,0], [size,1])
# 		label_ = tf.slice(label, [prev,0], [size,1])
# 		## loss within candidate
# 		l = _hinge(score_,label_)
# 		# l = tf.constant(5)
# 		return [i+tf.constant(1),l+acc]
	
	# return tf.while_loop(cond, body, loop_vars = [i0,i0])


class HingeModel():
	def __init__(self,dim_in,dim_out, hiddens, fns = [tf.nn.relu, tf.nn.relu], lr=1e-3):
		self.layers = []
		self.hiddens = hiddens

		mi = dim_in
		for h,f in zip(self.hiddens, fns):
			layer = Layer(mi,h,f )
			mi = h
			self.layers.append(layer)
		
		self.tfX = tf.placeholder(tf.float32, shape = (None, dim_in))
		self.tfY = tf.placeholder(tf.float32, shape = (None, N_CAND+1))

		## last layer
		# w = np.random.randn(mi,dim_out)/np.sqrt(2.0/mi)
		# self.W = tf.Variable(w.astype(np.float32))
		b = np.zeros(dim_out)
		self.W = tf.Variable(tf.truncated_normal([mi, dim_out]))
		self.b = tf.Variable(b.astype(np.float32))

		self.logits = self.forward(self.tfX) # logits = Wx + b (no activation applied)
		self.tmp = tf.reshape(self.logits, (-1,N_CAND+1))

		self.cost = hinge(self.tmp,self.tfY) # all candidate sets have size 11, y is one-hot
		# self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.tfY, logits=self.tmp)
		
		self.train_op =  tf.train.AdamOptimizer(learning_rate = lr).minimize(self.cost)
		self.prediction = self.predict(self.tfX)
		
		self.init = tf.initialize_all_variables()
		

	def forward(self, X):
		inputs = X
		for l in self.layers:
			outputs = l.forward(inputs)
			inputs = outputs
		return tf.matmul(outputs, self.W)+self.b

	def predict(self, X):
		## return logits, not one-hot
		out = self.forward(X)
		tmp = tf.reshape(out, (-1,N_CAND+1))
		return tf.argmax(tmp,1)
		


def train_eval(model, X,Y,epochs=50, batch_sz = 32, print_every = 40, savename=None):
	## shuffle data each time
	X,Y = shuffle_data(X,Y)
	sep = int(0.1*Y.shape[0])
	X_dev,Y_dev = X[:sep*(N_CAND+1)],Y[:sep]
	X_train,Y_train = X[sep*(N_CAND+1):], Y[sep:]
	n_batch = Y_train.shape[0]//batch_sz
	costs = []
	with tf.Session() as sess:
		sess.run(model.init)
		for i in range(epochs):
				for b in range(n_batch):
					X_b, Y_b = X_train[batch_sz*b*(1+N_CAND): batch_sz*(b+1)*(1+N_CAND)],Y_train[batch_sz*b: batch_sz*(b+1)]
					assert X_b.shape[0]//(1+N_CAND) == Y_b.shape[0], (X_b.shape,Y_b.shape)
					sess.run(model.train_op, feed_dict={model.tfX: X_b,model.tfY: Y_b})
					tmp,c = sess.run([model.tmp,model.cost], feed_dict={model.tfX: X_b,model.tfY: Y_b})
					# print(tmp.shape) #(32,11)
					# print(Y_b)
					# print(c)
					## at some point cost suddenly become 0???
					if b%print_every==0:
						c = sess.run(model.cost,feed_dict={model.tfX: X_dev, model.tfY: Y_dev})
						p = sess.run(model.prediction, feed_dict={model.tfX: X_dev})
						acc = acc_rate(p,Y_dev)
						# print(p,Y_dev)
						costs.append(c)
						# print("epoch: %s, n_batch: %s, cost on valid: %s, accuracy: %s"%(i,b,c, acc))
		yp = sess.run(model.prediction, feed_dict={model.tfX:X_dev})
		acc = acc_rate(yp,Y_dev)
		print("On validation set accuracy", acc)

		if savename:
			model.saver.save(sess, savename+".ckpt")


def main():
	ds=1
	pca = pca_we()
	X = np.load(data_path+"FFNN_var/X_hinge_fixed"+str(ds)+".npy")
	Y = np.load(data_path+"FFNN_var/Y_hinge_fixed"+str(ds)+".npy")
	rm = {}
	X,Y = feat_select(X,Y,rm=rm, n_pca = 100, pca = pca)
	
	h_sz1 = np.random.random_integers(low= 100, high=2000,size=5)
	h_sz2 = np.random.random_integers(low= 100, high=1500,size=5)
	
	for hiddens in zip(h_sz1, h_sz2):	
		model = HingeModel(X.shape[1],1,hiddens)
		train_eval(model,X,Y)


if __name__ == '__main__':
	# make_Y()
	# X,Y = make_data()
	main()
	
