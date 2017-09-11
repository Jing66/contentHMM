# Use hinge loss but not concatenate all candidates together
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from multi_class import Layer
from multi_class import hinge as _hinge

#############
#	Data 	#
#############
data_path = "/home/ml/jliu164/code/data/model_input/"
def load_data(ds=1):
	### return X, Y, candidate size. shuffled inside each candidate set
	X = np.load(data_path+"FFNN/X_rdn"+str(ds)+".npy")
	Y = np.load(data_path+"FFNN/Y_rdn"+str(ds)+".npy")
	cand_len = np.load(data_path+"FFNN/candidate_length_rdn"+str(ds)+".npy")
	
	assert X.shape[0]==Y.shape[0]==np.sum(cand_len), (X.shape[0], Y.shape[0], np.sum(cand_len))
	X_new = np.zeros(X.shape[1])
	Y_new = np.zeros(1)
	cand_len_new = np.array([-1])
	## shuffle batches by candidate size
	idx_incr = np.add.accumulate(cand_len) #[i]-[i+1] is where the candidate is
	idx = np.random.permutation(len(cand_len)) # shuffled orders
	# print(idx)
	for i in idx:
		prev = idx_incr[i-1] if i>0 else 0
		cur = idx_incr[i]
		x,y = X[prev:cur], Y[prev:cur]
		x,y = shuffle(x,y)
		X_new = np.vstack((X_new, x))
		Y_new = np.append(Y_new, y)
		cand_len_new = np.append(cand_len_new, (cur-prev))
	X_new = X_new[1:]
	Y_new = Y_new[1:]
	cand_len_new = cand_len_new[1:]
	assert X_new.shape[0]==Y_new.shape[0]==np.sum(cand_len_new), (X_new.shape[0], Y_new.shape[0], np.sum(cand_len_new))
	return X_new, Y_new.reshape((1,-1)), cand_len_new


#############
#	Model 	#
#############
def hinge_fixed(logits, label,delta=1):
	### hinge loss for score (#candidate set, candidate size), label (#candidate set)
	assert logits.get_shape()[0] == label.get_shape()[0]
	true_score = tf.reduce_sum(logits * label, 0)    
	L = tf.nn.relu((delta + logits -true_score) * (1-label))
	final_loss = tf.reduce_mean(L)
	return final_loss


def hinge(score, label, cand_len):
	### compute for each candidate set, the hinge loss
	i0=tf.constant(0)
	cond = lambda i,acc: tf.less(i, tf.size(cand_len))
	def body(i, acc):
		cur = tf.gather(cand_len, [i])
		prev = tf.cond(i>0, lambda: tf.gather(cand_len,[i-1]), lambda: tf.constant(0, dtype=int32))
		size = cur-prev
		score_ = tf.slice(score, [prev,0], [size,1])
		label_ = tf.slice(label, [prev,0], [size,1])
		## loss within candidate
		l = _hinge(score_,label_)
		# l = tf.constant(5)
		return [i+tf.constant(1),l+acc]
	
	return tf.while_loop(cond, body, loop_vars = [i0,i0])


class Model():
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
		self.tfCl = tf.placeholder(tf.float32, shape = (None, )) # candidate length tensor (incremented)

		self.W = tf.Variable(tf.truncated_normal([mi, dim_out]))
		self.b = tf.Variable(tf.constant(0, shape=[dim_out]))

		self.logits = self.forward(self.tfX) # logits = Wx + b (no activation applied)
		# self.cost = hinge(self.logits, self.tfY, self.tfCl)
		self.cost = hinge_fixed(tf.reshape(self.logits,(-1,11),self.tfY)) # CASE all candidate sets have size 11, y is one-hot
		
		self.train_op =  tf.train.AdamOptimizer().minimize(self.cost)
		self.prediction = self.predict(self.tfX, self.tfCl)
		
		self.init = tf.initialize_all_variables()
		

	def forward(self, X):
		inputs = X
		for l in self.layers:
			outputs = l.forward(inputs)
			inputs = outputs
		return tf.matmul(outputs, self.W)+self.b

	def predict(self, X, cand_len):
		out = self.forward(X)
		## shoul return a list of argmax for each chunk in cand_len
		i0=tf.constant(0)
		
		cond = lambda i,acc: tf.less(i, tf.size(cand_len))
		def body(i, acc):
			cur = tf.gather(cand_len, [i])
			prev = tf.cond(i>0, lambda: tf.gather(cand_len,[i-1]), lambda: tf.constant(0, dtype=int32))
			size = cur-prev
			out_ = tf.slice(out, [prev,0], [size,1])
			target = tf.argmax(out_)
			# append to tensor
			outputs.append(target)
			return outputs
		output = tf.while_loop(cond, body, loop_vars = [i0,[]])
		return tf.stack(output)


def main():
	pass



if __name__ == '__main__':
	X,Y,cl = load_data()
	
