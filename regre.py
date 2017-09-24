## Regression model -- X is the same and Y is the cosine similarity of candidate sentence to true summary
import numpy as np
import pickle
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.metrics import binary_accuracy as accuracy
import h5py
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.utils import shuffle

from all_feat import feat_select
from eval_model import accuracy_at_k
#############
#	Data 	#
#############

input_path = "/home/ml/jliu164/code/contentHMM_input/"
topic ='War Crimes and Criminals'
EOS_SIM = -1 # similarity for any sentence vs. EOS
N_CAND = 10

def _length_indicator(ds = 1, topic ='War Crimes and Criminals'):
	# save an array to indicate length
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	ind = [len(d) for d in doc]
	return np.array(ind)

def similarity(doc, summary):
	### build a list for comparison of article sentence to summary sentences. 
	sim = []
	for d,s in zip(doc,summary):
		uni_doc = [set(i) for i in d]
		uni_sum = [set(i) for i in s]
		s = np.zeros((len(d), len(s))) # [i,j] = similarity(sentence i in d, sentence j in s)
		for i in range(s.shape[0]):
			## compare d[i] and s[1...n]
			for j in range(s.shape[1]):
				# Unigram cosine similarity
				numer = len(uni_doc[i].intersection(uni_sum[j]))-2
				denom = math.sqrt((len(uni_doc[i])-2)*(len(uni_sum[j])-2))
				s_ = float(numer)/denom # omit start/end of sentence
				s[i][j] = s_
		sim.append(s)
	print("...computed similarity scores for %s files." %(len(sim)))
	return sim


def make_Y(ds = 1,savename = "../data/model_input/FFNN_var/Y_regres"):
	### y for regression. sim(sent, EOS) = -1(EOS_SIM) for all sentence
	summary, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	sim = similarity(doc,summary)
	cand_rec = pickle.load(open("../data/model_input/FFNN/cand_record/rdn_sample"+str(ds)+".pkl","rb")) # doesn't include EOS
	Y = np.zeros(1)

	for i in range(len(summary)):
		sim_ = sim[i]
		cand_rec_ = cand_rec[i]
		selected = np.argmax(sim_,axis=0)
		
		## predict first: y = sim(cand_i, sum_0)
		y = sim_[cand_rec_[0],0]
		y = np.append(y,EOS_SIM)
		## middle rows
		for j in range(sim_.shape[1]-1):
			y_ = sim_[cand_rec_[j+1],j+1]
			y_ = np.append(y_, EOS_SIM)
			y = np.hstack((y, y_))
			
		## EOS prediction
		y_ = np.full(len(cand_rec_[-1]),EOS_SIM)
		y_ = np.append(y_,1) # EOS in this case is a perfect match score
		y = np.hstack((y, y_))
		Y = np.hstack((Y,y))

	Y = Y[1:]
	print("Y.shape",Y.shape)
	if savename:
		np.save(savename+str(ds), Y)



def make_Y_easy(ds=1,savename = "../data/model_input/FFNN_var/Y_regres_easy"):
	### y for regression in an easy task. sim(sent, EOS) = -1(EOS_SIM) for all sentence
	summary, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	sim = similarity(doc,summary)
	len_ind = _length_indicator(ds=ds)
	len_incr = np.add.accumulate(len_ind)
	len_incr2 = np.hstack((np.zeros(1),len_incr))
	cand_rec = pickle.load(open("../data/model_input/FFNN/cand_record/rdn_sample_easy"+str(ds)+".pkl","rb")) # doesn't include true value or EOS
	Y = np.zeros(1)

	c = lambda v: np.argwhere(len_incr>=v)[0].astype(int) #find out which article the n_i belongs to
	comp = np.vectorize(c)
	c2 = lambda v: (v-len_incr2)[np.argwhere((v-len_incr2)>=0)[-1]].astype(int) # find out the position in that article from n_i
	pos = np.vectorize(c2)

	for i in range(len(summary)):
		cand_rec_ = cand_rec[i]
		cand_rec_ = [c.astype(int) for c in cand_rec_]
		selected = np.argmax(sim[i],axis=0)
		## predict first: y = sim(cand_i, sum_0)
		idx_doc = comp(cand_rec_[0])
		idx_pos = pos(cand_rec_[0])
		y = np.zeros(N_CAND+1)
		for k, idx in enumerate(idx_doc):
			y[k] = sim[idx][idx_pos[k],0]
		y[-2] = sim[i][selected[0],0]
		y[-1] = EOS_SIM

		## middle rows
		for j in range(len(selected)-1):
			y_ = np.zeros(N_CAND+1)
			idx_doc = comp(cand_rec_[j+1])
			idx_pos = pos(cand_rec_[j+1])
			for k, idx in enumerate(idx_doc):
				y[k] = sim[idx][idx_pos[k],0]
			
			y_[-2] = sim[i][selected[j+1],j+1]
			y_[-1] = EOS_SIM
			
			y = np.hstack((y, y_))
		## EOS prediction
		y_ = np.full(len(cand_rec_[-1]),EOS_SIM)
		y_ = np.append(y_,1) # EOS in this case is a perfect match score
		y = np.hstack((y, y_))
		Y = np.hstack((Y,y))
	Y = Y[1:]
	print("Y.shape",Y.shape)
	if savename:
		np.save(savename+str(ds), Y)




def load_data(rm,suffix = "", ds=1):
	X = np.load("../data/model_input/FFNN/X_rdn"+suffix+str(ds)+".npy")
	Y = np.load("../data/model_input/FFNN_var/Y_regres"+suffix+str(ds)+".npy").reshape((-1,1))
	X = feat_select(X,Y,rm, n_pca = None, pca = None)
	assert len(X) == len(Y),(X.shape, Y.shape)
	cand_len = np.load("../data/model_input/FFNN/candidate_length_rdn"+suffix+str(ds)+".npy").astype(int)

	sep = int(0.1*len(cand_len))
	cl_train, cl_dev = cand_len[:-sep], cand_len[-sep:]
	X_train, X_dev = X[:np.sum(cl_train)], X[-np.sum(cl_dev):]
	Y_train, Y_dev = Y[:np.sum(cl_train)], Y[-np.sum(cl_dev):]
	return X_train, X_dev, Y_train, Y_dev, cl_train, cl_dev


#############
#	Model 	#
#############
# Regression model
def build_base_model( dim_in ,h_sz = [170,100], fn = ["tanh",'relu'], weights = None):
	model = Sequential()
	model.add(Dense(h_sz[0], activation = fn[0], input_shape = (dim_in,))) # input layer
	#model.add(Dropout(0.2))
	for i in range(1,len(h_sz)):
		model.add(Dense(h_sz[i], activation = fn[i])) # hidden
	model.add(Dense(1)) # last layer
	if weights:
		model.set_weights(weights)
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model



def train(rm = {}):
	X_train_easy,X_dev_easy,Y_train_easy,Y_dev_easy, _, cl_dev_easy = load_data(rm=rm,suffix="_easy")
	X_train,X_dev,Y_train,Y_dev,_, cl_dev = load_data(rm = rm)
	print("X_train shape", X_train.shape,"Y_train.shape",Y_train.shape)
	h_sz1 = np.random.random_integers(low= 100, high=2000,size=5)
	h_sz2 = np.random.random_integers(low= 100, high=1500,size=5)

	for hiddens in zip(h_sz1, h_sz2):
		model1 = build_base_model(X_train.shape[1],h_sz = hiddens)
		model1.fit(X_train_easy,Y_train_easy, verbose=0)
		Yp = model1.predict(X_dev_easy)
		var = explained_variance_score(Y_dev_easy, Yp) # best is 1.0
		r2 = r2_score(Y_dev_easy, Yp) # best is 1.0
		acc,_ = accuracy_at_k(Yp, Y_dev_easy,cl_dev_easy, 1)
		print("\n>> Easy case. Hidden: %s, accuracy: %s, explained variance: %s, r2: %s"%(hiddens,acc, var, r2))

		model2 = build_base_model(X_train.shape[1],h_sz = hiddens,weights = model1.get_weights())
		model2.fit(X_train,Y_train, verbose=0)
		loss = model2.evaluate(X_dev, Y_dev)
		Yp = model2.predict(X_dev)
		var = explained_variance_score(Y_dev, Yp) # best is 1.0
		r2 = r2_score(Y_dev, Yp) # best is 1.0
		acc,_ = accuracy_at_k(Yp, Y_dev,cl_dev, 1)
		print(" loss: %s, accuracy: %s, explained variance: %s, r2: %s"%(loss,acc, var, r2))



if __name__ == '__main__':
	# make_Y()
	# make_Y_easy()
	train(rm = {})