from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
import json
import pickle
import math
from scipy.sparse import dok_matrix
from scipy import io
from utils import *
import h5py
from functools import reduce

_We_dim = 300

EOD = "*EOD*"
START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"

data_path = "/home/ml/jliu164/code/data/"
src_path = "/home/ml/jliu164/code/Summarization/"
input_path = "/home/ml/jliu164/code/contentHMM_input/"
# input_path = data_path+"/seq_input/"
NUM_CAND = 10
_PAD_score = 0



#################################################
################### Get data ###################
#################################################

# save all sequence vectors as input. use word lemma. add_EOD: NOT omit punctuation/numbers
def _get_seq2vec(topic, add_EOD = True, content = True, save = True):
	with open("/home/ml/jliu164/code/data/we_file.json") as f:
		We = json.load(f)
		unk_vec = We["UNK"]
	for i in range(3):
		c = 0
		_path = input_path+("contents/" if content else "summaries/")+topic+"/"+topic+str(i)+".pkl"
		doc,_ = pickle.load(open(_path,"rb"))
		if add_EOD:
			doc_ = [i[-1].append(EOD) for i in doc]
		sentences = [i for val in doc for i in val]
		print("Will process %s sentences"%(len(sentences)))
		
		we_sents = np.zeros((1,_We_dim+2)) # last 2 numbers indicate position, if end of document
		for sent in sentences:
			eod = False
			pos = 0.0 # flag for last sentence of document
			vec_sent = np.zeros(_We_dim)
			for w in sent:
				if w in set([START_SENT, END_SENT,SOD]):
					continue
				if w == EOD:
					eod = True
				vec = np.array(We.get(w,unk_vec),dtype=np.float32)
				vec_sent += vec
			pos += 1
			# assert np.all(vec_sent) != 0.0
			if np.all(vec_sent) == 0:
				print(sent)
			vec_sent = np.hstack((vec_sent,np.array([float(pos)/len(sentences)])))
			if eod:
				vec_sent = np.hstack((vec_sent,np.array([1])))
			else:
				vec_sent=np.hstack((vec_sent,np.array([0])))
			# print(vec_sent.reshape((-1,)).shape)
			
			we_sents = np.vstack((we_sents,vec_sent.reshape(-1,)))
			c += 1
		if save:
			we_sents = we_sents[1:,...] # the first row was empty for initialization
			_save = data_path+"model_input/baseNN/"+topic+str(i)+("_content" if content else "_summary")+"_sent&vec.npy"
			np.save( _save, we_sents)
		print("%s sentences"%(len(we_sents)))


# read importance score for every sentence
def _get_importance(topic, ds, filename = "pred/pred_M0_train_model.npy"):
	m = np.load(src_path+filename) # this is for each word! Need to convert into sentence importance score
	docs,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	N = sum([len(i) for i in docs])
	print("converting m_w (%s) to M_sent (%s) " %(len(m), N)) # N: number of sentences
	c_w = 0 # word index
	c_s = 0 # sentence index
	M = np.zeros((N,2))
	for doc in docs:
		for sentences in doc:
			M_ij = [] # scores in one sentence
			for word in sentences:
				if word in set([START_SENT, END_SENT,SOD, EOD]):
					continue
				M_ij.append(m[c_w])
				c_w += 1
			M_ij = np.array(M_ij)
			M[c_s,:] = [np.mean(M_ij), np.max(M_ij)]
			c_s += 1
	assert len(M) == N, "# Sentence should match: %s from origin, %s from M."%(N,len(M))
	return M
	

def _get_data(ds=1, topic ='War Crimes and Criminals' ,save_name = data_path+"model_input/baseNN/War"):
	"""
	ds = 0: dev set, ds=1: train set, ds=2: test set
	candidate set: the next NUM_CAND-1 sentences of x[i]
	X: (#sentences, we_dim + avg(M) + max(M))
	Y: scores, binary: (#sentences, #candidates). Y[i][j] = 1(sentence[j] in candidate[i] chosen as summary)
	"""
	X_saved = True
	Y_saved = True
	try:
		X = io.mmread(save_name+str(ds)+"_X.mtx")
	except IOError:
		X_saved = False
	try:
		Y = io.mmread(save_name+str(ds)+"_Y.mtx")
	except IOError:
		Y_saved = False
	if X_saved and Y_saved:
		return X,Y

	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	summary, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	assert len(doc) == len(summary), "More documents than summary!" if len(doc) > len(summary) else "Less documents than summary!"
	
	doc = doc[:2]
	summary = summary[:2]

	print("total number of sentences: "+str(sum([len(i) for i in doc])))
	if not Y_saved:
		print("Getting Y...")
		# get Y
		Y = dok_matrix((sum([len(i) for i in doc]), NUM_CAND),dtype = np.int32)
		idx = 0
		with open("/home/ml/jliu164/code/data/we_file.json") as f:
			We = json.load(f)
			unk_vec = We["UNK"]
		for d,s in zip(doc,summary):
			print("*************************************")
			print("Document has length %s; summary has length %s"%(len(d),len(s)))
			uni_doc = [set(i) for i in d] 
			uni_sum = [set(i) for i in s]
			y = np.zeros(len(s)) # s[i] is most close to d[j]
			for i in range(len(s)):
				# max_sim = -1
				max_dist = np.inf
				s_vec = reduce((lambda x,y: x+y),[np.array(We.get(w,unk_vec)) for s in d[i]])
				for j in range(len(d)):
					# unigram cosine similarity
					# numer = len(uni_doc[j].intersection(uni_sum[i]))
					# denom = math.sqrt(len(uni_doc[j])*len(uni_sum[i]))
					# sim = float(numer)/denom
					# if sim > max_sim:
					# 	y[i] = j
					# 	max_sim = sim

					# Euclidean distance
					d_vec = reduce((lambda x,y: x+y),[np.array(We.get(w,unk_vec)) for w in d[j]])
					dist = np.sum((s_vec - d_vec)**2)
					if dist < max_dist:
						y[i] = j
						max_dist = dist
			y = sorted(list(set(y))) # every sentence in summary is similar to a different sentence in doc
			c = 0
			for k in range(max(0,int(y[0]-NUM_CAND)),int(y[-1])+1):
				# print("updating [%s][%s]"%(k, int(y[c])-(k-idx)))
				if int(y[c])-(k-idx) < NUM_CAND: # only update the ones x[i] can reach
					Y[idx+k,int(y[c])-(k-idx)] = 1
					if (y[c])-(k-idx) == 0:
						c+=1
				if c == len(y):
					break
			idx += len(d)
			# print(int(y[-1]+1), idx, idx - y[-1],c)
			Y[int(y[-1]+1):idx,...] += np.full((1,NUM_CAND),_PAD_score) # rows after last sentence chosen: pad with -inf
			# also pad empty candidates with -inf/NaN?
		if save_name:
			io.mmwrite(save_name+str(i)+"_Y.mtx", Y)
	
	if not X_saved:
		print("Getting X...")
		# get X
		try:
			fp= data_path+"model_input/baseNN/"+topic+str(ds)+"_content_sent&vec.npy"
			X = np.load(fp) # sentence vectors
			assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
		except IOError:
			print("Sequence vectors not available! "+fp)
		M = _get_importance(topic,ds)
		X = np.hstack((X,M))
		print(X.shape)
		assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)

		if save_name:
			io.mmwrite(save_name+str(ds)+"_X.mtx", X)
	
	print("X, Y generated! location: "+str(save_name)+str(ds))
	return X,Y



#################################################
############ Baseline Model #####################
#################################################
def build_base_model():
	h_sz = 200
	fn = ["tanh","relu"]
	dim_in = 304
	dim_out = 10
	model = Sequential()
	model.add(Dense(h_sz, activation = fn[0], input_shape = (dim_in,)))
	model.add(Dense(dim_out, activation = fn[1]))
	return model



def train():
	_p = data_path+"model_input/baseNN/War1_"
	X_train = io.mmread(_p+"X.mtx")
	Y_train = io.mmread(_p+"Y.mtx").toarray()
	print(X_train.shape)
	print(Y_train.shape)

	opt = optimizers.Adagrad()	
	model = build_base_model()
	print(model.summary())
	model.compile(loss="hinge", optimizer=opt)
	b_sz = 32
	hist = model.fit(X_train, Y_train,validation_split=0.15, batch_size=b_sz)
	print(hist.history.keys())
	# plt.plot(history.history['acc'])
	# plt.plot(history.history['val_acc'])

	# save model:
	file = h5py.File('models/baseline_War.h5py','w')
	weight = model.get_weights()
	for i in range(len(weight)):
		file.create_dataset('weight'+str(i),data=weight[i])
	file.close()

	scores = model.evaluate(X_train,Y_train)
	print(scores)


def test():
	_p = data_path+"model_input/baseNN/War2_"
	X_test = io.mmread(_p+"X.mtx")
	Y_test = io.mmread(_p+"Y.mtx").toarray()
	print(X_test.shape)
	print(Y_test.shape)
	opt = optimizers.Adagrad()	
	
	model = build_base_model()
	print(model.summary())
	model.compile(loss="hinge", optimizer=opt)

	file=h5py.File('models/baseline_War.h5py','r')
	weight = []
	for i in range(len(file.keys())):
		weight.append(file['weight'+str(i)][:])
	model.set_weights(weight)

	yp= model.predict(X_test)
	print(np.max(yp, axis=1))
	scores = model.evaluate(X_test,Y_test)
	print(scores)


if __name__ == '__main__':
	# _get_seq2vec('War Crimes and Criminals',save = True)
	_get_data(ds=1)
	# _get_importance('War Crimes and Criminals')

	# train()
	# test()
