# from keras.models import Sequential
# from keras.layers import Dense
import numpy as np
import json
import pickle
import math
from scipy.sparse import dok_matrix
from scipy import io
from utils import *

_We_dim = 300

EOD = "*EOD*"
START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"

data_path = "/home/ml/jliu164/code/data/"
src_path = "/home/ml/jliu164/code/Summarization/"
input_path = data_path+"/seq_input/"
NUM_CAND = 10
_PAD_score = -np.inf



#################################################
################### Get data ###################
#################################################

# save all sequence vectors as input. use word lemma. add_EOD: NOT omit punctuation/numbers
def _get_seq2vec(topic, add_EOD = False, content = True):
	with open("/home/ml/jliu164/code/data/we_file.json") as f:
		We = json.load(f)
		unk_vec = We["UNK"]
	for i in range(3):
		_path = input_path+("contents/" if content else "summaries/")+topic+"/"+topic+str(i)+".pkl"
		doc,_ = pickle.load(open(_path,"rb"))
		if add_EOD:
			doc_ = [i[-1].append(EOD) for i in doc]
		sentences = [i for val in doc for i in val]
		
		we_sents = np.zeros((1,_We_dim+1)) # last number indicate if end of document
		for sent in sentences:
			eod = False # flag for last sentence of document
			vec_sent = np.zeros(_We_dim)
			for w in sent:
				if w in set([START_SENT, END_SENT,SOD]):
					continue
				if w == EOD:
					eod = True
				vec = np.array(We.get(w,unk_vec),dtype=np.float32)
				vec_sent += vec
			assert np.all(vec_sent) != 0.0
			# if np.all(vec_sent) == 0:
			# 	print(sent)
			if eod:
				np.hstack((vec_sent,np.array([1])))
			else:
				np.hstack((vec_sent,np.array([0])))
			we_sents = np.vstack((we_sents,vec_sent.reshape((-1,))))
		_save = src_path+"model_input/"+topic+str(i)+("_content" if content else "_summary")+"_sent&vec.npy"
		np.save( _save, we_sents)


# calculate importance score for every sentence
def _get_importance(topic, filename = "M_tmp.txt", saved = False):
	if saved:
		M = pickle.load(open(data_path+filename.split(".")[0]+".pkl","rb"))
		return M
	_path = data_path+filename
	# feed into the model
	pass
	return M



def get_data(i=0, topic ='War Crimes and Criminals' ,save_name = "tmp", saved = False):
	"""
	i = 0: dev set, i=1: train set, i=2: test set
	candidate set: the next NUM_CAND-1 sentences of x[i]
	X: (#sentences, we_dim + avg(M) + max(M) + std(M))
	Y: scores, binary: (#sentences, #candidates). Y[i][j] = 1(sentence[j] in candidate[i] chosen as summary)
	"""
	if saved and save_name:
		X = io.mmread(save_name+str(i)+"_X.mtx")
		Y = io.mmread(save_name+str(i)+"_Y.mtx")
		return X,Y

	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(i)+".pkl","rb"))
	summary, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(i)+".pkl","rb"))
	assert len(doc) == len(summary), "More documents than summary!" if len(doc) > len(summary) else "Less documents than summary!"
	
	# doc = doc[:2]
	# summary = summary[:2]
	# get Y
	Y = dok_matrix((sum([len(i) for i in doc]), NUM_CAND),dtype = np.int32)
	idx = 0
	for d,s in zip(doc,summary):
		print("*************************************")
		print("Document has length %s; summary has length %s"%(len(d),len(s)))
		uni_doc = [set(i) for i in d[1:-1]] # omit start/end of sentence but keep start/end of article
		uni_sum = [set(i) for i in s[1:-1]]
		y = np.zeros(len(s)) # s[i] is most similar to d[j]
		for i in range(len(s)):
			max_sim = -1
			for j in range(len(d)):
				numer = len(uni_doc[j].intersection(uni_sum[i]))
				denom = math.sqrt(len(uni_doc[j])*len(uni_sum[i]))
				sim = float(numer)/denom
				if sim > max_sim:
					y[i] = j
					max_sim = sim
		y = sorted(list(set(y))) # every sentence in summary is similar to a different sentence in doc
		print(y)
		c = 0
		for k in range(max(0,int(y[0]-NUM_CAND)),int(y[-1])+1):
			print("updating [%s][%s]"%(k, int(y[c])-(k-idx)))
			if int(y[c])-(k-idx) < NUM_CAND: # only update the ones x[i] can reach
				Y[idx+k,int(y[c])-(k-idx)] = 1
				c+=1
			if c == len(y):
				break
		idx += len(d)
		print(int(y[-1]+1), idx, idx - y[-1])
		Y[int(y[-1]+1):idx,...] += np.full((1,NUM_CAND),_PAD_score) # rows after last sentence chosen: pad with -inf
		# also pad empty candidates with -inf/NaN?
	# print(Y)
	# get X
	try:
		X = np.load(src_path+"model_input/"+topic+str(i)+"_content_sent&vec.npy") # sentence vectors
	except IOError:
		print("Sequence vectors not available!")
	else:
		M = _get_importance()
		X = X.vstack((X,M))
		print(X.shape)

		assert len(X) == len(Y), (X.shape, Y.shape)

	if save_name:
		io.mmwrite(save_name+str(i)+"_X.mtx", X)
		io.mmwrite(save_name+str(i)+"_Y.mtx", Y)
	return X,Y




#################################################
############ Baseline Model #####################
#################################################

class BaseNN(object):
	def __init__(self,hidden_sz, fn):
		self.hidden = hidden_sz
		assert len(hidden_sz)==len(fn), "Layer size must equal activation function size!"

		model = Sequential()
		for h,f in zip(hidden_sz,fn):
			model.add(Dense(h,activation=f))
		self.model = model
		
		def fit(self,X,Y,loss="hinge",optimizer='sgd',epochs=40,batch_size=128):
			assert X.shape[0]==Y.shape[0], "%s X samples, %s Y samples"%(X.shape[0],Y.shape[0])

			
			self.model.compile(loss=loss, optimizer=optimizer)
			hist = self.model.fit(X,Y,epochs=epochs,batch_size=batch_size)

			print(hist)

		def eval(X,Y):
			return self.model.evaluate(X,Y)
			


def main():
	X_train,Y_train = get_data(1)
	assert len(X)==len(Y)

	model = BaseNN([200],['tanh'])
	model.fit(X_train, Y_train)



if __name__ == '__main__':
	_get_seq2vec('War Crimes and Criminals')
	# get_data(0)
	# main()
