import numpy as np
import json
import pickle
import math
import sys
import os
from collections import Counter
if sys.version_info >= (3,0):
	from keras.models import Sequential
	from keras.layers import Dense
	from keras import optimizers
	from keras.metrics import binary_accuracy as accuracy
	import h5py

from scipy import io
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
from functools import reduce

from old_baseline import _get_importance
sys.path.append(os.path.abspath('..'))
from content_hmm import *


EOD = "*EOD*"
START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"

data_path = "/home/ml/jliu164/code/data/"
src_path = "/home/ml/jliu164/code/Summarization/"
input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
# input_path = data_path+"/seq_input/"
NUM_CAND = 10
SKIP_SET = set([START_SENT, END_SENT,SOD, EOD])


def _length_indicator(ds = 1, topic ='War Crimes and Criminals'):
	# save an array to indicate length
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	ind = [len(d) for d in doc]
	return np.array(ind)

# FIXED LENGTH
def _select_sentence(ds=1, topic ='War Crimes and Criminals', savename = data_path+"model_input/FFNN/selected_sentences"):
	# save a list of list. for each selected[i][k], kth sentence in article[i] is selected as summary
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	summary, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	assert len(doc) == len(summary), "More documents than summary!" if len(doc) > len(summary) else "Less documents than summary!"
	
	print("Getting selected sentences...")
	selected = []
	idx = 0
	# print([len(s) for s in summary])
	for d,s in zip(doc,summary):
		uni_doc = [set(i) for i in d] 
		uni_sum = [set(i) for i in s]
		y = np.zeros(len(s)) # s[i] is most close to d[j]
		for i in range(len(s)):
			max_dist = np.inf
			
			for j in range(len(d)):
				# Unigram cosine similarity
				numer = len(uni_doc[j].intersection(uni_sum[i]))
				denom = math.sqrt(len(uni_doc[j])*len(uni_sum[i]))
				dist = 1- float(numer)/denom

				if dist < max_dist:
					y[i] = j
					max_dist = dist
					
		# y = sorted(list(set(y))) # every sentence in summary is similar to a different sentence in doc
		y = sorted(y)
		assert len(y)!=0
		selected.append(y)
	# print([len(s) for s in selected])
	assert len(selected) == len(summary)
	if savename:
		import json
		with open(savename+str(ds)+'.json','w') as f:
			json.dump(selected, f)





def _feat_source(ds = 1, topic = 'War Crimes and Criminals'):
	# feature of source articles. count clusters distributions for each article. 
	# return X_source: (#articles, #clusters)
	sources, _ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	model = pickle.load(open(_model_path+topic+".pkl","rb"))
	n_cluster = model._m
	X_source = np.zeros((len(sources), n_cluster))
	_,flat = model.viterbi(sources)
	assert(len(flat) == sum([len(c) for c in sources]))
	print("model._m = "+str(model._m))
	idx = 0
	for i in range(len(sources)):
		doc = sources[i]
		flat_ = flat[idx: idx+len(doc)]
		flat_count = dict(Counter(flat_))
		for c_id, c in flat_count.items():
			X_source[i][c_id] = c
		idx += len(doc)
	return X_source


def _feat_sum_sofar(ds = 1, topic = 'War Crimes and Criminals',savename = data_path+"model_input/FFNN/X_summary"):
	# feature of summary so far. [topic distributions] + [#wordoverlap with source] + [position of last chosen sentence]. If no summary so far then set all to 0
	# return X_sum: (#summary sentences, #clusters + 2)
	summaries, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	sources, _ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	model = pickle.load(open(_model_path+topic+".pkl","rb"))

	# cluster distribution feature
	_, flat = model.viterbi(summaries)
	n_feat = 2 # #features other than clusters
	X_ = np.zeros((sum([len(s)+1 for s in summaries]), model._m +n_feat))
	print(X_.shape)

	idx = 0
	count = 0
	for i in range(len(summaries)):
		s = summaries[i]
		flat_ = flat[idx: idx+len(s)]
		flat_count = dict(Counter(flat_))
		# repeat source content for all combination of summary (summary so far) [0 to len(s)]
		for c_id, c in flat_count.items():
			X_[count:count+len(s)+1, c_id] = c 
		count += len(s)+1
		idx += len(s)
	
	# word overlap feature: summary sentence X source article
	idx = 0
	X_sum = np.zeros(X_.shape[0])
	for source, summary in zip(sources, summaries):
		for i in range(len(summary)+1):
			sum_so_far = summary[:i]
			uni_doc = [set(i) for i in source] 
			uni_sum = [set(i) for i in sum_so_far]
			if not uni_sum:
				X_sum[idx] = 0 # no summary so far
			else:
				doc_set = reduce((lambda a,b: a.union(b)),uni_doc)
				sum_set = reduce((lambda a,b: a.union(b)),uni_sum)
				X_sum[idx] = len(doc_set.intersection(sum_set))
			idx += 1
	X_[...,model._m] = X_sum

	# position feature: position of last chosen sentence. If no sentence chosen, index -1
	p_selected = data_path+"model_input/FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	assert sum([len(s) for s in selected_tot]) == sum([len(s) for s in summaries])
	s_inserted = [[-1]+s for s in selected_tot]
	pos = [int(i) for val in s_inserted for i in val]
	X_[...,model._m+1] = np.array(pos)

	if savename:
		np.save(savename+str(ds), X_)
	return X_


def _feat_cand_noninterac(ds = 1, topic = 'War Crimes and Criminals',savename = data_path+"model_input/FFNN/X_cand"):
	# features that doesn't need interaction with source and summary so far. [M]+[emission/cluster_id from HMM] + [pos] + [tf-idf]
	# return (#n_sentences in source, 14)
	sources, _ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	flat_sent =  [i for val in sources for i in val]
	
	n_feat = 14
	X_ = np.zeros((len(flat_sent),n_feat))
	print(X_.shape)
	
	# position feature
	X_pos = np.zeros(X_.shape[0])
	idx = 0
	for doc in sources:
		X_pos[idx:idx+len(doc)] = range(len(doc))
		idx += len(doc)
	X_[...,0] = X_pos
	
	# HMM feature (dim= model._m + 1)
	model = pickle.load(open(_model_path+topic+".pkl","rb"))
	_, flat = model.viterbi(sources)
	X_hmm = np.zeros((X_.shape[0],model._m+1))
	X_hmm[...,0] = np.array(flat) # cluster id
	emis_prob = model.sent_logprob(flat_sent)
	X_hmm[...,1:11] = emis_prob.T # sentence emission log probability
	X_[...,1:12] = X_hmm

	# importance score feature
	context_sz = 4
	M = _get_importance(topic, ds,context_sz)
	print(M.shape)
	X_[...,12:14] = M

	# tf-idf score feature
	#############################
	# TO IMPLEMENTE--can only do max/avg without POS/NER here
	#############################

	if savename:
		np.save(savename+str(ds), X_)
	return X_


def make_X(ds = 1, topic = 'War Crimes and Criminals', savedir = data_path+"model_input/FFNN/"):
	# interaction between [source]+[summary_so_far]+[1 candidate out of NUM_CAND]. ADD [trans HMM prob]+[POS/NER overlap]
	
	# load components
	X_source = _feat_source(ds=ds,topic=topic) #(#articles,...)
	try:
		X_summary = np.load(savedir+"X_summary"+str(ds)+".npy") # (summary sentences,...)
	except IOError:
		print("summary part not ready...generating...")
		X_summary = _feat_sum_sofar(ds=ds,topic=topic)
	try:
		X_cand = np.load(savedir+"X_cand"+str(ds)+".npy")  #(#n_sentences in source,...)
	except IOError:
		print("candidate part not ready...generating...")
		X_cand = _feat_cand_noninterac(ds=ds,topic=topic)
	p_selected = data_path+"model_input/FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	len_ind = _length_indicator(ds=ds,topic=topic)
	print(len_ind)

	X = np.zeros(X_source.shape[1]+X_summary.shape[1]+X_cand.shape[1])
	# match generated vectors
	idx_sum = 0
	idx_cand = 0
	for idx_source in range(len(selected_tot)):
		selected_idx = selected_tot[idx_source]
		# locally reference vectors
		source_vec = X_source[idx_source]
		sum_vec = X_summary[idx_sum:idx_sum+len(selected_idx)+1]
		cand_vec = X_cand[idx_cand: idx_cand+len_ind[idx_source]]
		idx_sum += len(selected_idx)+1
		idx_cand += len_ind[idx_source]
		# before the first sentence getting selected as summary
		if selected_idx[0] < NUM_CAND:
			n_i = min(NUM_CAND,len_ind[idx_source])
		else:
			n_i = int(selected_idx[0]+1)
		print("\n>>n = %s. selected idx:%s. first fill in %s rows."%(len_ind[idx_source],selected_idx, n_i))
		tmp = np.hstack((source_vec,sum_vec[0]))[np.newaxis,:]
		tmp = np.broadcast_to(tmp, (n_i, tmp.shape[1])) # (6,22)
		x_prev = np.hstack((tmp,cand_vec[:n_i]))
		# loop inside all indicies
		for idx_ in range(len(selected_idx)-1):
			cur_idx = selected_idx[idx_]
			next_idx = selected_idx[idx_+1]
			n_i = min(NUM_CAND+cur_idx+1, len_ind[idx_source]) - cur_idx -1 if next_idx<NUM_CAND+cur_idx else (next_idx-cur_idx)
			
			_sum_vec = sum_vec[idx_]
			_cand_vec = cand_vec[int(cur_idx+1):int(cur_idx+n_i+1)]
			print("selected idx:%s. next locally fill in rows[%s-%s)."%(cur_idx, cur_idx+1,cur_idx+n_i+1))
			tmp = np.hstack((source_vec, _sum_vec))[np.newaxis,:]
			tmp = np.broadcast_to(tmp, (_cand_vec.shape[0],tmp.shape[1])) #(5,22)
			tmp_ = np.hstack((tmp,_cand_vec))
			x_prev = np.vstack((x_prev, tmp_))
			
			#############################
			# TODO: Implement overlap/transition prob
			#############################
		# Last row
		n_i = min(NUM_CAND+selected_idx[-1]+1, len_ind[idx_source]) - selected_idx[-1] -1 
		tmp = np.hstack((source_vec,sum_vec[-1]))[np.newaxis,:]
		tmp = np.broadcast_to(tmp, (int(n_i), tmp.shape[1]))
		tmp_ = np.hstack((tmp,cand_vec[int(selected_idx[-1]+1):int(selected_idx[-1]+1+n_i)]))
		x_prev = np.vstack((x_prev, tmp_))

		X = np.vstack((X,x_prev))
	X = X[1:]
	print("X.shape",X.shape)

	np.save(savedir+"X"+str(ds),X)

	return X


def make_Y(ds = 1, topic = 'War Crimes and Criminals', savename = data_path+"model_input/FFNN/Y"):
	# X: [source]+[summary_so_far]+[1 candidate out of NUM_CAND]. Y: 1(this candidate was selected)
	len_ind = _length_indicator(ds=ds,topic=topic)
	p_selected = data_path+"model_input/FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	assert selected_tot
	
	Y = np.zeros(1,dtype=int)
	
	for i in range(len(selected_tot)):
		selected = np.array(selected_tot[i])
		indicies = [selected[0]]
		# first row
		if selected[0] < NUM_CAND:
			n_i = min(NUM_CAND,len_ind[i])
		else:
			n_i = selected[0]+1
		print("\n >>Article length = %s, selected = %s, initial fill %s rows" %(len_ind[i], selected,n_i))
		# middle rows
		for j in range(len(selected)-1):
			cur_idx = selected[j]
			next_idx = selected[j+1]
			indicies.append(n_i+next_idx-cur_idx-1)
			n_i += min(NUM_CAND+cur_idx+1, len_ind[i]) - cur_idx -1 if next_idx<NUM_CAND+cur_idx else (next_idx-cur_idx)
		# last row
		n_i += min(NUM_CAND+selected[-1]+1, len_ind[i]) - selected[-1] -1
		print("X_i has rows total %s, label index are %s"%(n_i, indicies))

		y = np.zeros(int(n_i),dtype=np.int32)
		mask = np.array(indicies).astype(int)
		y[mask] = 1	
		Y = np.hstack((Y,y))

	Y = Y[1:]
	if savename:
		np.save(savename+str(ds), Y)
	return Y



def load_data(ds=1,dp=data_path+"model_input/FFNN/"):
	# read X,Y
	try:
		X = np.load(dp+"X"+str(ds)+".npy")
	except IOError:
		print("X not generated...")
	try:
		Y = np.load(dp+"Y"+str(ds)+".npy")
	except IOError:
		print("Y not generated...")
		
	assert X.shape[0]==Y.shape[0],(X.shape, Y.shape)
	# down sample
	pos_id = np.nonzero(Y)[0]
	neg_id = np.where(Y==0)[0]
	print("Y has %s pos label and %s neg label"%(len(pos_id),len(neg_id)))
	neg_sample_id = neg_id[np.random.choice(len(neg_id), len(pos_id), replace=False)]
	Y_pos = Y[pos_id]
	Y_neg = Y[neg_sample_id]
	X_pos = X[pos_id]
	X_neg = X[neg_sample_id]
	
	X_ = np.vstack((X_pos,X_neg))
	Y_ = np.vstack((Y_pos[:,np.newaxis],Y_neg[:,np.newaxis]))
	print("X.shape",X_.shape)
	print("Y.shape",Y_.shape)
	X_,Y_ = shuffle(X_,Y_)
	return X_,Y_




#################################################
############ Baseline Model #####################
#################################################


def _accuracy(yp,Y):
	assert len(yp) == len(Y)
	return float(np.sum(yp==Y))/len(Y)


# binary classification
def build_base_model( dim_in ,h_sz = [170,100], fn = ["sigmoid",'relu','sigmoid']):
	assert len(h_sz)+1 == len(fn)
	model = Sequential()
	model.add(Dense(h_sz[0], activation = fn[0], input_shape = (dim_in,))) # input layer
	for i in range(1,len(h_sz)):
		model.add(Dense(h_sz[i], activation = fn[i])) # hidden
	model.add(Dense(1, activation = fn[-1])) # last layer
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model


def _save_model(model, savename):
	pass


def train():
	_p = data_path+"model_input/FFNN/"
	X,Y = load_data(ds=1,dp=_p)
	N = len(X)
	X_dev, X_train = X[:int(0.1*N)],X[int(0.1*N):]
	Y_dev, Y_train = Y[:int(0.1*N)],Y[int(0.1*N):]

	h_sz1 = np.random.random_integers(low=100,high=1000,size=10)
	h_sz2 = np.random.random_integers(low=100,high=1000,size=10)
	best_loss = np.inf
	best_mode = None
	for h1,h2 in zip(h_sz1,h_sz2):
		print("\n\n>> Trying hidden size [%s,%s] "%(h1,h2))
		model = build_base_model(X.shape[1],h_sz = [h1,h2])
		model.fit(X_train,Y_train,epochs=20,verbose=0)
		score = model.evaluate(X_dev,Y_dev)
		yp = model.predict(X_dev)
		yp[np.where(yp>=0.5)]=1
		yp[np.where(yp<0.5)]=0
		# y_= model.predict(X_train)
		# y_[np.where(y_>=0.5)]=1
		# y_[np.where(y_<0.5)]=0
		# print(yp[:5])
		# print("Gold standard training set (length %s). %s 0s and %s 1s"%(len(X_train),len(np.where(Y_train==0)[0]),len(np.where(Y_train==1)[0])))
		# print("Predicting on training set(length %s). %s 0s and %s 1s"%(len(X_train),len(np.where(y_==0)[0]),len(np.where(y_==1)[0])))
		# print("Predicting on dev set(length %s). %s 0s and %s 1s"%(len(X_dev),len(np.where(yp==0)[0]),len(np.where(yp==1)[0])))
		acc = _accuracy(yp,Y_dev)
		precision, recall, f1, _ = precision_recall_fscore_support(yp,Y_dev)
		print("On Dev set--- Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s, binary_accuray:%s"%(acc, precision, recall, f1,score[0],score[1]))
		if score[0]<best_loss:
			best_model = model
			best_loss = score[0]
	print(best_model.summary())

#######################################
############ Execution ################
#######################################

def test():
	pass
	# test select_article_into_summary
	# savename = data_path+"model_input/FFNN/selected_sentences.json"
	# with open(savename,'r') as f:
	# 	l = json.load(f)
	# print(l[:5])

	# test make Y
	# _select_sentence(ds = 0)
	# Y = make_Y()
	# print(Y.shape) # (32313,)

	# X_source = _feat_source(ds = 1)
	# print(X_source.shape)
	# X_sum = _feat_sum_sofar(ds = 0)
	# _feat_cand_noninterac(ds = 1)
	# X = make_X() # (32313,)

	# X,Y = load_data()
	


if __name__ == '__main__':

	# test()
	train()