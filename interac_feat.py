
import numpy as np
import json
import pickle
import math
import sys
import os
import math
from functools import reduce

from all_feat import _length_indicator
sys.path.append(os.path.abspath('..'))
from content_hmm import *
####### interaction between candidate and summaries
# P(S_cand|S_last summary)
# Sim(cand, n previous summary). 0 if no summary yet
# pos(cand)-pos(last summary). <0 if no summary yet?>
# Importance score (avg) by unigram frequency in source
###################################

input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
data_path = "/home/ml/jliu164/code/data/"
savename =  data_path+"model_input/FFNN/X_interac_nprev"
NUM_CAND = 10
N_PREV = 3 # compare candidate with 3 previous summaries for similarity
START_SENT = "**START_SENT**"
START_DOC = "**START_DOC**"
END_SENT = "**END_SENT**"
UNK = "**UNK**"


n_dim = 3 + N_PREV
ds= 1
topic ='War Crimes and Criminals'

doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
summary, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
model = pickle.load(open(_model_path+topic+".pkl","rb"))
len_ind = _length_indicator(ds=ds,topic=topic)
# doc = doc[:5]
# summary = summary[:5]

p_selected = data_path+"model_input/FFNN/selected_sentences"+str(ds)+".json"
with open(p_selected,'r') as f:
	selected_tot = json.load(f) # indices of article sentences selected as summary
# assert len(selected_tot)==len(doc)==len(summary)

_, flats_cand = model.viterbi(doc)
_, flats_sum = model.viterbi(summary)

transition = model._trans
prior = model._priors
print("transition matrix shape",transition.shape)


def _similarity(sum_sents, cand_sents):
	## return cosine similarity (#cand_sents,N_PREV)
	x = np.zeros((len(cand_sents),N_PREV))
	words_sum = [set(c) for c in sum_sents]
	words_cands = [set(c) for c in cand_sents]
	offset = N_PREV - len(sum_sents)
	for i in range(len(cand_sents)):
		for j in range(len(sum_sents)):
			x[i][j+offset] = len(words_sum[j].intersection(words_cands[i]))/math.sqrt(len(words_sum[j])*len(words_cands[i]))
	return x


def _m_freq(freq_map,doc,cands_idx):
	# calculate importance score (avg) for indexed candidates in document. return (#cands_idx, 1)
	x = np.zeros(len(cands_idx))
	for i,d in enumerate(cands_idx):
		cand_idx = [word2idx[k] for k in doc[d][1:-1]]
		x_=0
		for idx in cand_idx:
			x_ += freq_map[idx]
		x_ /= len(cand_idx)
		x[i] = x_


X = np.zeros(n_dim)
idx_cand = 0
idx_sum = 0
for i in range(len(doc)):
	selected_idx=selected_tot[i]
	flat_c = flats_cand[idx_cand:idx_cand+len_ind[i]]
	flat_s = flats_sum[idx_sum:idx_sum+len(selected_idx)]
	## build frequency dictionary
	d = doc[i]
	doc_set = [set(a[1:-1]) for a in d]
	doc_words = reduce((lambda a,b: a.union(b)),doc_set)
	print("|V| in %sth source: %s" %(i,len(doc_words)))
	word2idx = dict(zip(list(doc_words),range(len(doc_words))))
	freq_map = np.zeros(len(doc_words))
	for sent in d:
		for w in sent[1:-1]:
			freq_map[word2idx[w]] += 1
	freq_map = freq_map/np.sum(freq_map) # normalize to probability

	## start of the article
	if selected_idx[0] < NUM_CAND:
		n_i = min(NUM_CAND,len_ind[i])
	else:
		n_i = int(selected_idx[0]+1)
	cands = np.array(flat_c[:n_i])
	X_ = np.zeros((int(n_i),n_dim))
	X_[...,0] = prior[cands] # P(S_cand) as prior
	print("\n>>%sth doc, n = %s. selected idx:%s. first fill in %s rows."%(i, len_ind[i],selected_idx, n_i))
	X_[...,1] = np.arange(n_i) +1 # +1 if no summary yet?
	X_[...,2] = _m_freq(freq_map,d,range(n_i))
	
	## loop inside all indicies
	for idx_ in range(len(selected_idx)-1):
		cur_idx = selected_idx[idx_]
		next_idx = selected_idx[idx_+1]
		n_i = min(NUM_CAND+cur_idx+1, len_ind[i]) - cur_idx -1 if next_idx<NUM_CAND+cur_idx else (next_idx-cur_idx)
		print("selected idx:%s. next locally fill in rows[%s-%s). n_i=%s"%(cur_idx, cur_idx+1,cur_idx+n_i+1,n_i))
		x = np.zeros((int(n_i),n_dim))
		if n_i != 0:
			## P(S_cand|S_last summary)
			f_s = np.array(flat_s[idx_]) # index from summary
			f_c = np.array(flat_c[int(cur_idx+1):int(cur_idx+n_i+1)])
			x[...,0] = transition[f_s,f_c]
			## pos(cand) - pos(last summary)
			x[...,1] = np.arange(n_i)
			## Importance score by frequence
			x[...,2] = _m_freq(freq_map,d,range(int(cur_idx+1),int(cur_idx+n_i+1)))
			## Sim(cand, last summary)
			last_sums = summary[i][idx_-(N_PREV-1):idx_+1] if idx_>2 else summary[i][:idx_+1]
			cands = doc[i][int(cur_idx+1):int(cur_idx+n_i+1)]
			x[...,-N_PREV:] = _similarity(last_sums, cands)

			X_ = np.vstack((X_,x))
			
	## Last row
	n_i = min(NUM_CAND+selected_idx[-1]+1, len_ind[i]) - selected_idx[-1] -1
	print("Lastly selected idx:%s. lastly locally fill in rows[%s-%s)."%(selected_idx[-1],selected_idx[-1]+1,selected_idx[-1]+n_i+1))
	if n_i!=0:
		x = np.zeros((int(n_i),n_dim))
		f_s = np.array(flat_s[-1])
		f_c = np.array(flat_c[int(selected_idx[-1]+1):int(selected_idx[-1]+1+n_i)])
		x[...,0] = transition[f_s,f_c]
		x[...,1] = np.arange(n_i)
		x[...,2] = _m_freq(freq_map,d,range(int(selected_idx[-1]+1),int(selected_idx[-1]+1+n_i)))

		cands = doc[i][int(selected_idx[-1]+1):int(selected_idx[-1]+n_i+1)]
		sums = summary[i][-N_PREV:]
		x[...,-N_PREV:] = _similarity(sums, cands)

		X_ = np.vstack((X_,x))
	print("X_.shape",X_.shape)
	X = np.vstack((X,X_))
	idx_cand += len_ind[i]
	idx_sum += len(selected_idx)
	
X = X[1:]
print("X_interac_nprev.shape",X.shape)
np.save(savename+str(ds),X)