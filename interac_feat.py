# interaction between candidate and summary
# P(S_cand|S_last summary)
# Sim(cand, last summary). 0 if no summary yet?
# pos(cand)-pos(last summary). +1 if no summary yet?
import numpy as np
import json
import pickle
import math
import sys
import os
import math

from new_baseline import _length_indicator
sys.path.append(os.path.abspath('..'))
from content_hmm import *

input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
data_path = "/home/ml/jliu164/code/data/"
savedir = data_path+"model_input/FFNN/"
NUM_CAND = 10

n_dim = 3
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
assert len(selected_tot)==len(doc)==len(summary)

_, flats_cand = model.viterbi(doc)
_, flats_sum = model.viterbi(summary)

transition = model._trans
prior = model._priors
print("transition matrix shape",transition.shape)


def _similarity(sum_sent, cand_sents):
	# return cosine similarity (#cand_sents,)
	x = np.zeros(len(cand_sents))
	words_sum = set(sum_sent)
	words_cands = [set(c) for c in cand_sents]
	for i in range(len(cand_sents)):
		x[i] = len(words_sum.intersection(words_cands[i]))/math.sqrt(len(words_sum)*len(words_cands[i]))
	return x

X = np.zeros(n_dim)
idx_cand = 0
idx_sum = 0
for i in range(len(doc)):
	selected_idx=selected_tot[i]
	flat_c = flats_cand[idx_cand:idx_cand+len_ind[i]]
	flat_s = flats_sum[idx_sum:idx_sum+len(selected_idx)]
	## start of the article
	if selected_idx[0] < NUM_CAND:
		n_i = min(NUM_CAND,len_ind[i])
	else:
		n_i = int(selected_idx[0]+1)
	cands = np.array(flat_c[:n_i])
	X_ = np.zeros((int(n_i),n_dim))
	X_[...,0] = prior[cands] # P(S_cand) as prior
	print("\n>>%sth doc, n = %s. selected idx:%s. first fill in %s rows."%(i, len_ind[i],selected_idx, n_i))
	X_[...,2] = np.arange(n_i) +1 # +1 if no summary yet?
	
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
			## Sim(cand, last summary)
			last_sum = summary[i][idx_]
			cands = doc[i][int(cur_idx+1):int(cur_idx+n_i+1)]
			x[...,1] = _similarity(last_sum, cands)

			## pos(cand) - pos(last summary)
			x[...,2] = np.arange(n_i)
			
			X_ = np.vstack((X_,x))
			
	## Last row
	n_i = min(NUM_CAND+selected_idx[-1]+1, len_ind[i]) - selected_idx[-1] -1
	print("Lastly selected idx:%s. lastly locally fill in rows[%s-%s)."%(selected_idx[-1],selected_idx[-1]+1,selected_idx[-1]+n_i+1))
	if n_i!=0:
		x = np.zeros((int(n_i),n_dim))
		f_s = np.array(flat_s[-1])
		f_c = np.array(flat_c[int(selected_idx[-1]+1):int(selected_idx[-1]+1+n_i)])
		x[...,0] = transition[f_s,f_c]
		
		cands = doc[i][int(selected_idx[-1]+1):int(selected_idx[-1]+n_i+1)]
		x[...,1] = _similarity(summary[i][-1], cands)
		x[...,2] = np.arange(n_i)
		
		X_ = np.vstack((X_,x))
	print("X_.shape",X_.shape)
	X = np.vstack((X,X_))
	idx_cand += len_ind[i]
	idx_sum += len(selected_idx)
	
X = X[1:]
print("X_interac.shape",X.shape)
np.save(savedir+"X_interac"+str(ds),X)