import numpy as np
import json
import pickle
import math
import sys
import os
import json
from scipy import io
from functools import reduce

from all_feat import _feat_source,_feat_sum_sofar,_feat_cand_noninterac, _length_indicator
from rdn_sample_cand import _x_EOD

sys.path.append(os.path.abspath('..'))
from content_hmm import *

############################
# Learning with different difficulty. 1). sample from other source. 2). sample from other sentences from summary
############################

START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"

FEAT2COL = {"src_cluster":(0,10),"src_se":(10,310),
	"sum_cluster":(310,320),"sum_overlap":(320,321),"sum_pos":(321,322),"sum_posbin":(322,323),"sum_num":(323,324),"sum_se":(324,624),
	"cand_pos":(624,625),"cand_posbin":(625,626),"cand_cluid":(626,627),"cand_prob":(627,638),"cand_M":(637,639),"cand_se":(639,939),
	"interac_trans":(939,940),"interac_pos":(940,941),"interac_M":(941,942),"interac_sim_nprev":(942,945),"interac_w_overlap":(945,947),"interac_emis":(946,964)}

CATE_FEAT = {"sum_overlap","sum_pos","sum_num","cand_pos","cand_cluid","interac_pos","interac_overlap"} # cannot normalize

data_path = "/home/ml/jliu164/code/data/model_input/"
cand_rec_path = data_path+"FFNN/cand_record/"
src_path = "/home/ml/jliu164/code/Summarization/"
input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"


NUM_CAND = 10
np.random.seed(7)



############## Easy Version: sample candidates from other source #################
def make_X_easy(ds = 1, topic = 'War Crimes and Criminals', savedir = data_path+"FFNN/",save=True):
	## load components
	X_source = _feat_source(ds=ds,topic=topic) #(#articles,...)
	print("X_source.shape",X_source.shape)
	try:
		X_summary = np.load(savedir+"X_summary"+str(ds)+".npy") # (summary sentences,...)
	except IOError:
		print("summary part not ready")	
	print("X_summary.shape",X_summary.shape)
	try:
		X_cand = np.load(savedir+"X_cand"+str(ds)+".npy")  #(#n_sentences in source,...)
	except IOError:
		print("candidate part not ready")
	print("X_candidate.shape",X_cand.shape)
	p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	len_ind = _length_indicator(ds=ds,topic=topic)
	assert np.sum(len_ind)==X_cand.shape[0]

	feats = [X_source,X_summary,X_cand] # add X_interac,X_lexical later
	dim_later = 26
	dim_eod = sum([x.shape[1] for x in feats]) +dim_later# total dim for <EOS> 
	X = np.zeros(dim_eod)
	print("dim_eod",dim_eod)
	print("X_init shape",X.shape)

	cand_rec = [] # save sampled sentences
	cand_len = [] # save length of each set of candidate

	idx_sum = 0
	idx_cand = 0
	for idx_source in range(len(selected_tot)):
		cand_rec_ = []
		selected_idx = np.array(selected_tot[idx_source]).astype(int)
		print("\n>>article no.%s, n = %s. selected index = %s"%(idx_source,len_ind[idx_source],selected_idx))
		## locally reference vectors
		source_vec = X_source[idx_source]
		sum_vec = X_summary[idx_sum:idx_sum+len(selected_idx)+1]
		cand_vec_true = X_cand[idx_cand: idx_cand+len_ind[idx_source]]

		## only sample from other sources. + true + EOS
		p_sample = np.ones(X_cand.shape[0])
		p_sample[idx_cand: idx_cand+len_ind[idx_source]] = 0
		assert np.sum(p_sample)==len(p_sample)-len_ind[idx_source], len(np.argwhere(p_sample==0))
		p_sample/= np.sum(p_sample)

		idx_sum += len(selected_idx)+1
		idx_cand += len_ind[idx_source]

		x = np.zeros(dim_eod)
		for idx_ in range(len(selected_idx)):
			cur_idx = -1 if idx_==0 else selected_idx[idx_-1] # index of last selection
			next_idx = selected_idx[idx_]
			n_i = np.random.choice(X_cand.shape[0], NUM_CAND-1,p=p_sample)
			print(">>sampled indicies from all other source articles: %s."%( n_i))
			cand_vec = X_cand[n_i]
			cand_vec = np.vstack((cand_vec,cand_vec_true[next_idx])) # true next sentence

			tmp = np.hstack((source_vec,sum_vec[idx_]))[np.newaxis,:]
			tmp = np.broadcast_to(tmp, (cand_vec.shape[0], tmp.shape[1]))
			tmp = np.hstack((tmp,cand_vec))

			x_prev = np.hstack((tmp, np.zeros((cand_vec.shape[0],dim_later))))
			## append EOS
			eos_x = _x_EOD(source_vec, sum_vec[idx_],dim_eod,len_ind[idx_source], cur_idx)
			x_prev = np.vstack((x_prev, eos_x)) # 10 candidates + <EOS>
			
			cand_rec_.append(n_i)
			cand_len.append(len(n_i)+2)
			x = np.vstack((x,x_prev))

		## last row to predict EOS: sample 10 instead of 9
		n_i = np.random.choice(X_cand.shape[0], NUM_CAND,p=p_sample)
		print("Lastly sampled indicies from all other source articles: %s."%( n_i))
		cand_rec_.append(n_i)
		cand_rec.append(cand_rec_)
		cand_len.append(len(n_i)+1)

		cand_vec = X_cand[n_i]
		tmp = np.hstack((source_vec,sum_vec[-1]))[np.newaxis,:]
		tmp = np.broadcast_to(tmp, (len(n_i), tmp.shape[1]))
		tmp = np.hstack((tmp,cand_vec))
		x_prev = np.hstack((tmp, np.zeros((len(n_i),dim_later))))
		## append EOS
		eos_x = _x_EOD(source_vec, sum_vec[-1],dim_eod,len_ind[idx_source], selected_idx[-1])
		x_prev = np.vstack((x_prev, eos_x)) # 10 candidates + <EOS>

		x = np.vstack((x,x_prev))
		x = x[1:]

		X = np.vstack((X,x))

	X = X[1:]
	print("X_shape,",X.shape)

	if save:
		np.save(savedir+"X_rdn_easy"+str(ds),X)
		np.save(savedir+"candidate_length_easy"+str(ds),np.array(cand_len))
		pickle.dump(cand_rec, open(cand_rec_path+"rdn_sample_easy"+str(ds)+".pkl","wb"))
		









if __name__ == '__main__':
	make_X_easy(ds=1,save=True)