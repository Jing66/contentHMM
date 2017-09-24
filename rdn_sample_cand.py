import numpy as np
import json
import pickle
import math
import sys
import os
import json
from collections import Counter

from scipy import io
from functools import reduce


from all_feat import _feat_source,_feat_sum_sofar,_feat_cand_noninterac, _length_indicator, FEAT2COL, CATE_FEAT
sys.path.append(os.path.abspath('..'))
from content_hmm import *

#######################################
# Random sample from next sentences, if not enough, sample from previous sentences within source. 
# for each source and summary: [10 candidates + EOS] where EOS is a separate sentence
#######################################

START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"


data_path = "/home/ml/jliu164/code/data/model_input/"
cand_rec_path = data_path+"FFNN/cand_record/"
src_path = "/home/ml/jliu164/code/Summarization/"
input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
# input_path = data_path+"/seq_input/"


NUM_CAND = 10
np.random.seed(7)


def _EOS_embedding():
	# generate embedding for EOS
	try:
		eos_emb = np.load(data_path+"EOS_embedding.npy")
	except:
		print("No embedding for EOS")
		with open("/home/ml/jliu164/code/data/we_file.json") as f:
			We = json.load(f)
		eos_emb = We["."]
		np.save(data_path+"EOS_embedding",np.array(eos_emb))
	return eos_emb

EOS_embedding = _EOS_embedding()


def _x_EOD(source, summary, n_dim, len_source, last_pos):
	## feature vector for <end of summary>. Source and summary are (1,dim(feat)) vectors.
	x = np.zeros(n_dim)
	x[0:len(source)] = source
	x[len(source):len(source)+len(summary)] = summary
	x[len(source)+len(summary)] = len_source+1
	x[len(source)+len(summary)+1] = 6 # bucket of position

	pos_emb = FEAT2COL["cand_se"]
	x[pos_emb[0]:pos_emb[1]] = EOS_embedding
	x[pos_emb[1]+2] = len_source - last_pos
	return x


def make_X_rdn(ds = 1, topic = 'War Crimes and Criminals', savedir = data_path+"FFNN/",save=True):
	### randomly sample from candidates (if the next 10 sentences doesn't reach ) and add EOS
	## load components
	X_source = _feat_source(ds=ds,topic=topic) #(#articles,...)
	print("X_source.shape",X_source.shape)
	try:
		X_summary = np.load(savedir+"X_summary"+str(ds)+".npy") # (summary sentences,...)
	except IOError:
		print("summary part not ready...generating...")
		
	print("X_summary.shape",X_summary.shape)
	try:
		X_cand = np.load(savedir+"X_cand"+str(ds)+".npy")  #(#n_sentences in source,...)
	except IOError:
		print("candidate part not ready...generating...")
	print("X_candidate.shape",X_cand.shape)
	p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	len_ind = _length_indicator(ds=ds,topic=topic)
	
	cand_rec = [] # save the indicies where sample is needed. won't save EOS

	## feat: [P(S_cand|S_last summary)] + [Sim(cand, last summary)]
	X_interac = np.load(savedir+"X_interac_rdn"+str(ds)+".npy")
	print("X_interac.shape",X_interac.shape)
	## feat: [Verb overlap] + [Noun overlap] of candidate vs. summary so far
	# X_lexical = np.load(savedir+"X_lexical_rdn"+str(ds)+".npy")
	# print("X_lexical.shape",X_lexical.shape)

	feats = [X_source,X_summary,X_cand] # add X_interac,X_lexical later
	# X = np.zeros(sum([x.shape[1] for x in feats]))
	dim_later = 26
	dim_eod = sum([x.shape[1] for x in feats]) +dim_later# total dim for <EOS> 
	X = np.zeros(dim_eod)
	print("dim_eod",dim_eod)
	print("X_init shape",X.shape)
	
	cand_len = [] # length of each set of candidate
	idx_sum = 0
	idx_cand = 0
	# selected_tot = selected_tot[:5]
	for idx_source in range(len(selected_tot)):
		cand_rec_ = []
		selected_idx = np.array(selected_tot[idx_source]).astype(int)
		print("\n")
		# locally reference vectors
		source_vec = X_source[idx_source]
		sum_vec = X_summary[idx_sum:idx_sum+len(selected_idx)+1]
		cand_vec = X_cand[idx_cand: idx_cand+len_ind[idx_source]]
	
		eos_x = _x_EOD(source_vec, sum_vec[0],dim_eod,len_ind[idx_source],-1)

		idx_sum += len(selected_idx)+1
		idx_cand += len_ind[idx_source]
		## before the first sentence getting selected as summary
		if selected_idx[0] < NUM_CAND:
			n_i = np.arange(min(NUM_CAND,len_ind[idx_source])) # not sample--true in arange
		else:
			n_i = np.random.choice(selected_idx[0] -1,NUM_CAND-1,replace=False) # indicies of sampled candidate.
			n_i = np.append(n_i,selected_idx[0]) # 9 random sample + true candidate

		print(">>n = %s. selected idx:%s. first fill in %s rows."%(len_ind[idx_source],selected_idx, n_i))
		
		tmp = np.hstack((source_vec,sum_vec[0]))[np.newaxis,:]
		tmp = np.broadcast_to(tmp, (len(n_i), tmp.shape[1]))
		tmp = np.hstack((tmp,cand_vec[n_i]))
		x_prev = np.hstack((tmp, np.zeros((len(n_i),dim_later))))
		x_prev = np.vstack((x_prev, eos_x)) # 10 candidates + <EOS>
		
		cand_rec_.append(n_i)
		cand_len.append(len(n_i)+1)
		## loop inside all indicies. Summary so far: cur_idx. predict <next_idx> always at the end or in range
		for idx_ in range(len(selected_idx)-1):
			cur_idx = selected_idx[idx_] # last idx of summary so far
			next_idx = selected_idx[idx_+1]

			if len_ind[idx_source]<NUM_CAND: # [2],5
				n_i = np.arange(len_ind[idx_source])

			elif cur_idx+NUM_CAND < next_idx: # sample. [14,26] or [4,16] or [10,20] n=30
				p = np.ones(len_ind[idx_source]).astype(np.float32)
				p[next_idx] =0
				p/=np.sum(p)
				n_i = np.random.choice(len_ind[idx_source],size=min(NUM_CAND-1,len_ind[idx_source]),p=p,replace=False) # completely random sample
				n_i = np.append(n_i,next_idx) # + true candidate
			else: # arange + sample.[14,16] or [24,26] n=30
				n_i = np.arange(cur_idx,len_ind[idx_source])[:NUM_CAND]
				
				if next_idx:
					rdn = np.random.choice(next_idx ,size=NUM_CAND - len(n_i),replace=False) # if next=[0], sample from all sentences
				else:
					p = np.ones(len_ind[idx_source]).astype(np.float32)
					p[next_idx] = 0
					p/= np.sum(p)
					rdn = np.random.choice(len_ind[idx_source],size=NUM_CAND-len(n_i),p=p,replace=False)
				if len(rdn):
					n_i = np.concatenate((n_i,rdn)) # true pred in range
			
			_sum_vec = sum_vec[idx_+1]
			_cand_vec = cand_vec[n_i]
			print("selected idx:%s.summary index:%s. next locally fill n_i=%s"%(cur_idx,idx_+1, n_i))
			cand_len.append(len(n_i)+1)
			cand_rec_.append(n_i)
			
			eos_x = _x_EOD(source_vec, _sum_vec,dim_eod,len_ind[idx_source],cur_idx)
			
			tmp = np.hstack((source_vec, _sum_vec))[np.newaxis,:]
			tmp = np.broadcast_to(tmp, (_cand_vec.shape[0],tmp.shape[1])) 
			tmp_ = np.hstack((tmp,_cand_vec))
			tmp_ = np.hstack((tmp_, np.zeros((len(n_i),dim_later))))

			x_prev = np.vstack((x_prev, tmp_))
			x_prev = np.vstack((x_prev, eos_x)) # 10 candidates + <EOS>
		
		## Last row: predict EOS
		if len_ind[idx_source]<NUM_CAND: # [2],5
			n_i = np.arange(len_ind[idx_source])
		elif selected_idx[-1]+NUM_CAND >= len_ind[idx_source]: # [26], n=30;
			non_rdn = np.arange(selected_idx[-1]+1,len_ind[idx_source])[:NUM_CAND]
			n_i = np.concatenate((non_rdn, np.random.choice(int(max(selected_idx[-1],len_ind[idx_source])),NUM_CAND-len(non_rdn),replace=False)))
		else: # [10],19
			n_i = np.random.choice(np.arange(selected_idx[-1]+1,len_ind[idx_source]),NUM_CAND,replace=False)
		
		print("Lastly selected idx:%s. lastly locally fill in rows n_i = %s."%(selected_idx[-1],n_i))
		cand_len.append(len(n_i)+1)
		cand_rec_.append(n_i)

		_sum_vec = sum_vec[-1]
		_cand_vec = cand_vec[n_i]

		eos_x = _x_EOD(source_vec, _sum_vec,dim_eod,len_ind[idx_source],selected_idx[-1]) 
		tmp = np.hstack((source_vec, _sum_vec))[np.newaxis,:]
		tmp = np.broadcast_to(tmp, (_cand_vec.shape[0],tmp.shape[1])) 
		tmp_ = np.hstack((tmp,_cand_vec))
		tmp_ = np.hstack((tmp_, np.zeros((len(n_i),dim_later))))
		x_prev = np.vstack((x_prev, tmp_))
		x_prev = np.vstack((x_prev, eos_x)) # 10 candidates + <EOS>

		print("local x shape",x_prev.shape)
		X = np.vstack((X, x_prev))
		cand_rec.append(cand_rec_)

	print("# candidate_length:",len(cand_len))
	X = X[1:]
	X[...,-26:-20] = X_interac
	# X[...,-20:] = X_lexical
	print("X.shape",X.shape)
	
	if save:
		np.save(savedir+"X_rdn"+str(ds),X)
		np.save(savedir+"candidate_length_rdn"+str(ds),np.array(cand_len))
		pickle.dump(cand_rec, open(cand_rec_path+"rdn_sample"+str(ds)+".pkl","wb"))



def make_Y_rdn(ds = 1, topic = 'War Crimes and Criminals', savename = data_path+"FFNN/Y_rdn"):
	len_ind = _length_indicator(ds=ds,topic=topic)
	p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	
	Y = np.zeros(1,dtype=int)

	for i in range(len(selected_tot)):
		selected_idx = np.array(selected_tot[i])
		
		## first row
		if selected_idx[0] < NUM_CAND:
			n_i = min(NUM_CAND,len_ind[i]) # not sample--true in arange
			indicies = [selected_idx[0]]
		else:
			n_i = NUM_CAND # the last one is true candidate
			indicies = [NUM_CAND-1]
		n_i += 1
		print("\n >>Article length = %s, selected = %s, initial fill %s rows" %(len_ind[i], selected_idx,n_i))
		
		## middle rows
		for j in range(len(selected_idx)-1):
			cur_idx = selected_idx[j]
			next_idx = selected_idx[j+1]
			if len_ind[i]<NUM_CAND: 
				indicies.append(n_i+next_idx - cur_idx)
				n_i += len_ind[i]
			elif cur_idx+NUM_CAND < next_idx: 
				n_i += NUM_CAND
				indicies.append(n_i-1)
			else: 
				indicies.append(n_i+next_idx - cur_idx)
				n_i += NUM_CAND
			n_i += 1
		# last row
		n_i += min(NUM_CAND,len_ind[i])+1
		indicies.append(n_i-1)
		print("Y_i has rows total %s, label index are %s"%(n_i, indicies))

		y = np.zeros(int(n_i),dtype=np.int32)
		mask = np.array(indicies).astype(int)
		y[mask] = 1	
		Y = np.hstack((Y,y))

	Y = Y[1:]
	print("Y.shape",Y.shape)
	
	if savename:
		np.save(savename+str(ds), Y)
	return Y




if __name__ == '__main__':
	make_X_rdn(ds=1,save=True)
	# make_Y_rdn(ds=1)
