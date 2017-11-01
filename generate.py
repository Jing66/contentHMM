import numpy as np
import pickle
import sys
import os
import json
from collections import Counter
import h5py
import re
import math
from functools import reduce
from scipy import stats
import matplotlib.pyplot as plt

from rdn_sample_cand import EOS_embedding, _x_EOD
from utils import _gen_file_for_M as _gen_X4M
from utils import pca_we
from all_feat import feat_select, FEAT2COL
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from interac_feat import _similarity, N_PREV, _m_freq
if sys.version_info[0] >=3:
	from pred_M import M_predict
	from eval_model import rouge_score

# else:
# 	sys.path.append(os.path.abspath('..'))
# 	from content_hmm import ContentTagger
# 	from corenlpy import AnnotatedText as A

topic ='War Crimes and Criminals'
input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
data_path = "/home/ml/jliu164/code/data/"
cur_path = "/home/ml/jliu164/code/Summarization/"
utils_path = "/home/ml/jliu164/code/data/utils/"
save_path = "/home/ml/jliu164/code/filter_results/topic2files(content).pkl"
corpus_path =  "/home/rldata/jingyun/nyt_corpus/content_annotated/"
fail_path = '/home/ml/jliu164/code/contentHMM_input/fail/'
extract_dir = "/home/ml/jliu164/code/contentHMM_extract/contents/"


START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"
UNK = "**UNK**"
SKIP_SET = {SOD,END_SENT,START_SENT}
epsilon = 1e-8
N_CAND = 10
DIM_EMB = 300
BEAM_SIZE = 3
EOS = N_CAND+1 # index for EOS is always going to be this number

pos2idx = json.load(open("../data/utils/pos2idx2.json"))
lemma2idx = json.load(open("../data/utils/lemma2idx2.json"))
idx2lemma = inverted_dict = dict([[v,k] for k,v in lemma2idx.items()])
lemma2pos = json.load(open("../data/utils/lemma2pos2.json"))
lemma2ner = json.load(open("../data/utils/lemma2ner2.json"))

pos2idx_prev = json.load(open("../data/utils/pos2idx1.json")) # force same index as training
ner2idx_prev = json.load(open("../data/utils/ner2idx1.json"))


def generate_summary(idx, source, model, weights, rm ={},n_pca = 0, force_length = np.inf):
	""" 
	Given source article and content model, no gold standard summary, generate a summary.
	source: list of sentences, each list of words. model: content HMM model. weights: list of numpy arrays for doing prediction
	"""
	vectors, utils, dims = load_vectors(idx, source, model,n_pca)
	X_src, X_cand ,x_se,X_prev = vectors
	state_prob, emis_dense, flat,freq_map,word2idx,pca = utils
	dim_src,dim_sum,n_dim = dims

	## [summary so far]
	eos = False
	chosen = np.array([-1]).astype(int) # indicies of selected summary
	count = 0
	while not eos and chosen[-1] < len(source)-1:

		# print("\n>>generating %s summary sentence:"%(count))
		## feature vector for summary so far. same for all candidates thus n_row = 1
		X_sum = np.zeros(dim_sum) 
		X_sum[:4+model._m] = sum_so_far(source,state_prob, chosen,model)
		X_sum[-DIM_EMB:] = np.mean(x_se[chosen[1:]],axis=0) if len(chosen)>1 else np.zeros(DIM_EMB) # average all chosen summary embeddings. ALT: last embedding
		X_fixed = np.hstack((X_src, X_sum)) # fixed part of the vector

		## select candidates
		n_i = np.arange(chosen[-1]+1, min(len(source),N_CAND+chosen[-1]+1)) # boundary for #candidates
		
		## feature vector for interaction
		X_interac = interac(source, flat, chosen, n_i, model, freq_map,word2idx, emis_dense)
		eos_vec = _x_EOD(X_src, X_sum,X_prev[-1], n_dim, len(source), chosen[-1]) ## feature vector for eos. appended at the end
		x_cand = np.concatenate((X_prev[n_i],X_prev[n_i+1],X_cand[n_i], X_interac),axis=1)

		tmp = np.broadcast_to(X_fixed, (len(n_i), X_fixed.shape[0]))
		x_cand = np.hstack((tmp, x_cand))
		X = np.vstack((x_cand, eos_vec))
		print("X.shape",X.shape)
		
		## Forward pass: fn(X(11 x n_dim) * W(n_dim x 1)) => y_hat(11 x 1)
		X = feat_select(X, None, rm, n_pca = n_pca, pca = pca)
		y_hat = forward(X, weights)
		
		target = np.argmax(y_hat)
		print("yp.shape",y_hat.shape,"argmax",target)

		## If we let model to decide when to stop and it predict EOS , OR we force oracle length and we've reached that
		if (force_length == np.inf and target == y_hat.shape[0]-1 and len(chosen) > 1) or force_length != np.inf and len(chosen) > force_length:
			eos = True
			print("<EOS>")
		elif target == y_hat.shape[0] -1:
			chosen = np.append(chosen,np.argmax(y_hat[:-1]) + chosen[-1]+1)
		else:
			chosen = np.append(chosen,(target + chosen[-1]+1))

		print("chosen summary indicies",chosen)	
		count +=1
	
	return chosen[1:]


def search_summary(idx, source, model, weights, rm ={},n_pca = 0):
	"""beam search with no oracle length."""
	## load everthing needed
	vectors, utils, dims = load_vectors(idx, source, model,n_pca)
	X_src, X_cand ,x_se,X_prev = vectors
	state_prob, emis_dense, flat,freq_map,word2idx,pca = utils
	dim_src,dim_sum,n_dim = dims

	eos = np.full(BEAM_SIZE, 0) # indicate EOS of a path. 0 if keep going
	chosen = [] # indicies of selected summary. each element is a path
	for _ in range(BEAM_SIZE):
		chosen.append(np.array([-1]))
	best_scores = np.zeros(BEAM_SIZE)
	## main loop for search
	while not eos.all():
		stop_idx = np.argwhere(eos).ravel() # which paths stopped expanding cuz of <EOS>
		print("\n>> New iteration. stop_idx is %s"%(stop_idx))
		X_sum = np.zeros((np.sum(eos==0),dim_sum)) # each 10cand has a summary so far
		sum_idx = 0
		for i in range(BEAM_SIZE):
			if not eos[i]:
				X_sum[sum_idx,:4+model._m] = sum_so_far(source,state_prob, chosen[i][1:],model)
				X_sum[sum_idx,-DIM_EMB:] = np.mean(x_se[chosen[i][1:]],axis=0) if chosen[i][-1]!=-1 else np.zeros(DIM_EMB) # average all chosen summary embeddings. ALT: last embedding
				sum_idx +=1
		tmp = np.broadcast_to(X_src, (np.sum(eos==0), X_src.shape[0]))
		X_fixed = np.hstack((tmp, X_sum))
		# print("X_fixed shape",X_fixed.shape)

		## select candidates
		if  chosen[0][-1]==-1:
			n_i = np.arange( min(len(source),N_CAND))
			n_i_list = [n_i] # if first span only do 10 candidates
		else:
			n_i_list = [np.arange(c[-1]+1, min(len(source),N_CAND+c[-1]+1)) for i,c in enumerate(chosen) if not eos[i]] # raw list of candidates
		print("# candidates for each n_i",[len(i) for i in n_i_list])
		## feature vector for interaction
		X_interac = [] # same length as n_i_list
		sum_idx = 0
		for i in range(BEAM_SIZE):
			if not eos[i] and sum_idx < len(n_i_list):
				X_interac_ = interac(source, flat, chosen[sum_idx][1:], n_i_list[sum_idx], model, freq_map,word2idx, emis_dense)
				X_interac.append(X_interac_)
				sum_idx += 1
		# print("interac shape",[i.shape for i in X_interac])
		
		
		eos_vec = [] # ignore EOS for the first round
		if len(chosen[0]>1):
			sum_idx = 0
			for i in range(BEAM_SIZE):
				if not eos[i] and sum_idx < len(n_i_list):
					eos_= _x_EOD(X_src, X_sum[sum_idx],X_prev[-1], n_dim, len(source),chosen[sum_idx][-1])
					# eos_ = _x_EOD(X_src, X_sum[sum_idx], n_dim, len(source), chosen[sum_idx][-1]) ## feature vector for eos. appended at the end
					eos_vec.append(eos_)
					sum_idx += 1
	
		print("# EOS vector",len(eos_vec))

		x_cand = [] # same length as n_i_list
		for i,n_i in enumerate(n_i_list):
			x_cand_ = np.concatenate((X_prev[n_i],X_prev[n_i+1],X_cand[n_i], X_interac[i]),axis=1)
			# x_cand_ = np.hstack((X_cand[n_i], X_interac[i]))
			x_cand.append(x_cand_)
		# print("X cand+interac.shape",x_cand.shape)

		x = [] # same length as n_i_list
		for i,n_i_ in enumerate(n_i_list):
			tmp = np.broadcast_to(X_fixed[i], (len(n_i_), X_fixed.shape[1]))
			x_ = np.hstack((tmp, x_cand[i]))
			x.append(x_)
		# print("X cand+src+summary.shape",x_cand.shape)
		
		X = [np.vstack((x_, eos_vec[i])) for i,x_ in enumerate(x) ] if len(chosen[0]>1) else x
	
		
		## Forward pass: fn(X(11 x n_dim) * W(n_dim x 1)) => y_hat(11 x 1)
		X = [feat_select(x, None, rm, n_pca = n_pca, pca = pca) for x in X]
		y_hats = [forward(x, weights).ravel() for x in X] 
		# print("y_hats",y_hats) # same length as n_i_list
		print("y_hats.shape",[y_hat.shape for y_hat in y_hats])

		## update avg score for each path
		new_scores = [] # [(score, which path <idx of n_i_list>, which candidate <idx of n_i>)]
		i_p = 0
		for i, y_hat in enumerate(y_hats):
			while i_p in stop_idx:
				i_p += 1
			new_score = (best_scores[i_p]*(len(chosen[i_p])-1) + y_hat)/len(chosen[i_p]) # new average
			new_scores.extend([(ns,i_p,ts) for ts,ns in enumerate(new_score)])
			i_p +=1
		for i in stop_idx:
			new_scores.append((best_scores[i], i, EOS))
		
		new_scores.sort(key = lambda tup:-tup[0])
		targets = [(p,c) for _, p,c in new_scores[:BEAM_SIZE]] #[(path, candidate)]
		best_scores = [s for s, _, _ in new_scores[:BEAM_SIZE]] # update best score
		
		print("targets",targets) 
		# print("best scores up until now",best_scores)

		last_chosen = [np.copy(c) for c in chosen]
		print("last chosen", last_chosen)
		
		## update targets in chosen
		for i in range(len(targets)):
			p,t = targets[i] # path, target index
			chosen_ = last_chosen[p] # find which path it comes from, indexing to the previous
			if t >= N_CAND and len(chosen[p])>1 or t+chosen_[-1]+1>=len(source):
				eos[i] = 1
			else:
				eos[i] = 0
				chosen_ = np.append(chosen_, t+chosen_[-1]+1)
			
			chosen[i] = chosen_

		print("new <EOS> indicator",eos)
		# print("best scores so far", best_scores)
		print("chosen summary indicies",chosen)	
	
	for i in range(BEAM_SIZE):
		if len(chosen[i][1:]) >0:
			return chosen[i][1:]



def search_summary_oracle(idx, source, model, weights, rm ={},n_pca = 0, oracle_len = 2):
	"""beam search with oracle length."""
	## load everthing needed
	vectors, utils, dims = load_vectors(idx, source, model,n_pca)
	X_src, X_cand ,x_se, X_prev= vectors
	state_prob, emis_dense, flat,freq_map,word2idx,pca = utils
	dim_src,dim_sum,n_dim = dims
	print("Required oracle length",oracle_len)
	eos = np.full(BEAM_SIZE, 0) # indicate EOS of a path. 0 if keep going
	chosen = [] # indicies of selected summary. each element is a path
	for _ in range(BEAM_SIZE):
		chosen.append(np.array([-1]))
	best_scores = np.zeros(BEAM_SIZE)
	eos_cand = np.full(BEAM_SIZE,0) # indicate next candidate should just be EOS
	## main loop for search
	while not eos.all():
		stop_idx = np.argwhere(eos).ravel() # which paths stopped expanding cuz of <EOS>
		print("\n>> New iteration. stop_idx is %s"%(stop_idx))
		X_sum = np.zeros((np.sum(eos==0),dim_sum)) # each 10cand has a summary so far
		sum_idx = 0
		for i in range(BEAM_SIZE):
			if not eos[i]:
				X_sum[sum_idx,:4+model._m] = sum_so_far(source,state_prob, chosen[i][1:],model)
				X_sum[sum_idx,-DIM_EMB:] = np.mean(x_se[chosen[i][1:]],axis=0) if chosen[i][-1]!=-1 else np.zeros(DIM_EMB) # average all chosen summary embeddings. ALT: last embedding
				sum_idx +=1
		tmp = np.broadcast_to(X_src, (np.sum(eos==0), X_src.shape[0]))
		X_fixed = np.hstack((tmp, X_sum))
		# print("X_fixed shape",X_fixed.shape)

		## select candidates
		if  chosen[0][-1]==-1:
			n_i = np.arange( min(len(source),N_CAND))
			n_i_list = [n_i] # if first span only do 10 candidates
		else:
			n_i_list = [np.arange(c[-1]+1, min(len(source),N_CAND+c[-1]+1)) for i,c in enumerate(chosen) if not eos[i]] # raw list of candidates
		print("# candidates for each n_i",[len(i) for i in n_i_list])
		
		eos_vec = []
		sum_idx = 0
		for i in range(BEAM_SIZE):
			if eos_cand[i] and sum_idx < len(n_i_list):
				eos_ = _x_EOD(X_src, X_sum[sum_idx], n_dim, len(source), chosen[sum_idx][-1]) ## feature vector for eos. appended at the end
				eos_vec.append(eos_)
				sum_idx += 1
		print("# EOS vector",len(eos_vec))

		x = [] # same length as n_i_list
		sum_idx = 0
		for i in range(BEAM_SIZE):
			if not eos_cand[i] and not eos[i] and sum_idx < len(n_i_list):
				X_interac_ = interac(source, flat, chosen[sum_idx][1:], n_i_list[sum_idx], model, freq_map,word2idx, emis_dense)
				print("X-interac shape",X_interac_.shape)
				x_cand_ = np.hstack((X_cand[n_i_list[sum_idx]], X_interac_))
				tmp = np.broadcast_to(X_fixed[sum_idx], (len(n_i_list[sum_idx]), X_fixed.shape[1]))
				x_ = np.hstack((tmp, x_cand_))
				x.append(x_)
				sum_idx +=1
		# print("X cand+src+summary.shape",[x_.shape for x_ in x])
		
		X = []
		sum_idx = 0
		eos_idx = 0
		for i in range(BEAM_SIZE):
			if eos_cand[i]:
				X.append(eos_vec[eos_idx].reshape((1,-1)))
				eos_idx += 1
			elif eos[i]:
				continue
			elif sum_idx < len(n_i_list):
				X.append(x[sum_idx])
				sum_idx +=1
		print("X shape",[X_.shape for X_ in X])

		## Forward pass: fn(X(11 x n_dim) * W(n_dim x 1)) => y_hat(11 x 1)
		X = [feat_select(x, None, rm, n_pca = n_pca, pca = pca) for x in X]
		y_hats = [forward(x, weights).ravel() for x in X] 
		# print("y_hats",y_hats) # same length as n_i_list
		print("y_hats.shape",[y_hat.shape for y_hat in y_hats])

		## update avg score for each path
		new_scores = [] # [(score, which path <idx of n_i_list>, which candidate <idx of n_i>)]
		i_p = 0
		for i, y_hat in enumerate(y_hats):
			# i_p = i + np.sum(stop_idx<=i)
			while i_p in stop_idx:
				i_p += 1
			# print(stop_idx<=i,"i_p",i_p)
			new_score = (best_scores[i_p]*(len(chosen[i_p])-1) + y_hat)/len(chosen[i_p]) # new average
			new_scores.extend([(ns,i_p,ts) for ts,ns in enumerate(new_score)])
			i_p += 1
		for i in stop_idx:
			new_scores.append((best_scores[i], i, EOS))
		
		new_scores.sort(key = lambda tup:-tup[0])
		targets = [(p,c) for _, p,c in new_scores[:BEAM_SIZE]] #[(path, candidate)]
		best_scores = [s for s, _, _ in new_scores[:BEAM_SIZE]] # update best score
		[best_scores.append(0) for _ in range(BEAM_SIZE-len(best_scores))] # pad with 0
		print("targets",targets) 
		print("best scores up until now",best_scores)

		last_chosen = [np.copy(c) for c in chosen]
		# print("last chosen", last_chosen)
		## update targets in chosen
		for i in range(len(targets)):
			p,t = targets[i] # path, target index
			chosen_ = last_chosen[p] # find which path it comes from, indexing to the previous
			
			if eos_cand[i]==1 or t+chosen_[-1]+1>=len(source):
				eos[i]= 1
			elif len(chosen_) == oracle_len:
				eos_cand[i]=1
				chosen_ = np.append(chosen_, t+chosen_[-1]+1)
			else:
				eos[i] = 0
				# eos_cand[i]=0
				chosen_ = np.append(chosen_, t+chosen_[-1]+1)
			chosen[i] = chosen_
		# source article too short
		for i in range(len(targets),BEAM_SIZE):
			eos[i]=1
			

		print("new <EOS> indicator",eos, "EOS cand indicator",eos_cand)
		print("chosen summary indicies",chosen)	
		
	return chosen[0][1:]

	



def sum_so_far(source,state_prob, indicies,model,n_bin=5):
	""" compute feature vec for summary so far, excluding embeddings cuz they can be added back later.
		indicies: array of index for which sentences are chosen as summary
		return (1 x dim_sum) vector
	"""
	X = np.zeros(model._m+4)
	## we don't have real summary distribution
	X[:model._m] = state_prob
	## if there has/hasn't been selected summary
	if len(indicies)>0:
		X[-4] = sum([len(source[i]) for i in indicies]) # n_word overlap == length of total summary sentences
		X[-3] = indicies[-1] # position of last chosen sentence
		bins = np.arange(-1,len(source)+1,float(len(source))/n_bin)
		X[-2] = np.digitize(indicies[-1],bins)

	else:
		X[-3] = -1
		X[-4] = 0
		X[-2] = 0
	X[-1] = len(indicies) # number of summary so far
	# print(X)
	return X


def build_x_prev(doc,model):
	## Lexical feature: top 1000 words unigram
	words2col = json.load(open(data_path+"model_input/FFNN/top1000unigram2idx.json","rb"))
	top_words = set(words2col.keys())
	x_top = np.zeros((len(doc),1000))
	row = 0
	for sent in doc:
		words = set(sent)
		intersec_idx = [words2col[i] for i in words.intersection(top_words)]	
		x_top[row][intersec_idx] += 1
		row += 1
	## Pr(sent) for all sentence
	x_pr = np.array([np.sum(model.forward_algo([sent])[-1]) for sent in doc])
	x_pr = x_pr.reshape((-1,1))
	## POS unigram. (#sentences, # POS tags). [i,j] = No. of the POS tag mapped by pos2idx appeared in sentence j
	x_pos = np.zeros((len(doc), len(pos2idx_prev))) #(None,33 )
	for i,sent in enumerate(doc):
		x = np.zeros(len(pos2idx_prev))
		for word in sent:
			if lemma2pos.get(word):
				x[pos2idx_prev[lemma2pos[word]]-1] += 1
		x_pos[i] = x
	## NER unigram
	x_ner = np.zeros((len(doc), len(ner2idx_prev))) #(None, 12)
	for i,sent in enumerate(doc):
		x = np.zeros(len(ner2idx_prev))
		for word in sent:
			if lemma2ner.get(word):
				x[ner2idx_prev[lemma2ner[word]]-1] += 1
		x_ner[i] = x
	X = np.concatenate((x_pr,x_top,  x_pos, x_ner),axis=1)
	X = np.vstack((np.zeros(X.shape[1]).reshape((1,-1)),X))
	return X

def interac(source, flat, sum_indicies,cand_indicies,model,freq_map, word2idx, emis_dense):
	""" compute feature vec for interaction
		return size (#candidate x dim_interac)
	"""
	flat = np.array(flat).astype(int)
	X = np.zeros((len(cand_indicies),8+2*(model._m-1))) #(??, 26)
	cand_sent = [source[i] for i in cand_indicies]
	sum_sent = [source[i] for i in sum_indicies]
	
	## Pr(S_cand|S_last summary)
	trans = model._trans
	prior = model._priors
	tmp = trans[flat[sum_indicies[-1]],flat[cand_indicies]] if len(sum_indicies)>0 else prior[flat[cand_indicies]]
	X[...,0] = tmp
	## pos(cand)-pos(last summary) 
	# X[...,1] = cand_indicies - sum_indicies[-1] if sum_indicies[-1]!=-1 else 0 
	X[...,1] = cand_indicies - sum_indicies[-1] if len(sum_indicies)>0 else cand_indicies+1
	## Importance (avg) by frequency
	X[...,2] = _m_freq(freq_map,source,cand_indicies, word2idx)
	## Similarity(cand, [:3] prev summary)
	X[...,3:6] = _similarity(sum_sent, cand_sent)
	## #Noun/verb overlap with summary and Pr(w)
	X[...,6:] = _overlap(source, sum_indicies, cand_indicies,model, emis_dense)
	# print("interac X",X)
	return X


def _overlap(source, sum_indicies, cand_indicies,  model, emis_dense):
	### given a list of sentences and summaries, count the overlapping between each candidate sent vs. all summary so far (0:2). also include logprob of those noun/verbs from the content model (2:4)
	### idx: the index of source article in given all documents
	### return (#candidates,2+2*(model._m-1))
	n_dim_= 2+2*(model._m-1)
	x = np.zeros((len(cand_indicies),n_dim_))
	
	verb_pattern = re.compile("VB")
	noun_pattern = re.compile("NN")
	
	noun_pos =  set([k for k,v in pos2idx.items() if noun_pattern.search(k)])
	verb_pos = set([k for k,v in pos2idx.items() if verb_pattern.search(k)])
	noun_idx = set([v for k,v in pos2idx.items() if noun_pattern.search(k)])
	verb_idx = set([v for k,v in pos2idx.items() if verb_pattern.search(k)]) # list of mappings for pos

	
	lemma_idx = set([lemma2idx.get(l) for s in sum_indicies for l in source[s] if lemma2idx.get(l)]) if len(sum_indicies)>0 else set([])# idx of all lemmas in summary
	# print("lemma_idx",lemma_idx)
	summary_noun = [l for s in sum_indicies for l in source[s] if lemma2pos.get(l) in noun_pos]
	summary_verb = [l for s in sum_indicies for l in source[s] if lemma2pos.get(l) in verb_pos]

	for i,cand_idx in enumerate(cand_indicies):
		# print("\nCandidate index:",cand_idx)
		words_idx = [lemma2idx.get(s) for s in source[cand_idx] if lemma2idx.get(s)] ##  one candidate sentence, by index of word

		common_word_idx = set(words_idx).intersection(lemma_idx) ##idx of all common words in summary and candidate
		common_pos = [lemma2pos[idx2lemma[l]] for l in common_word_idx] # a list of POS for words in both summary and candidate
		counter_pos = Counter(common_pos)
		# print("common_pos",common_pos,"counter_pos",counter_pos)
		x_ = np.zeros(n_dim_)

		common_lemma = [idx2lemma[s] for s in common_word_idx]
		# print("common_lemma",common_lemma)
		cand_noun = [l for l in source[cand_idx] if lemma2pos.get(l) in noun_pos]
		cand_verb = [l for l in source[cand_idx] if lemma2pos.get(l) in verb_pos]
		# print("cand_noun",cand_noun)

		n_noun = sum([counter_pos[n] for n in noun_pos])
		n_verb = sum([counter_pos[v] for v in verb_pos])
		
		## [i] = len(Overlap)/sqrt(len(candidate_verb)*len(summary_verbs))
		
		common_verb = [l for l in common_lemma if lemma2pos[l] in verb_pos]
		if common_verb:
			x_[0]= float(n_verb)/math.sqrt(len(summary_verb)*len(cand_verb))
			x_[2:1+model._m] = _emis_uni(common_verb, model, emis_dense)  
	
		
		common_noun = [l for l in common_lemma if lemma2pos[l] in noun_pos]
		if common_noun:
			x_[1] = float(n_noun)/math.sqrt(len(summary_noun)*len(cand_noun))
			x_[1-model._m:] = _emis_uni(common_noun, model, emis_dense) 

		x[i,...] = x_
	return x


def _emis_uni(words, model, emis_dense):
	## return emission logprob for words by model from each topic: (model._m, )
	x = np.zeros(model._m-1)
	word2idx = model._map
	for w in words:
		if w in STOPWORDS:
			continue
		# numer = np.array([np.sum(e.tocsr().toarray()[word2idx.get(w,2)]) for e in model._emis]) ## UNK = 2
		numer = np.array([np.sum(e[word2idx.get(w,2)]) for e in emis_dense])
		denom = np.sum(numer)
		prob_all = np.log(numer+epsilon)-np.log(denom+epsilon)
		x += prob_all
	return x


# def _x_EOD(source, summary,n_dim, len_source, last_pos):
# 	## feature vector for <end of summary>. Source and summary are (1,dim(feat)) vectors.
#	## Doesn't consider the prev sentence (last sentence in article)
# 	x = np.zeros(n_dim)
# 	x[0:len(source)] = source
# 	x[len(source):len(source)+len(summary)] = summary
# 	x[len(source)+len(summary)] = len_source+1 # position
# 	x[len(source)+len(summary)+1] = 6 # bucket of position

# 	pos_emb = FEAT2COL["cand_se"]
# 	x[pos_emb[0]:pos_emb[1]] = EOS_embedding
# 	x[pos_emb[1]+2] = len_source - last_pos
# 	return x

###############
# Calculation #
###############

def forward(x,W, rm ={}):
	## forward pass. W is a list of tuple, W[i][0] is weight, W[i][1] is bias. fn = [tanh, relu, sigmoid]
	fn = ["relu","tanh",'relu','relu'] # last one is always sigmoid
	h = x
	for i,f in enumerate(fn):
		z = h.dot(W[i][0])+W[i][1]
		if f == "relu":
			z = z*(z>0)
		else:
			z = np.tanh(z)
		h = z
	sigmoid = lambda x: 1 / (1 + np.exp(-x))
	out = sigmoid(z.dot(W[-1][0])+W[-1][1])
	return out.reshape((-1,1))


####################
# Importance score #
####################
#generate importance score for each sentence on the fly
def gen_XM(source,idx,context_sz =4):
	tmpf = "/home/ml/jliu164/code/data/importance_input/generation_"+str(idx)+".txt"
	try:
		f1 = open(tmpf,"r")
	except IOError:
		print("file for sentence importance score not generated")
		vocab_path = utils_path+"word2idx.json"
		with open(vocab_path) as f:
			vocab = json.load(f)
		print("In total %s word to index mapping"%(len(vocab)))
		tmpf = data_path+"importance_input/generation_"+str(idx)+".txt"
		_gen_X4M(source,vocab,context_sz = context_sz,filename = tmpf)

def _M_sent(idx, context_sz =4):
	genf = cur_path+"pred/Importance/"+str(context_sz)+"_generation_"+str(idx)+".npy"
	try:
		sent_M =np.load(genf)
	except IOError:
		print("sentence importance score not generated")
		tmpf = data_path+"importance_input/generation_"+str(idx)+".txt"
		
		f = open(tmpf,"r")
		sent_M = M_predict(f, cur_path+"pred/Importance/"+str(context_sz)+"_generation_"+str(idx)+".npy")
	return sent_M

# return the importance score for each sentence in source article
def importance(idx, source,context_sz=4):
	m = _M_sent(idx, context_sz = context_sz)
	# print("m.shape",m.shape)
	X = np.zeros((len(source),2))
	count = 0
	idx_sent = 0
	for sent in source:
		x_ = 0
		max_x = 0
		for w in sent[1:-1]:
			if w in SKIP_SET:
				continue
			else:
				x_ += m[count]
				max_x = max(max_x,m[count])
				count+=1
		x_ /= len(sent)
		X[idx_sent] = x_
		idx_sent += 1
	# print("M.shape",X.shape)
	return X



def article_embedding(idx, source,We = None):
	if not We:
		with open(utils_path+"we_file.json") as f:
			We = json.load(f)
	
	unk_vec = np.array(We["UNK"]).astype(float)
	x_se = np.zeros((len(source),DIM_EMB)) #(#sent,300) for embedding
	for i in range(len(source)):
		sent_vec = np.zeros(DIM_EMB)
		count = 0
		for w in source[i]:
			if w in set([START_SENT, END_SENT,SOD]):
				continue
			v = We.get(w,unk_vec)
			sent_vec += np.array(v).astype(float)
			count += 1
		sent_vec /= count
		x_se[i] = sent_vec
	
	np.save(data_path+"generation_input/embeddings_"+str(idx),x_se)
	print("x_se.shape",x_se.shape)



def load_vectors(idx, source, model, n_pca):
	"""
	load all useful variables and return.
	"""
	## set dimensions
	dim_src = DIM_EMB + model._m +1
	dim_sum = DIM_EMB + model._m + 4
	dim_cand = DIM_EMB + model._m + 4
	dim_interac = 8+2*(model._m-1)
	n_dim = dim_src + dim_interac+dim_cand+dim_sum + 1046*2
	print("Total dimension",n_dim)
	## pca decompose
	pca = pca_we() if n_pca !=0 else None
	## build frequency map
	doc_set = [set(a[1:-1]) for a in source]
	doc_words = reduce((lambda a,b: a.union(b)),doc_set)
	print("|V| in source: %s, length = %s" %(len(doc_words),len(source)))
	word2idx = dict(zip(list(doc_words),range(len(doc_words)))) # words in source to index mapping
	freq_map = np.zeros(len(doc_words))
	for sent in source:
		for w in sent[1:-1]:
			freq_map[word2idx[w]] += 1
	freq_map = freq_map/np.sum(freq_map) # normalize to probability
	## counts for each state
	state_prob = np.load(extract_dir+topic+".npy")
	## dense array for emission probability of the model
	emis_dense = [e.tocsr().toarray() for e in model._emis]
	
	## [source] = [embedding] + [cluster]
	X_src = np.zeros(dim_src)
	try:
		x_se = np.load(data_path+"generation_input/embeddings_"+str(idx)+".npy")
	except IOError:
		print("Embeddings for this article not generated!!!")

	x_dist = np.zeros(model._m) #(10,) for distribution
	_,flat = model.viterbi(source)
	flat_count = dict(Counter(flat))
	for c_id, c in flat_count.items():
		x_dist[c_id] = c
	X_src[:model._m] = x_dist
	X_src[model._m:-1] = np.mean(x_se,axis=0)
	X_src[-1] = len(source)

	## [cand] = [pos] + [emis] + [importance] + [embeddings]
	X_cand = np.zeros((len(source),dim_cand))
	X_cand[...,0] = np.arange(len(source)) #(#sent)
	X_cand[...,1] = np.array(flat)
	X_cand[...,2:2+model._m] =  model.sent_logprob(source).T #(#sent,10)
	M = importance(idx,source)
	X_cand[...,2+model._m: 4+model._m] = M
	X_cand[...,-DIM_EMB:] = x_se

	## Features from previous sentence.(#sent+1, 1046) [i,:] = feat(the prev sentence of sentence i+1)
	X_prev = build_x_prev(source, model)

	vectors = X_src, X_cand, x_se, X_prev
	utils = state_prob, emis_dense, flat,freq_map,word2idx, pca
	dims = dim_src,dim_sum,n_dim
	return vectors, utils, dims


###############
# Evaluation  #
###############

def eval_length(generated_idxs):
	"""use MSE to compare generated summary length to oracle. Visualize using scatter plot"""
	base_len = 3
	summary,_ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	## compare ability of predicting EOS
	generated_length = np.array([len(s) for s in generated_idxs])
	true_length = np.array([len(s) for s in summary])
	mse = np.mean(np.square(generated_length - true_length))
	mse_baseline = np.mean(np.square(np.full(len(summary),base_len) - true_length))
	print("Length mse generated",mse,"mse_baseline",mse_baseline)

	sorted_idx = np.argsort(true_length)
	tau, p_value = stats.kendalltau(true_length[sorted_idx], generated_length[sorted_idx]) # version that deals with ties
	print("Kendall's tau value", tau)
	
	## visualize length distribution using scatter plot
	x = np.arange(len(true_length))
	plt.scatter(x,true_length[sorted_idx],color='red', label="Oracle", s=1)
	plt.scatter(x,generated_length[sorted_idx],color='blue',label="System",s=1)
	plt.xlabel('# article')
	plt.ylabel('Summary Length')
	plt.title('Oracle vs system length')
	plt.legend(loc='upper right')
	plt.show()
		

def eval_rouge_score(indicies):
	_,doc = pickle.load(open(input_path+"contents/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	_,summary = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	generated = []
	reference = []
	for i,d in enumerate(doc):
		if len(indicies[i])==2:
			generated.append([d[s] for s in indicies[i]])
			reference.append(summary[i])
	scores = rouge_score(generated,reference)
	print("Rouge score",scores)

###############
#   Test      #
###############
def importance_generate():
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	## Importance
	for i in range(len(doc)):
		gen_XM(doc[i],i)
		_M_sent(i)
	

def test_beamsearch(idx=2):
	doc,origin_doc = pickle.load(open(input_path+"contents/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	model = pickle.load(open(_model_path+topic+".pkl","rb"),encoding="latin-1",errors="ignore")
	model_path = cur_path+"models/ffnn_newfeats/Normalize_(1401, 255, 564, 209).h5py"
	file = h5py.File(model_path,"r")
	weights = []
	for i in range(0,len(file.keys()),2):
		weights.append((file['weight'+str(i)][:], file['weight'+str(i+1)][:]))
	print(">> Model loaded. %s hidden layers: %s"%(len(weights)-1,[k[1].shape[0] for k in weights[:-1]]))
	p_selected = data_path+"model_input/FFNN/selected_sentences2.json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	print("=> Oracle mapping", selected_tot[idx])
	# generated_idx = search_summary_oracle(idx, doc[idx],model,weights, oracle_len =len(selected_tot[idx]))
	generated_idx = search_summary(idx, doc[idx],model,weights)
	print("Output: ",generated_idx)

def test_generate():
	# idx = 15 # index of the file in test data set
	# rm = {'interac_emis', 'cand_prob', 'sum_cluster', 'src_cluster'}# remove features while doing forward pass
	rm = {}
	n_pca = 0

	doc,origin_doc = pickle.load(open(input_path+"contents/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	_, origin_summary = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	model = pickle.load(open(_model_path+topic+".pkl","rb"),encoding="latin-1",errors="ignore")
	
	## load model weights
	model_path = cur_path+"models/ffnn_newfeats/Normalize_(1401, 255, 564, 209).h5py"
	# model_path = cur_path+"models/ffnn_newfeats/NormalizeExp_(868, 1228, 666, 1629).h5py"
	file = h5py.File(model_path,"r")
	weights = []
	for i in range(0,len(file.keys()),2):
		weights.append((file['weight'+str(i)][:], file['weight'+str(i+1)][:]))
	print(">> Model loaded. %s hidden layers: %s"%(len(weights)-1,[k[1].shape[0] for k in weights[:-1]]))
	
	## actual summary
	p_selected = data_path+"model_input/FFNN/selected_sentences2.json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)

	generated = []
	generated_idxs = []
	for i,d in enumerate(doc):
		# d = [d_ for d_ in d if len(d_)>2]
		print("\n\nGenerating article No. %s, original doc length %s"%(i,len(origin_doc[i])))
		# generated_idx = search_summary_oracle(i, d,model,weights,rm=rm, n_pca = n_pca, oracle_len= len(origin_summary[i]))
		generated_idx = search_summary(i, d,model,weights,rm=rm, n_pca = n_pca)
		# generated_idx = generate_summary(i, d,model,weights,rm=rm, n_pca = n_pca, force_length=np.inf)
		print("==> generated",generated_idx)
		assert len(generated_idx)>0
		generated_idxs.append(generated_idx)
		generated.append([origin_doc[i][s] for s in generated_idx])
		print("==> Gold standard", selected_tot[i])

	pickle.dump(generated_idxs,open("pred/generated_newFeat_beamSearch.pkl","wb"))
	scores = rouge_score(generated, origin_summary)
	print(scores)


if __name__ == '__main__':
	# importance_generate()

	test_generate()
	# test_beamsearch(idx=7) #2,6
	
	# generated_idxs = pickle.load(open("pred/generated_idxs_beamSearch.pkl","rb"))
	# eval_rouge_score(generated_idxs)
	# eval_length(generated_idxs)