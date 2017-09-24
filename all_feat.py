import numpy as np
import json
import pickle
import math
import sys
import os
from collections import Counter

from scipy import io
from functools import reduce

from utils import MyPCA, pca_we
sys.path.append(os.path.abspath('..'))
from content_hmm import *


EOD = "*EOD*" # predict the end
START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"


FEAT2COL = {"src_cluster":(0,10),"src_se":(10,310),
	"sum_cluster":(310,320),"sum_overlap":(320,321),"sum_pos":(321,322),"sum_posbin":(322,323),"sum_num":(323,324),"sum_se":(324,624),
	"cand_pos":(624,625),"cand_cluid":(625,626),"cand_prob":(626,636),"cand_M":(636,638),"cand_se":(638,938),
	"interac_trans":(938,939),"interac_pos":(939,940),"interac_M":(940,941),"interac_sim_nprev":(941,944),"interac_w_overlap":(944,946),"interac_emis":(946,964)}
CATE_FEAT = {"sum_overlap","sum_pos","sum_num","cand_pos","cand_cluid","interac_pos","interac_overlap"} # cannot normalize
EMB = {"src_se","cand_se","sum_se"}

data_path = "/home/ml/jliu164/code/data/"
src_path = "/home/ml/jliu164/code/Summarization/"
input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
# input_path = data_path+"/seq_input/"
NUM_CAND = 10
SKIP_SET = set([START_SENT, END_SENT,SOD, EOD])
np.random.seed(7)

def _length_indicator(ds = 1, topic ='War Crimes and Criminals'):
	# save an array to indicate length
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	ind = [len(d) for d in doc]
	return np.array(ind)


def _select_sentence(ds=1, topic ='War Crimes and Criminals', savename = data_path+"model_input/FFNN/selected_sentences"):
	# save a list of list. for each selected[i][k], kth sentence in article[i] is selected as summary
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	summary, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	assert len(doc) == len(summary), "More documents than summary!" if len(doc) > len(summary) else "Less documents than summary!"
	
	print("Getting selected sentences...")
	selected = []
	idx = 0
	for d,s in zip(doc,summary):
		uni_doc = [set(i) for i in d] 
		uni_sum = [set(i) for i in s]
		y = np.zeros(len(s)) # s[i] is most close to d[j]
		for i in range(len(s)):
			max_dist = np.inf
			
			for j in range(len(d)):
				# Unigram cosine similarity
				numer = len(uni_doc[j].intersection(uni_sum[i]))-2
				denom = math.sqrt((len(uni_doc[j])-2)*(len(uni_sum[i])-2))
				dist = 1- float(numer)/denom # omit start/end of sentence

				if dist < max_dist:
					y[i] = j
					max_dist = dist
				
		# y = sorted(list(set(y))) # every sentence in summary is similar to a different sentence in doc
		y = sorted(y)
		
		assert len(y)!=0
		assert np.max(y)<=len(d)
		selected.append(y)
	assert len(selected) == len(summary)
	print("selected summary sentences for %s articles"%(len(selected)))
	y = selected[0]
	
	if savename:
		import json
		with open(savename+str(ds)+'.json','w') as f:
			json.dump(selected, f)

def _seq2vec(content, ds =1, topic = 'War Crimes and Criminals', savedir = data_path+"model_input/FFNN/"):
	## generate all sequence vectors as input. We(sentence) = avg(We(words))
	# content: if true, process for content words. otherwise process for summaries
	with open("/home/ml/jliu164/code/data/we_file.json") as f:
		We = json.load(f)
		unk_vec = We["UNK"]	
	_We_dim = len(unk_vec)
	_path = input_path+("contents/" if content else "summaries/")+topic+"/"+topic+str(ds)+".pkl"
	doc,_ = pickle.load(open(_path,"rb"))
		
	sentences = [i for val in doc for i in val]
	print("Will process %s sentences. ds=%s, for content = %s"%(len(sentences),ds,content))
		
	we_sents = np.zeros((len(sentences),_We_dim))
	c=0
	for sent in sentences:
		pos = 0
		vec_sent = np.zeros(_We_dim)
		for w in sent:
			if w in set([START_SENT, END_SENT,SOD]):
				continue
			
			vec = np.array(We.get(w,unk_vec),dtype=np.float32)
			vec_sent += vec
			pos += 1
		
		# assert np.all(vec_sent) != 0.0
		if np.all(vec_sent) == 0:
			print(sent)
		vec_sent = vec_sent/pos
		we_sents[c,...] = vec_sent
		c+=1
	print("sequence vector shape",we_sents.shape)
	_save = savedir+("content" if content else "summary")+"X_SeqVec"+str(ds)
	np.save(_save,we_sents)
	return we_sents
	
def _get_importance(topic, ds,context_sz):
	docs,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	m = np.load("pred/pred_M"+str(context_sz)+("_test" if ds==2 else "_train")+"_model.npy")
	print("m.shape",m.shape)
	X = np.zeros((sum([len(d) for d in docs]),2))
	idx_doc = 0
	count = 0
	for doc in docs:
		idx_sent = 0
		for sent in doc:
			x_ = 0
			max_x = 0
			for w in sent:
				if w in SKIP_SET:
					continue
				else:
					x_ += m[count]
					max_x = max(max_x,m[count])
					count+=1
			x_ /= len(sent)
			X[idx_doc+idx_sent] = x_
			idx_sent += 1
		idx_doc += len(doc)
	print("M.shape",X.shape)
	return X

def _feat_source(ds = 1, topic = 'War Crimes and Criminals'):
	## feature of source articles. count clusters distributions for each article. 
	## return X_source: (#articles, #clusters)
	sources, _ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	model = pickle.load(open(_model_path+topic+".pkl","rb"))
	## load sequence embeddings
	_seqdir = data_path+"model_input/FFNN/"+"contentX_SeqVec"+str(ds)+".npy"
	x_seq = np.load(_seqdir)

	n_cluster = model._m
	n_dim = x_seq.shape[1]
	X_source = np.zeros((len(sources), n_cluster+n_dim))
	_,flat = model.viterbi(sources)
	assert(len(flat) == sum([len(c) for c in sources]))
	print("model._m = "+str(model._m))
	idx = 0
	for i in range(len(sources)):
		doc = sources[i]
		x_seq_= x_seq[idx:idx+len(doc)]
		X_seq = np.mean(x_seq,axis=0)

		flat_ = flat[idx: idx+len(doc)]
		flat_count = dict(Counter(flat_))
		for c_id, c in flat_count.items():
			X_source[i][c_id] = c
		X_source[i][-n_dim:]=X_seq
		idx += len(doc)
	
	return X_source


def _feat_sum_sofar(ds = 1, topic = 'War Crimes and Criminals',savename = data_path+"model_input/FFNN/X_summary"):
	# return X_sum: (#summary sentences, #clusters + 4)
	summaries, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	sources, _ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	model = pickle.load(open(_model_path+topic+".pkl","rb"))
	# sources = sources[:5]
	# summaries = summaries[:5]

	n_feat = 4 # num_features other than clusters or embeddings
	n_bin = 5 # number of buckets
	X_ = np.zeros((sum([len(s)+1 for s in summaries]), model._m +n_feat)) #shape (#sentences, #clusters)
	
	## feat: cluster distribution feature
	_, flat = model.viterbi(summaries)
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
	
	## feat: word overlap feature: summary sentence vs. source article
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

	## feat: position of last chosen sentence. If no sentence chosen, index -1
	p_selected = data_path+"model_input/FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	# assert sum([len(s) for s in selected_tot]) == sum([len(s) for s in summaries])
	s_inserted = [[-1]+s for s in selected_tot]
	pos = [int(i) for val in s_inserted for i in val]
	X_[...,model._m+1] = np.array(pos)
	## ADD feat: bin of previous position. Total n_bin buckets. first one always 0, the rest index start at 1.
	x_bins = np.zeros(X_.shape[0])
	idx = 0
	for i,s in enumerate(sources):
		bins = np.arange(-1,len(s)+1,float(len(s))/n_bin)
		x_bin = np.digitize(np.array([-1]+selected_tot[i]),bins)
		x_bin[0] = 0
		x_bins[idx:idx+1+len(summaries[i])]=x_bin
	X_[...,model._m+2] = np.array(x_bins)

	## feat: number of Summaries chosen so far
	x = np.zeros(X_.shape[0])
	idx = 0
	for s in summaries:
		x[idx:idx+len(s)+1] = np.arange(len(s)+1)
		idx += len(s)+1
	X_[...,-1] = x

	## load sequence embeddings
	_seqdir = data_path+"model_input/FFNN/"+"summaryX_SeqVec"+str(ds)+".npy"
	x_seq = np.load(_seqdir)
	X_seq = np.zeros((X_.shape[0],x_seq.shape[1]))
	idx_seq = 0
	idx_x = 0
	for s in summaries:
		X_seq[idx_x+1:idx_x+len(s)+1,...] = x_seq[idx_seq:idx_seq+len(s),...]
		idx_x += len(s)+1
		idx_seq += len(s)
	X_ = np.hstack((X_,X_seq))
	if savename:
		np.save(savename+str(ds), X_)
	print("X_summary.shape",X_.shape)
	return X_


def _feat_cand_noninterac(ds = 1, topic = 'War Crimes and Criminals',savename = data_path+"model_input/FFNN/X_cand"):
	# features that doesn't need interaction with source and summary so far. [M]+[emission/cluster_id from HMM] + [pos] + [tf-idf]
	# return (#n_sentences in source, 14)
	sources, _ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	flat_sent =  [i for val in sources for i in val]
	_seqdir = data_path+"model_input/FFNN/"+"contentX_SeqVec"+str(ds)+".npy"
	x_seq = np.load(_seqdir)

	n_feat = 14
	X_ = np.zeros((len(flat_sent),n_feat+x_seq.shape[1]))
	print("_feat_cand_noninterac: X_.shape",X_.shape)
	print("x_seq.shape",x_seq.shape)
	
	# position feature
	X_pos = np.zeros(X_.shape[0])
	idx = 0
	for doc in sources:
		X_pos[idx:idx+len(doc)] = range(len(doc))
		idx += len(doc)
	X_[...,0] = X_pos
	
	## HMM feature (dim= model._m + 1)
	model = pickle.load(open(_model_path+topic+".pkl","rb"))
	_, flat = model.viterbi(sources)
	X_hmm = np.zeros((X_.shape[0],model._m+1))
	X_hmm[...,0] = np.array(flat) # cluster id
	emis_prob = model.sent_logprob(flat_sent)
	X_hmm[...,1:11] = emis_prob.T # sentence emission log probability
	X_[...,1:12] = X_hmm

	## importance score feature
	context_sz = 4
	M = _get_importance(topic, ds,context_sz)
	print("Importance score shape", M.shape)
	X_[...,12:14] = M

	## sequence embeddings feature at the end
	X_[...,n_feat:] = x_seq

	## tf-idf score feature
	#############################
	# TO IMPLEMENTE--can only do max/avg without POS/NER here
	#############################
	if savename:
		np.save(savename+str(ds), X_)
	return X_


def make_X(ds = 1, topic = 'War Crimes and Criminals', savedir = data_path+"model_input/FFNN/"):
	# interaction between [source]+[summary_so_far]+[1 candidate out of NUM_CAND].
	# ADD [trans HMM prob]+[POS/NER overlap]
	
	# load components
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
	# exit(0)
	p_selected = data_path+"model_input/FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	len_ind = _length_indicator(ds=ds,topic=topic)
	# print(len_ind)

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
		# print("\n>>n = %s. selected idx:%s. first fill in %s rows."%(len_ind[idx_source],selected_idx, n_i))
		tmp = np.hstack((source_vec,sum_vec[0]))[np.newaxis,:]
		tmp = np.broadcast_to(tmp, (n_i, tmp.shape[1])) # (6,22)
		x_prev = np.hstack((tmp,cand_vec[:n_i]))
		# loop inside all indicies
		for idx_ in range(len(selected_idx)-1):
			cur_idx = selected_idx[idx_]
			next_idx = selected_idx[idx_+1]
			n_i = min(NUM_CAND+cur_idx+1, len_ind[idx_source]) - cur_idx -1 if next_idx<NUM_CAND+cur_idx else (next_idx-cur_idx)
			
			_sum_vec = sum_vec[idx_+1] # ----BUG----
			_cand_vec = cand_vec[int(cur_idx+1):int(cur_idx+n_i+1)]
			# print("selected idx:%s. next locally fill in rows[%s-%s).n_i=%s"%(cur_idx, cur_idx+1,cur_idx+n_i+1,n_i))
			if n_i:
				tmp = np.hstack((source_vec, _sum_vec))[np.newaxis,:]
				tmp = np.broadcast_to(tmp, (_cand_vec.shape[0],tmp.shape[1])) #(5,22)
				tmp_ = np.hstack((tmp,_cand_vec))
				x_prev = np.vstack((x_prev, tmp_))
		# Last row
		n_i = min(NUM_CAND+selected_idx[-1]+1, len_ind[idx_source]) - selected_idx[-1] -1
		if n_i:
			tmp = np.hstack((source_vec,sum_vec[-1]))[np.newaxis,:]
			tmp = np.broadcast_to(tmp, (int(n_i), tmp.shape[1]))
			tmp_ = np.hstack((tmp,cand_vec[int(selected_idx[-1]+1):int(selected_idx[-1]+1+n_i)]))
			x_prev = np.vstack((x_prev, tmp_))

		X = np.vstack((X,x_prev))
	X = X[1:]
	print("X_concat shape",X.shape)
	#############################
	# interac_feat.py: feature transition prob
	# lexical_feat.py: feature POS overlap counts
	#############################
	## feat: [P(S_cand|S_last summary)] + [Sim(cand, last summary)]
	X_interac = np.load(savedir+"X_interac_nprev"+str(ds)+".npy")
	print("X_interac.shape",X_interac.shape)
	X = np.hstack((X,X_interac))
	## feat: [Verb overlap] + [Noun overlap] of candidate vs. summary so far
	X_lexical = np.load(savedir+"X_lexical"+str(ds)+".npy")
	print("X_lexical.shape",X_lexical.shape)
	X = np.hstack((X,X_lexical))
	
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
		# print("\n >>Article length = %s, selected = %s, initial fill %s rows" %(len_ind[i], selected,n_i))
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



def feat_select(X,Y, rm, n_pca = 100, pca = None):
	## select features
	## Removing features by me
	X_ = np.zeros((X.shape[0],1))
	
	idx = 0
	for feat in FEAT2COL:
		if feat in rm:
			continue
		start,stop = FEAT2COL[feat]
		x =X[...,start:stop]
		## PCA
		if pca and n_pca and feat in EMB:
			# print("n_pca",n_pca)
			x = pca.transform(n_pca, x)
			# print("x.shape",x.shape)
			
		X_ = np.hstack((X_,x))
		idx += (stop-start)
	

	X_ = X_[...,1:]
	print("X_.shape",X_.shape)
	return X_




#######################################
############ Execution ################
#######################################

def test():
	## test select_article_into_summary
	# savename = data_path+"model_input/FFNN/selected_sentences.json"
	# with open(savename,'r') as f:
	# 	l = json.load(f)
	# print(l[:5])

	## test make Y
	# _select_sentence(ds =1)

	# Y = make_Y(ds=2)
	# print(Y.shape)  #### ds=2: 4909. ds=1:33519

	# X_seq = _seq2vec(False,ds=2)
	# M = _get_importance("War Crimes and Criminals",1,4)

	# X_source = _feat_source(ds = 2)
	# print("X_source.shape",X_source.shape)
	X_sum = _feat_sum_sofar(ds = 1)
	
	# _feat_cand_noninterac(ds = 2)
	# X = make_X(ds=1)

	# X,Y = load_data(False, ds = 1)
	# from sklearn.feature_selection import mutual_info_classif
	# from sklearn.feature_selection import SelectKBest
	# X_new = SelectKBest(mutual_info_classif, k=500).fit_transform(X, Y.ravel())
	# pca = PCA(n_components = 35)
	# X_ = pca.fit_transform(X)
	# print(X.shape)
	# cov = pca.get_covariance()
	# print(cov.shape)
	pass
	


if __name__ == '__main__':
	test()
