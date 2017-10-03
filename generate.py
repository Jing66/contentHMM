import numpy as np
import pickle
import sys
import os
import json
from collections import Counter
import h5py
import re
import math

from rdn_sample_cand import _x_EOD
from utils import _gen_file_for_M as _gen_X4M
from utils import pca_we
from all_feat import feat_select

if sys.version_info[0] >=3:
	from pred_M import M_predict
else:
	from interac_feat import _similarity, N_PREV, _m_freq
	from lexical_feat import _emis_uni
	sys.path.append(os.path.abspath('..'))
	from content_hmm import ContentTagger
	from corenlpy import AnnotatedText as A

topic ='War Crimes and Criminals'
input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
data_path = "/home/ml/jliu164/code/data/"
cur_path = "/home/ml/jliu164/code/Summarization/"
utils_path = "/home/ml/jliu164/code/data/utils/"
save_path = "/home/ml/jliu164/code/filter_results/topic2files(content).pkl"
corpus_path =  "/home/rldata/jingyun/nyt_corpus/content_annotated/"
fail_path = '/home/ml/jliu164/code/contentHMM_input/fail/'

START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"
SKIP_SET = {SOD,END_SENT,START_SENT}

N_CAND = 10
DIM_EMB = 300

	
idx = 3 # index of the file in test data set
# rm = {"cand_se","src_se","sum_se"} # remove features while doing forward pass
rm = {}

def generate_summary(annotated_source, source, model, weights, rm ={},n_pca = 0):
	""" 
	Given source article and content model, no gold standard summary, generate a summary.
	annotated_source: from corenlpy. source: list of sentences, each list of words. model: content HMM model. weights: list of numpy arrays for doing prediction
	"""
	## set dimensions
	dim_src = DIM_EMB + model._m
	dim_sum = DIM_EMB + model._m + 4
	dim_cand = DIM_EMB + model._m + 4
	dim_interac = 8+2*(model._m-1)
	n_dim = dim_src + dim_interac+dim_cand+dim_sum

	## pca decompose
	pca = pca_we() if n_pca else None

	## build frequency map
	doc_set = [set(a[1:-1]) for a in source]
	doc_words = reduce((lambda a,b: a.union(b)),doc_set)
	print("|V| in source: %s" %(len(doc_words)))
	word2idx = dict(zip(list(doc_words),range(len(doc_words)))) # words in source to index mapping
	freq_map = np.zeros(len(doc_words))
	for sent in source:
		for w in sent[1:-1]:
			freq_map[word2idx[w]] += 1
	freq_map = freq_map/np.sum(freq_map) # normalize to probability

	## [source] = [embedding] + [cluster]
	X_src = np.zeros(dim_src)
	try:
		x_se = np.load(data_path+"generation_input/embeddings_"+str(idx)+".npy")
	except IOError:
		print("generating embeddings for this article...")
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

	x_dist = np.zeros(model._m) #(10,) for distribution
	_,flat = model.viterbi([source])
	flat_count = dict(Counter(flat))
	for c_id, c in flat_count.items():
		x_dist[c_id] = c
	X_src[:model._m] = x_dist
	X_src[model._m:] = np.mean(x_se,axis=0)
	print("X_src generated",X_src.shape)

	## [cand] = [pos] + [emis] + [importance] + [embeddings]
	X_cand = np.zeros((len(source),dim_cand))
	X_cand[...,0] = np.arange(len(source)) #(#sent)
	X_cand[...,1] = np.array(flat)
	X_cand[...,2:2+model._m] =  model.sent_logprob(source).T #(#sent,10)
	M = importance(source)
	X_cand[...,2+model._m: 4+model._m] = M
	X_cand[...,-DIM_EMB:] = x_se
	print("X_cand generated",X_cand.shape)

	## [summary so far]
	eos = False
	chosen = np.array([-1]).astype(int) # indicies of selected summary
	count = 0
	while (not eos) and chosen[-1] < len(source)-1:
		print("\n>>generating %s summary sentence:"%(count))
		## feature vector for summary so far. same for all candidates thus n_row = 1
		X_sum = np.zeros(dim_sum) 
		X_sum[:14] = sum_so_far(source,chosen,model)

		X_sum[-DIM_EMB:] = np.mean(x_se[chosen[1:]],axis=0) if len(chosen)>1 else np.zeros(DIM_EMB) # average all chosen summary embeddings. ALT: last embedding
		X_fixed = np.hstack((X_src, X_sum)) # fixed part of the vector

		## select candidates
		n_i = np.arange(chosen[-1]+1, min(len(source),10+chosen[-1]+1)) # boundary for #candidates
		print("candidate n_i",n_i)
		## feature vector for interaction
		X_interac = interac(annotated_source,source, flat, chosen, n_i, model, freq_map,word2idx)
		eos_vec = _x_EOD(X_src, X_sum, n_dim, len(source), chosen[-1]) ## feature vector for eos. appended at the end
		x_cand = np.hstack((X_cand[n_i], X_interac))
		tmp = np.broadcast_to(X_fixed, (len(n_i), X_fixed.shape[0]))
		x_cand = np.hstack((tmp, x_cand))

		X = np.vstack((x_cand, eos_vec))
		
		## Forward pass: fn(X(11 x n_dim) * W(n_dim x 1)) => y_hat(11 x 1)
		y_hat = forward(X, weights, rm=rm, n_pca=n_pca,pca=pca)
		print("yp.shape",y_hat.shape)
		target = np.argmax(y_hat) + chosen[-1]+1
		print("chosen target",target)

		if target == len(X)-1 and len(chosen) > 2:
			eos = True
			print("<EOS>")
		else:
			y_hat = y_hat[:-1]
			target = np.argmax(y_hat) + chosen[-1]+1
			chosen = np.append(chosen,target)
		print("chosen summary indicies",chosen)
		count +=1
		
	return chosen[1:]



def sum_so_far(source, indicies,model,n_bin=5):
	""" compute feature vec for summary so far, excluding embeddings cuz they can be added back later.
		indicies: array of index for which sentences are chosen as summary
		return (1 x dim_sum) vector
	"""
	X = np.zeros(model._m+4)
	## we don't have real summary distribution
	X[:model._m] = 0

	X[-4] = sum([len(source[i]) for i in indicies[1:]]) # n_word overlap == length of total summary sentences
	X[-3] = indicies[-1] # position of last chosen sentence
	#bin of previous position.
	if indicies[-1] ==-1:
		X[-2] = 0
	else:
		bins = np.arange(-1,len(source)+1,float(len(source))/n_bin)
		X[-2] = np.digitize(indicies[-1],bins)
	X[-1] = len(indicies)-1 # number of summary so far
	# print(X)
	return X



def interac(annotated_source, source, flat, sum_indicies,cand_indicies,model,freq_map, word2idx):
	""" compute feature vec for interaction
		return size (#candidate x dim_interac)
	"""
	flat = np.array(flat).astype(int)
	X = np.zeros((len(cand_indicies),8+2*(model._m-1))) #(??, 26)
	cand_sent = [source[i] for i in cand_indicies]
	sum_sent = [source[i] for i in sum_indicies[1:]]
	annotated_sum_sent = [annotated_source[i] for i in sum_indicies[1:]]
	annotated_cand_sent = [annotated_source[i] for i in cand_indicies]
	## Pr(S_cand|S_last summary)
	trans = model._trans
	prior = model._priors
	tmp = trans[flat[cand_indicies], flat[sum_indicies[-1]]] if sum_indicies[-1]!=-1 else prior[flat[cand_indicies]]
	X[...,0] = tmp
	## pos(cand)-pos(last summary) 
	# X[...,1] = cand_indicies - sum_indicies[-1] if sum_indicies[-1]!=-1 else 0 
	X[...,1] = cand_indicies - sum_indicies[-1]
	## Importance (avg) by frequency
	X[...,2] = _m_freq(freq_map,source,cand_indicies, word2idx)
	## Similarity(cand, [:3] prev summary)
	X[...,3:6] = _similarity(sum_sent, cand_sent)
	## #Noun/verb overlap with summary and Pr(w)
	X[...,6:] = _overlap(annotated_cand_sent, annotated_sum_sent,model)
	# print("interac X",X)
	return X


def _overlap(cands,sum_so_far,  model):
	### given a list of sentences and summaries, count the overlapping between each candidate sent vs. all summary so far (0:2). also include logprob of those noun/verbs from the content model (2:4)
	### return (#candidates,2+2*(model._m-1))
	n_dim_= 2+2*(model._m-1)
	x = np.zeros((len(cands),n_dim_))
	if not sum_so_far or not cands:
		return x
	verb_pattern = re.compile("VB")
	noun_pattern = re.compile("NN")
	summary_tokens = [s['tokens'] for s in sum_so_far]
	sum_verbs = set([t['lemma'] for tokens in summary_tokens for t in tokens if verb_pattern.search(t['pos'])])
	sum_nouns = set([t['lemma'] for tokens in summary_tokens for t in tokens if noun_pattern.search(t['pos'])])
	for i,cand in enumerate(cands):
		cand_token = cand['tokens']
		cand_verb = set([t['lemma'] for t in cand_token if verb_pattern.search(t['pos'])])
		cand_noun = set([t['lemma'] for t in cand_token if noun_pattern.search(t['pos'])])
		
		n_verb = sum_verbs.intersection(cand_verb)
		n_noun = sum_nouns.intersection(cand_noun)
		x_ = np.zeros(n_dim_)
		## [i] = len(Overlap)/sqrt(len(candidate)*len(summary_verbs))
		if n_verb:
			x_[0]= float(len(n_verb))/math.sqrt(len(cand_token))
			x_[2:1+model._m] = _emis_uni(n_verb, model._map)
		if n_noun:
			x_[1] = float(len(n_noun))/math.sqrt(len(cand_token))
			x_[1-model._m:] = _emis_uni(n_noun, model._map) 
		x[i,...] = x_
	# print("__overlap x",x)
	return x

###############
# Calculation #
###############
def forward(x,W, rm ={},n_pca = None, pca = None):
	## forward pass. W is a list of tuple, W[i][0] is weight, W[i][1] is bias. fn = [tanh, relu, sigmoid]
	x = feat_select(x, None, rm, n_pca = n_pca, pca = None)
	h1 = np.tanh(x.dot(W[0][0]+W[0][1]))
	# print("h1.shape",h1.shape)
	h2 = np.maximum(h1.dot(W[1][0])+W[1][1],0) #relu
	# print("h2.shape",h2.shape, "W2.shape",W[2][0].shape)
	sigmoid = lambda x: 1 / (1 + np.exp(-x))
	out = sigmoid(h2.dot(W[2][0])+W[2][1])
	return out.reshape((-1,1))


####################
# Importance score #
####################
#generate importance score for each sentence on the fly
def gen_XM(source,context_sz =4):
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

# use python3
def _M_sent(context_sz =4):
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
def importance(source,context_sz=4):
	m = _M_sent(context_sz = context_sz)
	print("m.shape",m.shape)
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
	print("M.shape",X.shape)
	return X


###############
#   Test      #
###############
def test():
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+"2.pkl","rb"))
	print("source length",len(doc[idx]))
	model = pickle.load(open(_model_path+topic+".pkl","rb"))

	## Importance
	# gen_XM(doc[idx]) #python2
	# _M_sent() #python3
	# importance(doc[idx]) #python2

	## load annotated article
	files_ = pickle.load(open(save_path))[topic]
	with open(fail_path+topic+"_Failed.txt") as f:
		paths = f.readlines()
		failed = set([p.split("/")[-1].split(".")[0] for p in paths])
	word2idx = model._map
	files = [fs for fs in files_[:-1] if fs.split('/')[-1].split(".")[0] not in failed]
	files = files[int(-np.around(0.1*len(files))):]
	xml = open(corpus_path+files[idx]).read()
	annotated_source = A(xml).sentences

	# s = open("/home/rldata/jingyun/nyt_corpus/content/"+re.sub("_annotated","",files[idx]).strip(".xml")).read()
	# print(s)
	# exit(0)

	## load model weights
	model_path = cur_path+"models/ffnn_2step_2.0/bi_classif_War_AllFeat(1493, 1109)_nonbin.h5py"
	file = h5py.File(model_path,"r")
	weights = []
	for i in range(0,len(file.keys()),2):
		weights.append((file['weight'+str(i)][:], file['weight'+str(i+1)][:]))
	print(">> Model loaded. %s hidden layers: %s"%(len(weights)-1,[k[1].shape[0] for k in weights[:-1]]))

	print(generate_summary(annotated_source, doc[idx], model,weights, rm=rm))
	## actual summary
	p_selected = data_path+"model_input/FFNN/selected_sentences2.json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)
	print(selected_tot[idx])

if __name__ == '__main__':
	test()

