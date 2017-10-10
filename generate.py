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
import matplotlib.pyplot as plt

from rdn_sample_cand import _x_EOD
from utils import _gen_file_for_M as _gen_X4M
from utils import pca_we
from all_feat import feat_select
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

pos2idx = json.load(open("../data/utils/pos2idx2.json"))
lemma2idx = json.load(open("../data/utils/lemma2idx2.json"))
idx2lemma = inverted_dict = dict([[v,k] for k,v in lemma2idx.items()])
lemma2pos = json.load(open("../data/utils/lemma2pos2.json"))

def generate_summary(idx, source, model, weights, rm ={},n_pca = 0):
	""" 
	Given source article and content model, no gold standard summary, generate a summary.
	source: list of sentences, each list of words. model: content HMM model. weights: list of numpy arrays for doing prediction
	"""
	## set dimensions
	dim_src = DIM_EMB + model._m
	dim_sum = DIM_EMB + model._m + 4
	dim_cand = DIM_EMB + model._m + 4
	dim_interac = 8+2*(model._m-1)
	n_dim = dim_src + dim_interac+dim_cand+dim_sum

	## pca decompose
	pca = pca_we() if n_pca !=0 else None
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
	X_src[model._m:] = np.mean(x_se,axis=0)
	print("X_src generated",X_src.shape)

	## [cand] = [pos] + [emis] + [importance] + [embeddings]
	X_cand = np.zeros((len(source),dim_cand))
	X_cand[...,0] = np.arange(len(source)) #(#sent)
	X_cand[...,1] = np.array(flat)
	X_cand[...,2:2+model._m] =  model.sent_logprob(source).T #(#sent,10)
	M = importance(idx,source)
	X_cand[...,2+model._m: 4+model._m] = M
	X_cand[...,-DIM_EMB:] = x_se
	print("X_cand generated",X_cand.shape)

	## [summary so far]
	eos = False
	chosen = np.array([-1]).astype(int) # indicies of selected summary
	count = 0
	while (not eos) and chosen[-1] < len(source)-1:
		# print("\n>>generating %s summary sentence:"%(count))
		## feature vector for summary so far. same for all candidates thus n_row = 1
		X_sum = np.zeros(dim_sum) 
		X_sum[:14] = sum_so_far(source,state_prob, chosen,model)

		X_sum[-DIM_EMB:] = np.mean(x_se[chosen[1:]],axis=0) if len(chosen)>1 else np.zeros(DIM_EMB) # average all chosen summary embeddings. ALT: last embedding
		X_fixed = np.hstack((X_src, X_sum)) # fixed part of the vector

		## select candidates
		n_i = np.arange(chosen[-1]+1, min(len(source),10+chosen[-1]+1)) # boundary for #candidates
		# print("candidate n_i",n_i)
		## feature vector for interaction
		X_interac = interac(source, flat, chosen, n_i, model, freq_map,word2idx, emis_dense)
		eos_vec = _x_EOD(X_src, X_sum, n_dim, len(source), chosen[-1]) ## feature vector for eos. appended at the end
		x_cand = np.hstack((X_cand[n_i], X_interac))
		tmp = np.broadcast_to(X_fixed, (len(n_i), X_fixed.shape[0]))
		x_cand = np.hstack((tmp, x_cand))

		X = np.vstack((x_cand, eos_vec))
		
		## Forward pass: fn(X(11 x n_dim) * W(n_dim x 1)) => y_hat(11 x 1)
		y_hat = forward(X, weights, rm=rm, n_pca=n_pca,pca=pca)
		# print("yp.shape",y_hat.shape)
		target = np.argmax(y_hat) + chosen[-1]+1
		# print("chosen target",target)

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



def sum_so_far(source,state_prob, indicies,model,n_bin=5):
	""" compute feature vec for summary so far, excluding embeddings cuz they can be added back later.
		indicies: array of index for which sentences are chosen as summary
		return (1 x dim_sum) vector
	"""
	X = np.zeros(model._m+4)
	## we don't have real summary distribution
	X[:model._m] = state_prob

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



def interac(source, flat, sum_indicies,cand_indicies,model,freq_map, word2idx, emis_dense):
	""" compute feature vec for interaction
		return size (#candidate x dim_interac)
	"""
	flat = np.array(flat).astype(int)
	X = np.zeros((len(cand_indicies),8+2*(model._m-1))) #(??, 26)
	cand_sent = [source[i] for i in cand_indicies]
	sum_sent = [source[i] for i in sum_indicies[1:]]
	
	## Pr(S_cand|S_last summary)
	trans = model._trans
	prior = model._priors
	tmp = trans[flat[sum_indicies[-1]],flat[cand_indicies]] if sum_indicies[-1]!=-1 else prior[flat[cand_indicies]]
	X[...,0] = tmp
	## pos(cand)-pos(last summary) 
	# X[...,1] = cand_indicies - sum_indicies[-1] if sum_indicies[-1]!=-1 else 0 
	X[...,1] = cand_indicies - sum_indicies[-1]
	## Importance (avg) by frequency
	X[...,2] = _m_freq(freq_map,source,cand_indicies, word2idx)
	## Similarity(cand, [:3] prev summary)
	X[...,3:6] = _similarity(sum_sent, cand_sent)
	## #Noun/verb overlap with summary and Pr(w)
	X[...,6:] = _overlap(source, sum_indicies, cand_indicies,model, emis_dense)
	# print("interac X",X)
	return X


def _overlap(source, sum_indicies_origin, cand_indicies,  model, emis_dense):
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

	sum_indicies = sum_indicies_origin[1:] # skip "no summary": -1
	
	
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
		# numer = np.array([np.sum(e.tocsr().toarray()[word2idx.get(w,2)]) for e in emis]) ## UNK = 2
		numer = np.array([np.sum(e[word2idx.get(w,2)]) for e in emis_dense])
		denom = np.sum(numer)
		prob_all = np.log(numer+epsilon)-np.log(denom+epsilon)
		x += prob_all
	return x

###############
# Calculation #
###############
def forward(x,W, rm ={},n_pca = None, pca = None):
	## forward pass. W is a list of tuple, W[i][0] is weight, W[i][1] is bias. fn = [tanh, relu, sigmoid]
	x = feat_select(x, None, rm, n_pca = n_pca, pca = None)
	h1 = np.tanh(x.dot(W[0][0])+W[0][1])
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


def eval_summary():
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	summary,_ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	generated_idxs = pickle.load(open("generated_idxs.pkl","rb"))
	generated_summary = []
	for i,idx in enumerate(generated_idxs):
		gen_sum = [doc[i][s] for s in idx]
		generated_summary.append(gen_sum)
	score = rouge_score(generated_summary, summary)
	print(score)
	## compare ability of predicting EOS
	generated_length = np.array([len(s) for s in generated_idxs])
	true_length = np.array([len(s) for s in summary])
	mse = np.mean(np.square(generated_length - true_length))
	mse_baseline = np.mean(np.square(np.full(len(summary),2.6) - true_length))
	print("Length mse generated",mse,"mse_baseline",mse_baseline)


###############
#   Test      #
###############
def test_generate():
	idx = 15 # index of the file in test data set
	# rm = {"cand_se","src_se","sum_se"} # remove features while doing forward pass
	rm = {}

	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	summary,_ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+"2.pkl","rb"),encoding="latin-1",errors="ignore")
	# print("source length",len(doc[idx]))
	model = pickle.load(open(_model_path+topic+"_old.pkl","rb"),encoding="latin-1",errors="ignore")

	
	## Importance
	# for i in range(123):
	# 	gen_XM(doc[i],i)
	# 	_M_sent(i)
	# exit(0)

	## Original source document
	# s = open("/home/rldata/jingyun/nyt_corpus/content/"+re.sub("_annotated","",files[idx]).strip(".xml")).read()
	# print(s)
	# exit(0)

	## load model weights
	model_path = cur_path+"models/ffnn_2step_2.0/bi_classif_War_AllFeat(388, 1460)_nonbin.h5py"
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
	for idx in range(len(doc)):
		generated_idx = generate_summary(idx, doc[idx],model,weights,rm=rm)
		print("generated",generated_idx)
		generated_idxs.append(generated_idx)
		generated.append([doc[idx][s] for s in generated_idx])
		print("Gold standard", selected_tot[idx])

	pickle.dump(generated_idxs,open("generated_idxs.pkl","wb"))
	scores = rouge_score(generated, summary)
	print(scores)


if __name__ == '__main__':
	test_generate()
	
	# eval_summary()
