# lexical features involving NER/POS -- have to read xml
import pickle
import numpy as np
from corenlpy import AnnotatedText as A
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import re
import json
import math
from new_baseline import _length_indicator
from content_hmm import *

root_dir = "/home/rldata/jingyun/nyt_corpus/content_annotated/"
fail_path = '/home/ml/jliu164/code/contentHMM_input/fail/'
save_path = "/home/ml/jliu164/code/filter_results/topic2files(content).pkl"
data_path = "/home/ml/jliu164/code/data/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"

topic ='War Crimes and Criminals'
savedir = data_path+"model_input/FFNN/"
NUM_CAND = 10
UNK =2
n_dim = 2
ds = 1
epsilon = 1e-8


files_ = pickle.load(open(save_path))[topic]
with open(fail_path+topic+"_Failed.txt") as f:
	paths = f.readlines()
	failed_l = [p.split("/")[-1].split(".")[0] for p in paths]
	failed = set(failed_l)
model = pickle.load(open(_model_path+topic+".pkl","rb"))
n_clus = model._m -1
n_dim_ = n_dim+2*n_clus
word2idx = model._map
len_ind = _length_indicator(ds=ds,topic=topic)

files = [fs for fs in files_[:-1] if fs.split('/')[-1].split(".")[0] not in failed]
selected = []

p_selected = data_path+"model_input/FFNN/selected_sentences"+str(ds)+".json"
with open(p_selected,'r') as f:
	selected = json.load(f) # indices of article sentences selected as summary
files = files[int(-np.around(0.1*len(files))):] if ds==2 else files[int(np.around(0.1*len(files))):-int(np.around(0.1*len(files)))]
print("**** Total %s files, %s selected to process******"%(len(files),len(selected)))
assert len(files)==len(selected)
# files = files[:3]
# selected = selected[:3]

def _emis_uni(words):
	## return emission logprob for words by model from each topic
	## return (model._m, )
	x = np.zeros(n_clus)
	emis = model._emis # emis is just the counts
	for w in words:
		if w in STOPWORDS:
			continue
		numer = np.array([np.sum(e.tocsr().toarray()[word2idx.get(w,UNK)]) for e in emis])
		denom = np.sum(numer)
		prob_all = np.log(numer+epsilon)-np.log(denom+epsilon)
		
		x += prob_all
	# print("emission logprob",x)
	return x




def _overlap(cands,sum_so_far):
	### given a list of sentences and summaries, count the overlapping between each candidate sent vs. all summary so far (0:2)
	### also include logprob of those noun/verbs from the content model (2:4)
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
			x_[2:2+n_clus] = _emis_uni(n_verb)
		if n_noun:
			x_[1] = float(len(n_noun))/math.sqrt(len(cand_token))
			x_[-n_clus:] = _emis_uni(n_noun) 
		x[i,...] = x_
	return x


X = np.zeros(n_dim_)
i=0
for f in files:
	selected_idx =selected[i]
	if f.split("/")[-1] in failed:
		continue
	xml = open(root_dir+f).read()
	text = A(xml).sentences
	sum_sent = [text[int(idx)] for idx in selected_idx]
	## first start
	if selected_idx[0] < NUM_CAND:
		n_i = min(NUM_CAND,len_ind[i])
	else:
		n_i = int(selected_idx[0]+1)
	sents = text[:n_i]
	X_ = _overlap(sents,[])
	## loop inside all indicies
	for idx_ in range(len(selected_idx)-1):
		cur_idx = selected_idx[idx_]
		next_idx = selected_idx[idx_+1]
		n_i = min(NUM_CAND+cur_idx+1, len_ind[i]) - cur_idx -1 if next_idx<NUM_CAND+cur_idx else (next_idx-cur_idx)
		# print("selected idx:%s. next locally fill in rows[%s-%s). n_i=%s"%(cur_idx, cur_idx+1,cur_idx+n_i+1,n_i))
		if n_i:
			x = _overlap(text[int(cur_idx+1):int(cur_idx+n_i+1)],sum_sent[:idx_])
			X_ = np.vstack((X_,x))
	## Last row
	n_i = min(NUM_CAND+selected_idx[-1]+1, len_ind[i]) - selected_idx[-1] -1
	# print("Lastly selected idx:%s. lastly locally fill in rows[%s-%s)."%(selected_idx[-1],selected_idx[-1]+1,selected_idx[-1]+n_i+1))
	if n_i:
		x = _overlap(text[int(selected_idx[-1]+1):int(selected_idx[-1]+1+n_i)],sum_sent)	
		X_ = np.vstack((X_,x))
	X = np.vstack((X,X_))	
	i+=1
X = X[1:]
print("X_lexical.shape",X.shape)
np.save(savedir+"X_lexical"+str(ds),X)
