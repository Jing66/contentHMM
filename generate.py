import numpy as np
import pickle
import sys
import os
import json
from collections import Counter
import h5py

from rdn_sample_cand import _x_EOD
from utils import _gen_file_for_M as _gen_X4M
if sys.version_info[0] >=3:
	from pred_M import M_predict
else:
	sys.path.append(os.path.abspath('..'))
	from content_hmm import ContentTagger

topic ='War Crimes and Criminals'
input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
data_path = "/home/ml/jliu164/code/data/"
cur_path = "/home/ml/jliu164/code/Summarization/"
utils_path = "/home/ml/jliu164/code/data/utils/"

START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"
SKIP_SET = {SOD,END_SENT,START_SENT}

N_CAND = 10
DIM_EMB = 300


def generate_summary(source, model, weights):
	""" 
	Given source article and content model, no gold standard summary, generate a summary.
	source: list of sentences, each list of words. model: content HMM model. weights: list of numpy arrays for doing prediction
	"""
	dim_src = DIM_EMB + model._m
	dim_sum = DIM_EMB + model._m + 4
	dim_cand = DIM_EMB + model._m + 4
	dim_interac = 8+2*(model._m-1)
	n_dim = dim_src + dim_interac+dim_cand+dim_sum

	## [source] = [embedding] + [cluster]
	X_src = np.zeros(dim_src)
	try:
		x_se = np.load("tmp_embeddings.npy")
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
		
		np.save("tmp_embeddings",x_se)
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
	count=0
	chosen = np.array([-1]).astype(int) # indicies of selected summary
	while not eos and count < len(source):
		print("generating %s summary sentence:"%(count))
		## feature vector for summary so far. same for all candidates thus n_row = 1
		X_sum = np.zeros(dim_sum) 
		X_sum[:14] = sum_so_far(source,chosen,model)
		X_sum[-DIM_EMB:] = np.mean(x_se[chosen[1:]],axis=0) # average all chosen summary embeddings. ALT: last embedding
		X_fixed = np.hstack((X_src, X_sum)) # fixed part of the vector
		print("X_fixed shape", X_fixed.shape)
		n_i = np.arange(chosen[-1]+1, len(source)) # candidate indicies

		## TODO: feature vector for interaction
		X_interac = interac(source, chosen, n_i, model)

		eos_vec = _x_EOD(X_src, X_sum, n_dim, len(source), chosen[-1]) ## feature vector for eos. appended at the end
		x_cand = np.hstack((X_cand[n_i], X_interac))
		tmp = np.broadcast_to(X_fixed, (len(n_i), X_fixed.shape[0]))
		x_cand = np.hstack((tmp, x_cand))

		X = np.vstack((x_cand, eos_vec))
		print("X.shape",X.shape)

		## X(11 x n_dim) * W(n_dim x 1) => y_hat(11 x 1)
		y_hat = forward(X, weights)
		print("yp.shape",y_hat.shape)
		target = np.argmax(y_hat)

		if target == len(X)-1:
			print("<EOS>")
			eos = True
		else:
			np.append(chosen,target)
		print("chosen summary indicies",chosen)
		count += 1
	return chosen



def sum_so_far(source, indicies,model):
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
		bins = np.arange(-1,len(source)+1,float(len(s))/n_bin)
		X[-2] = np.digitize(indicies[-1],bins)
	X[-1] = len(indicies)-1 # number of summary so far
	# print(X)
	return X



def interac(source, sum_indicies,cand_indicies,model):
	""" compute feature vec for interaction
		return size (#candidate x dim_interac)
	"""
	return np.zeros((len(cand_indicies),8+2*(model._m-1)))


def forward(x,W):
	## forward pass. W is a list of tuple, W[i][0] is weight, W[i][1] is bias
	inputs = x
	for ws in W:
		h = inputs.dot(ws[0])+ws[1]
		inputs = h
	return h.reshape((-1,1))

###############
# Importance score#
###############
#generate importance score for each sentence on the fly
def gen_XM(source,context_sz =4):
	tmpf = "/home/ml/jliu164/code/data/importance_input/generation.txt"
	try:
		f1 = open(tmpf,"r")
	except FileNotFoundError:
		print("file for sentence importance score not generated")
		vocab_path = utils_path+"word2idx.json"
		with open(vocab_path) as f:
			vocab = json.load(f)
		print("In total %s word to index mapping"%(len(vocab)))
		tmpf = data_path+"importance_input/generation.txt"
		_gen_X4M(source,vocab,context_sz = context_sz,filename = tmpf)

# use python3
def _M_sent(context_sz =4):
	genf = cur_path+"pred/Importance/"+str(context_sz)+"_generation.npy"
	try:
		sent_M =np.load(genf)
	except FileNotFoundError:
		print("sentence importance score not generated")
		tmpf = data_path+"importance_input/generation.txt"
		
		f = open(tmpf,"r")
		sent_M = M_predict(f, cur_path+"pred/Importance/"+str(context_sz)+"_generation.npy")
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
	source = doc[2]
	print("source length",len(source))
	model = pickle.load(open(_model_path+topic+".pkl","rb"))

	## load model weights
	model_path = cur_path+"models/ffnn_2step/bi_classif_War_Allfeat(1270, 150)_binary.h5py"
	file = h5py.File(model_path,"r")
	weights = []
	for i in range(0,len(file.keys()),2):
		weights.append((file['weight'+str(i)][:], file['weight'+str(i+1)][:]))
	print(">> Model loaded. %s hidden layers: %s"%(len(weights)-1,[k[1].shape[0] for k in weights[:-1]]))

	generate_summary(source, model,weights)



if __name__ == '__main__':
	test()

	