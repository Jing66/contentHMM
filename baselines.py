# calculates baseline for all models
import pickle
import numpy as np
import os
import sys
import json

from eval_model import rouge_score
sys.path.append(os.path.abspath('..'))
from content_hmm import *


input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
data_path = "/home/ml/jliu164/code/data/model_input/"
cand_rec_path = data_path+"FFNN/cand_record/"

topic ='War Crimes and Criminals'



def random_baseline():
	ds = 1
	cand_len = np.load(data_path+"FFNN/candidate_length_rdn_easy"+str(ds)+".npy")

	pTrue = float(len(cand_len))/np.sum(cand_len)
	print("Naving adding together way. pTrue = %s"%(pTrue))

	num_prob = cand_len/np.sum(cand_len)
	pt_prob = 1/cand_len
	pTrue = np.sum(num_prob*pt_prob)
	print("Serious Baseline pTrue = %s"%(pTrue))


def greedy_pick(ds=1):
	### use content model transitions to pick next summary sentence and calculated the baseline
	### Cannot predict EOS
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	# summary, _ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"))
	# doc_flat = [i for val in doc for i in val]
	model = pickle.load(open(_model_path+topic+".pkl","rb"))
	# len_ind = _length_indicator(ds=ds,topic=topic)
	cand_rec = pickle.load(open(cand_rec_path+"rdn_sample"+str(ds)+".pkl",'rb'))
	p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f) # indices of article sentences selected as summary

	_, tagged_doc = model.viterbi(doc)
	transition = model._trans
	prior = model._priors

	n_correct_first = 0
	n_correct = 0
	n_tot = 0
	n_tot_fos = 0
	idx = 0
	for i in range(len(selected_tot)):
		print("\n>>File #%s"%(i))
		# if idx>351:
		# 	break
		n_tot_fos +=1
		y_true = np.array(selected_tot[i]).astype(int)
		cand_rec_ = cand_rec[i]
		tagged = tagged_doc[idx:idx+len(doc[i])]
		
		n_tot += len(y_true)+1 # +1 for EOS
		# print("y_true",y_true)
		# print("tagged document",tagged)
		# pick first sentence
		cand_tags = np.array([tagged[t] for t in cand_rec_[0]]).astype(int)
		prior_ = prior[cand_tags]
		target = np.argmax(prior_)
		# print("target",target)
		if target == y_true[0]:
			n_correct += 1
			n_correct_first+=1 
		# pick the 2nd....nth sentence
		for k in range(len(y_true)-1):
			last_select = y_true[k]
			last_topic = tagged[last_select]
			next_trans = transition[last_topic] # a vector of logprob from last topic to all other topics
			cand_tags = np.array([tagged[t] for t in cand_rec_[k+1]]).astype(int) # topics of all candidate sentences
			cand_trans = next_trans[cand_tags]
			target = cand_rec_[k+1][np.argwhere(cand_trans==np.max(cand_trans))]
			rdn_idx = np.random.choice(len(target))
			target = target[rdn_idx]
			# print("candidates indicies",cand_rec_[k+1])
			# print("candidates cluster ids", cand_tags)
			# print("last cluster id",last_topic)
			# print("trans",next_trans)
			# print("target",target)
			if y_true[k+1] in target:
				n_correct +=1
		# pick EOS: always wrong
		idx += len(doc[i])
	acc = float(n_correct)/n_tot
	
	n_selected = np.sum(np.array([len(k) for k in selected_tot]))
	precision = float(n_correct)/n_selected
	recall = acc
	f1 = 2*precision*recall/(precision+recall)
	
	fos_acc = float(n_correct_first)/n_tot_fos
	print("n_correct",n_correct, "n_correct_first", n_correct_first)
	print("n_tot", n_tot, "n_tot_fos", n_tot_fos)
	print("Greedy transition baseline accuracy:%s, F1: %s"%(acc,f1))
	print("Predict First sentence accuracy:", fos_acc)


def top_k_rouge(ds=2):
	## Lead baseline.
	summary,_ = pickle.load(open(input_path+"summaries/"+topic+"/"+topic+str(ds)+".pkl","rb"),encoding="latin-1",errors="ignore")
	docs,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+str(ds)+".pkl","rb"),encoding="latin-1",errors="ignore")
	
	lw = 42 # median number of words per summary +1 (START_DOC)
	generation = []
	for d in docs:
		count = 0
		gen = []
		for sent in d:
			if count>lw:
				break
			else:
				gen.append(sent[:min(lw-count, len(sent))])
				count += len(sent)-2
		# print(gen)
		generation.append(gen)
	score = rouge_score(generation, summary)
	print(score)



def top_k(ds=2):
	### choose top k sentences where k = 0.1*len(article) as baseline. after chosen top k, predict EOS
	### DISCARDED: At summary level use top_k_rouge.
	from all_feat import _length_indicator
	p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f) # indices of article sentences selected as summary
	len_ind = _length_indicator() # length for each source article

	n_correct_eos = 0
	n_correct_fos = 0
	n_correct = 0
	n_tot = 0
	for i in range(len(len_ind)):
		y_true = selected_tot[i]
		yp = np.arange(int(0.1*len_ind[i]))
		n_tot += len(y_true)+1 # +1 for EOS
		## First sentence prediction
		if y_true[0]==0:
			n_correct_fos +=1
		## EOS prediction
		if len(y_true)==len(yp):
			n_correct_eos +=1
		## all prediction
		l = min(len(y_true),len(yp))
		eq = np.equal(y_true[:l], yp[:l])
		n_correct+=np.sum(eq)

	acc = float(n_correct)/n_tot
	eos_acc = float(n_correct_eos)/len(len_ind)
	fos_acc = float(n_correct_fos)/len(len_ind)
	print("n_correct",n_correct, "n_correct_first", n_correct_fos, "n_correct_EOS", n_correct_eos)
	print("n_tot", n_tot)
	print("Top k baseline accuracy:",acc)
	print("Predict <EOS> accuracy:", eos_acc)
	print("Predict <First> accuracy:", fos_acc)


def importance(ds=1,context_size = 4):
	### Choose next sentence based on importance score produced by Aishikc's model. can't predict EOS.
	from all_feat import _get_importance,_length_indicator
	M = _get_importance(topic, ds, 4) # M size (#sentences in all doc, 2)
	M = M[...,0] # only use the average
	print(M.shape)
	p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
	with open(p_selected,'r') as f:
		selected_tot = json.load(f) # indices of article sentences selected as summary
	len_ind = _length_indicator() # length for each source article
	cand_rec = pickle.load(open(cand_rec_path+"rdn_sample"+str(ds)+".pkl",'rb'))

	fos_correct = 0
	n_correct = 0
	n_tot = 0
	idx = 0
	for i in range(len(len_ind)):
		y_true = selected_tot[i]
		cand_rec_ = cand_rec[i]
		M_local = M[idx:idx+len_ind[i]]
		idx +=len_ind[i]
		n_tot += len(y_true) +1 # +1 for EOS
		# pick first summary sentence
		target = np.argmax(M_local[cand_rec_[0]])
		if target == y_true[0]:
			n_correct += 1
			fos_correct += 1
		# pick the 2nd....nth sentence
		for k in range(len(y_true)-1):
			last_select = y_true[k]
			target = np.argmax(M_local[cand_rec_[k+1]])
			if target == y_true[k+1]:
				n_correct += 1
	print("n_correct:%s, n_tot: %s, fos_correct:%s"%(n_correct, n_tot, fos_correct))
	n_selected = np.sum(np.array([len(k) for k in selected_tot]))
	acc = float(n_correct)/n_tot
	precision = float(n_correct)/n_selected
	recall = acc
	f1 = 2*precision*recall/(precision+recall)
	fos_acc = float(fos_correct)/len(len_ind)
	print("Importance baseline accuracy:%s, F1: %s"%(acc,f1))
	print("Predict <First> accuracy:", fos_acc)


if __name__ == '__main__':
	# random_baseline()
	# greedy_pick()
	# top_k()
	# importance()

	top_k_rouge()