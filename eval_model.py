import numpy as np
import pickle
import sys
import os
import json
from rouge import Rouge
from pythonrouge.pythonrouge import Pythonrouge
import matplotlib.pyplot as plt

from all_feat import _length_indicator

topic ='War Crimes and Criminals'
input_path = "/home/ml/jliu164/code/contentHMM_input/"
_model_path = "/home/ml/jliu164/code/contentHMM_tagger/contents/"
save_path = "../filter_results/topic2files(content).pkl"
data_path = "../data/model_input/"
fail_path = '../contentHMM_input/fail/'
cand_rec_path = data_path+"FFNN/cand_record/"

EOD = "*EOD*" # predict the end
START_SENT = "**START_SENT**"
SOD = "**START_DOC**"
END_SENT = "**END_SENT**"

N_CAND = 10

if sys.version_info[0] <3:
	from corenlpy import AnnotatedText as A
	ds=2

	files_ = pickle.load(open(save_path))[topic]
	with open(fail_path+topic+"_Failed.txt") as f:
		paths = f.readlines()
		failed_l = [p.split("/")[-1].split(".")[0] for p in paths]
		failed = set(failed_l)
	files = [fs for fs in files_[:-1] if fs.split('/')[-1].split(".")[0] not in failed]
	files = files[int(np.around(0.1*len(files))):-int(np.around(0.1*len(files)))] if ds==1 else files[int(-np.around(0.1*len(files))):]
	p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
	
	with open(p_selected,'r') as f:
		selected_tot = json.load(f)

	
	# print(files[-6])
	# print(selected_tot[-6])

###############
# separate case #
###############

def acc_(yp,Y,cand_len):
	n_correct = 0
	idx = 0
	for i in range(len(cand_len)):
		yp_ = yp[idx:idx+cand_len[i]]
		Y_ = Y[idx:idx+cand_len[i]]
		
		if np.argmax(Y_)==np.argmax(yp_):
			n_correct += 1
		
		idx += cand_len[i]
	return float(n_correct)/len(cand_len)

def accuracy_at_k(yp,Y,cand_len,k):
	### calculate acc@k. For exact match, k=1. Caveat: k <= min(cand_len)
	assert np.sum(cand_len)==len(yp)==len(Y), (len(yp),len(Y),np.sum(cand_len))
	n_correct = 0
	idx = 0
	for i in range(len(cand_len)):
		yp_ = yp[idx:idx+cand_len[i]].ravel()
		Y_ = Y[idx:idx+cand_len[i]].ravel()
		
		indicies = yp_.argsort()[-k:]

		if np.argmax(Y_) in indicies:
			n_correct += 1
		# print(yp_,Y_)
		# print((indicies),np.argmax(Y_))
		yp_[:] = 0
		yp_[indicies] = 1
		idx += cand_len[i]
	return float(n_correct)/len(cand_len), yp


def accuracy(yp,Y):
	### calculate exact match for 0/1 downsample
	assert len(yp) == len(Y)
	return float(np.sum(yp==Y))/len(Y)



def breakdown(yp,y,cand_len):
	### print a breakdown of performance for predicted scores. compare ability to predict <FOS> and <EOS>
	print("************ Performance breakdown on predicting <First> and <EOS> ***************")
	# eos_scores_first = []
	# eos_scores_true = []
	# first_scores_last = []
	# first_scores_true = []
	eos_correct = 0
	n_eos=0
	fos_correct = 1 if np.argmax(yp[:cand_len[0]])==np.argmax(y[:cand_len[0]]) else 0
	n_fos=1
	
	idx = 0
	next_fos = False
	for i in range(len(cand_len)):
		
		yp_ = yp[idx:idx+cand_len[i]].ravel()
		y_ = y[idx:idx+cand_len[i]].ravel()
		
		## accuracy for EOS
		if np.argmax(y_)==cand_len[i]-1:
			n_eos+=1
			if np.argmax(yp_)==cand_len[i]-1:
				eos_correct +=1
			next_fos = True
			idx += cand_len[i]
			continue
		
		## accuracy for First
		if next_fos:
			# print("FOS")
			n_fos+=1
			
			if np.argmax(y_)==np.argmax(yp_):
				fos_correct +=1
			next_fos = False
		
		
		idx += cand_len[i]
	
	print("Accuracy on getting EOS right: %s, accuracy on getting first sentence: %s" %(float(eos_correct)/n_eos,float(fos_correct)/n_fos))
	print(n_eos, n_fos)
	print(eos_correct, fos_correct)

#########################
# EOS evaluation detail #
#########################
def eos_confusion(yp,y,cand_len):
	## build confusion matrix for EOS prediction
	from sklearn.metrics import confusion_matrix
	
	yp = yp.ravel()
	eos_idx = []
	eos_idx_pred = []
	cand_len_incr =  np.hstack((np.zeros(1), np.add.accumulate(cand_len)))
	idx = 0
	for i in range(len(cand_len)):
		yp_ = yp[idx:idx+cand_len[i]]
		y_ = y[idx:idx+cand_len[i]]
		## index for when should be predicting EOS
		if np.argmax(y_)==cand_len[i]-1:
			eos_idx.append(cand_len_incr[i]+np.argmax(y_))
		## index for when model IS predicting EOS
		if np.argmax(yp_)==cand_len[i]-1:
			eos_idx_pred.append(cand_len_incr[i]+np.argmax(yp_))
		idx += cand_len[i]
	eos_idx = np.array(eos_idx).astype(int)
	eos_idx_pred = np.array(eos_idx_pred).astype(int)
	
	y_true = np.zeros(y.shape).astype(int)
	y_true[eos_idx] = 1 # only consider EOS
	ypred = np.zeros(yp.shape).astype(int)
	ypred[eos_idx_pred] = 1
	
	print(confusion_matrix(y_true, ypred,labels=[0,1]))


###############
# Evaluation  extrinsically #
###############
def rouge_score(generated, truth):
	## y_true,yp: a list of sentences, each a list of word tokens. <EOS> excluded
	hyps = []
	for doc in generated:
		hyp = []
		# print("\n")
		for sent in doc:
			# print(sent)
			if sent[0] == SOD:
				hyp.append((" ").join(sent[2:-1]))
			else:
				hyp.append((" ").join(sent[1:-1]))
		hyps.append((" ").join(hyp))
	refs = []
	for doc in truth:
		hyp = []
		for sent in doc:
			if sent[0] == SOD:
				hyp.append((" ").join(sent[2:-1]))
			else:
				hyp.append((" ").join(sent[1:-1]))
		refs.append((" ").join(hyp))
	# rouge = Rouge()
	# scores = rouge.get_scores(hyps,refs,avg=True)

	## Perl version
	ROUGE_path ="/home/ml/jliu164/.local/lib/python3.6/site-packages/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl" 
	data_path = "/home/ml/jliu164/.local/lib/python3.6/site-packages/pythonrouge/RELEASE-1.5.5/data"
	rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, stemming=True, stopwords=True, word_level=True, length_limit=True, length=50, use_cf=False, cf=95, scoring_formula="average", resampling=True, samples=1000, favor=True, p=0.5)
	rouge_1 = []
	rouge_2 = []
	rouge_l = []
	rouge_su4 = []
	for hyp,ref in zip(hyps,refs):
		# print(hyp)
		# print(ref)
		setting_file = rouge.setting(files=False, summary=[[hyp]], reference= [[[ref]]])
		result = rouge.eval_rouge(setting_file, recall_only=True, ROUGE_path=ROUGE_path, data_path=data_path)
		# print(result) # {'ROUGE-1': 0.31429, 'ROUGE-2': 0.08824, 'ROUGE-L': 0.22857, 'ROUGE-SU4': 0.12887}
		rouge_1.append(result["ROUGE-1"])
		rouge_2.append(result["ROUGE-2"])
		rouge_l.append(result["ROUGE-L"])
		rouge_su4.append(result["ROUGE-SU4"])
	rouge_1 = np.array(rouge_1)
	rouge_2 = np.array(rouge_2)
	rouge_l = np.array(rouge_l)
	rouge_su4 = np.array(rouge_su4)
	scores = {"rouge_1":np.mean(rouge_1),"rouge_2":np.mean(rouge_2),"rouge_L":np.mean(rouge_l),"rouge_su4":np.mean(rouge_su4)}
	return scores # {"rouge-1": {"f": _, "p": _, "r": _}, "rouge-2" : { ..     }, "rouge-3": { ... }}


def oracle_chance(n_range = np.arange(15), ds_range = range(2,3)):
	### if every time take next 10 candidates, what's the chance of next summary sentence in candidate?
	data_path = "/home/ml/jliu164/code/data/model_input/"
	n_in = 0
	n_all = 0
	for ds in ds_range:
		p_selected = data_path+"FFNN/selected_sentences"+str(ds)+".json"
		with open(p_selected,'r') as f:
			selected_tot = json.load(f) # indices of article sentences selected as summary
		for selected in selected_tot:
			n_all += len(selected) #selected:[0,4,5,13]
			if selected[0] in n_range:
				n_in += 1 # first sentence
			for i in range(len(selected)-1):
				if selected[i+1]-selected[i] in n_range:
					n_in += 1
	print("candidate range = %s, computed over ds range %s, chances of having the correct sentence in candidate set is %s"%(n_range,ds_range,float(n_in)/n_all))
	print("n_all",n_all,"n_in",n_in)



#########################
#  Generate summary intrinsic   #
#########################
def generate_summary_fake(yp, index):
	### yp: predicted y. index: which article to generate summary. ds: 1 if dev set, 2 if test set
	### not truly generating summary: already assuming gold standard summary
	print("file path:", files[index])
	print("selected sentences", selected_tot[index])

	print("yp.shape", yp.shape)
	cand_len = np.load("../data/model_input/FFNN/candidate_length_rdn"+str(ds)+".npy")
	cand_rec = pickle.load(open(cand_rec_path+"rdn_sample"+str(ds)+".pkl","rb"))
	xml = open("/home/rldata/jingyun/nyt_corpus/content_annotated/"+files[index]).read()
	text = A(xml)

	n_cand_set = len(cand_rec[index])
	n_cand = sum([len(s) for s in cand_rec[index]])+len(cand_rec[index])
	print("n_cand_set",n_cand_set, "n_cand",n_cand)
	offset_cand_len = sum([len(s) for s in selected_tot[:index]])+index
	cand_len_ = cand_len[offset_cand_len:offset_cand_len+n_cand_set]
	# print(cand_len_)
	offset_y = sum([len(i) for s in cand_rec[:index] for i in s]) + sum([len(cand_rec[i]) for i in range(index)])
	# print("offset_y",offset_y)
	cand_y = yp[offset_y:offset_y+n_cand]
	
	print("****** original text:\n")
	print([_pretty_sentence(s) for s in text.sentences])
	print("\n******selecting summary:")
	targets = _generate(cand_y, text.sentences, cand_len_, cand_rec[index])
	print("\n******* Actual summary:")
	print([_pretty_sentence(text.sentences[i]) for i in np.array(selected_tot[index]).astype(int)])
	print("Selected sentences: %s, actual sentences: %s"%(targets,selected_tot[index]))


def _generate(y, source, cand_len, cand_rec):
	### given a source article (list of sentence), predicted y, candidate length each time, sample record, generate a summary with EOS
	idx=0
	# print("cand_len", cand_len)
	# print("cand_rec", cand_rec)
	# print(y)
	targets = []
	for i in range(len(cand_len)):
		c = cand_len[i]
		y_ = y[idx:idx+c]
		target = np.argmax(y_)
		
		if target == c-1:
			print("<EOS>")
			break
		else:
			yp = source[cand_rec[i][target]]
			targets.append(cand_rec[i][target])
			# print("i",i,"target",target,"index in source",cand_rec[i][target])
			print(_pretty_sentence(yp))
			
		idx += c
	return targets


def _pretty_sentence(sent):
	## return a readable sentence from AnnotatedText sentence. sent: one sentence
	ll = [i['word'] for i in sent['tokens']]
	return (" ").join(ll)



if __name__ == '__main__':

	# oracle_chance() # 0.64
	# exit(0)

	dp = "(1030, 727)"
	yp = np.load("pred/yp_ffnn_2step/"+dp+".npy")
	# generate_summary_fake(yp,14)

	Y = np.load("../data/model_input/FFNN/Y_rdn1.npy")
	cl = np.load("../data/model_input/FFNN/candidate_length_rdn1.npy")
	sep = int(0.1*len(cl))
	cl_dev = cl[:sep]
	Y = Y[:np.sum(cl_dev)]
	print(dp)
	# breakdown(yp,Y,cl_dev)
	eos_confusion(yp,Y,cl_dev)



