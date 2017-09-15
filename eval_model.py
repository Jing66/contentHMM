import numpy as np
import pickle
import sys
import os
import json


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
#  Generate summary   #
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


def generate_summary(source, model):
	### Given source article and content model, no gold standard summary, generate a summary.
	### source: list of sentences, each list of words.
	## [source] = [embedding] + [cluster]
	with open("/home/ml/jliu164/code/data/utils/we_file.json") as f:
		We = json.load(f)
		unk_vec = We["UNK"]
	x_se = np.zeros((len(source),len(unk_vec))) #(#sent,300) for embedding
	for i in range(len(source)):
		sent_vec = np.zeros(len(unk_vec))
		count = 0
		for w in source[i]:
			if w in set([START_SENT, END_SENT,SOD]):
				continue
			v = We.get(w,unk_vec)
			sent_vec += v
			count += 1
		sent_vec /= count
		x_se[i] = sent_vec
	x_se_source = np.mean(x_se,axis=0)
	x_dist = np.zeros(model._m) #(10,) for distribution
	_,flat = model.viterbi(source)
	flat_count = dict(Counter(flat))
	for c_id, c in flat_count.items():
		x_dist[c_id] = c
	## [cand] = [pos] + [emis] + [importance] + [embeddings]
	flat_sent =  [i for val in source for i in val]
	X_cand = np.zeros((len(source),314)) # 314 is dimension of candidate
	X_cand[...,0] = np.arange(len(source)) #(#sent)
	X_cand[...,1] = np.array(flat)
	X_cand[...,2:2+model._m] =  model.sent_logprob(flat_sent) #(#sent,10)
	M = _importance(source)
	X_cand[...,2+model._m: 4+model._m] = M
	X[...,-4-model._m:] = x_se


# TODO: generate importance score for each sentence on the fly
def _importance(source):
	return np.zeros((len(source),2))

#########################
# Evaluate EOS in binary #
#########################








def _pretty_sentence(sent):
	## return a readable sentence from AnnotatedText sentence. sent: one sentence
	ll = [i['word'] for i in sent['tokens']]
	return (" ").join(ll)



def test():
	sys.path.append(os.path.abspath('..'))
	from content_hmm import *
	doc,_ = pickle.load(open(input_path+"contents/"+topic+"/"+topic+"2.pkl","rb"))
	source = doc[0]
	model = pickle.load(open(_model_path+topic+".pkl","rb"))
	generate_summary(doc, model)


if __name__ == '__main__':
	test()
	exit(0)

	dp = "(1030, 727)"
	yp = np.load("pred/yp_ffnn_2step/"+dp+".npy")
	# generate_summary_fake(yp,14)

	Y = np.load("../data/model_input/FFNN/Y_rdn1.npy")
	cl = np.load("../data/model_input/FFNN/candidate_length_rdn1.npy")
	sep = int(0.1*len(cl))
	cl_dev = cl[:sep]
	Y = Y[:np.sum(cl_dev)]
	print(dp)
	breakdown(yp,Y,cl_dev)




