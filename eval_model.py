import numpy as np
import pickle
from corenlpy import AnnotatedText as A

topic ='War Crimes and Criminals'
save_path = "../filter_results/topic2files(content).pkl"
data_path = "../data/model_input/"
fail_path = '../contentHMM_input/fail/'
cand_rec_path = data_path+"FFNN/cand_record/"


files_ = pickle.load(open(save_path))[topic]
with open(fail_path+topic+"_Failed.txt") as f:
	paths = f.readlines()
	failed_l = [p.split("/")[-1].split(".")[0] for p in paths]
	failed = set(failed_l)
files = [fs for fs in files_[:-1] if fs.split('/')[-1].split(".")[0] not in failed]



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
	eos_scores_first = []
	eos_scores_true = []
	first_scores_last = []
	first_scores_true = []

	idx = 0
	for i in range(len(cand_len)):
		yp_ = yp[idx:idx+cand_len[i]].ravel()
		y_ = y[idx:idx+cand_len[i]].ravel()
		
		if np.argmax(y_)==cand_len[i]-1:
			eos_scores_true.append(yp_[-1])
			eos_scores_first.append(yp_[0])
		if np.argmax(y_) == 0:
			first_scores_true.append(yp_[0])
			first_scores_last.append(yp[-1])
		
		idx += cand_len[i]
	eos_diff = np.array(eos_scores_true) - np.array(eos_scores_first)
	first_diff = np.array(first_scores_true) - np.array(first_scores_last)
	print("Difference scores predicting <EOS> on first candidate vs. on EOS: total %s cases"%(len(eos_diff)))
	print("max diff: %s; min diff: %s; mean diff: %s; std diff: %s"%(np.max(eos_diff),np.min(eos_diff),np.mean(eos_diff),np.std(eos_diff)))
	print("Difference scores predicting <FOS> on EOS vs. on first candidate : total %s cases"%(len(first_diff)))
	print("max diff: %s; min diff: %s; mean diff: %s; std diff: %s"%(np.max(first_diff),np.min(first_diff),np.mean(first_diff),np.std(first_diff)))


def generate_summary(y, source, cand_len, cand_rec):
	### given a source article (list of sentence), predicted y, candidate length each time, sample record, generate a summary with EOS
	idx=0
	for i in range(len(cand_len)):
		c = cand_len[i]
		y_ = y[idx:idx+c]
		target = np.argmax(y_)
		print(y_, target)
		if target == c-1:
			print("<EOS>")
			break
		else:
			yp = source[cand_rec[i][target]]
			print(yp)
			
		idx += c
		


def test():
	yp = np.array([0.15,0.2,0.1,0.05,0.2,0.8,0.1])
	y = np.zeros_like(yp)
	y[-2]=1
	y[0]=1
	cand_len = np.array([4,3])
	acc = _accuracy_at_k(yp,y,cand_len,1)
	print(acc)



if __name__ == '__main__':
	# test()

	dp = "2step(1048, 351, 864)"
	yp = np.load("pred/yp_ffnn_rdn_"+dp+".npy")
	# Y = np.load("../data/model_input/FFNN/Y_rdn1.npy")
	# cl = np.load("../data/model_input/FFNN/candidate_length_rdn1.npy")
	# sep = int(0.1*len(cl))
	# cl_dev = cl[:sep]
	# Y = Y[:np.sum(cl_dev)]
	# print(dp)
	# breakdown(yp,Y,cl_dev)

	cand_len = np.load("../data/model_input/FFNN/candidate_length_rdn1.npy")
	cand_rec = pickle.load(open(cand_rec_path+"rdn_sample.pkl","rb"))
	xml = open("/home/rldata/jingyun/nyt_corpus/content_annotated/"+files[-1]).read()
	text = A(xml).sentences
	local_idx = len(text)+1
	local_idx_y = np.sum(cand_len[-local_idx:])
	print("****** original text:\n")
	print(text)
	print("******selecting summary:\n")
	generate_summary(yp[-local_idx_y:], text, cand_len[-local_idx:], cand_rec[-1])
	print(cand_rec[-1])



