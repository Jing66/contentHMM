import numpy as np

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
	Y = np.load("../data/model_input/FFNN/Y_rdn1.npy")
	cl = np.load("../data/model_input/FFNN/candidate_length_rdn1.npy")
	sep = int(0.1*len(cl))
	cl_dev = cl[:sep]
	Y = Y[:np.sum(cl_dev)]
	print(dp)
	breakdown(yp,Y,cl_dev)

	# print(accuracy_at_k(yp,Y,cl_dev,1))