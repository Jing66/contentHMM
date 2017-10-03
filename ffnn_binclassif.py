import numpy as np
import json
import math

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.metrics import binary_accuracy as accuracy
import h5py
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest


from all_feat import feat_select
from eval_model import accuracy_at_k, accuracy
from utils import MyPCA, pca_we

data_path = "/home/ml/jliu164/code/data/"

np.random.seed(10)

def _downsample(X,Y):
	## return downsampled version of x and y
	Y = Y.ravel()
	pos_id = np.nonzero(Y)[0]
	neg_id = np.where(Y==0)[0]
	
	neg_sample_id = neg_id[np.random.choice(len(neg_id), len(pos_id), replace=False)]
	Y_pos = Y[pos_id]
	Y_neg = Y[neg_sample_id]
	X_pos = X[pos_id]
	X_neg = X[neg_sample_id]
	
	X = np.vstack((X_pos,X_neg))
	Y = np.vstack((Y_pos[:,np.newaxis],Y_neg[:,np.newaxis]))
	return X,Y
	


def load_data(suffix="",ds=1,dp="model_input/FFNN/", downsample = True, rm = set(),n_pca = 100, pca = None):
	# read X,Y
	try:
		X = np.load(data_path+dp+"X_rdn"+suffix+str(ds)+".npy")
	except IOError:
		print("X not generated")
		print(data_path+dp+"X_rdn"+suffix+str(ds)+".npy")
		exit(0)
		
	try:
		Y = np.load(data_path+dp+"Y_rdn"+suffix+str(ds)+".npy")
	except IOError:
		print("Y not generated")
		print(data_path+dp+"Y_rdn"+suffix+str(ds)+".npy")
		exit(0)
		
	assert X.shape[0]==Y.shape[0],(X.shape, Y.shape)
	Y = Y.reshape((-1,1))
	# print("Removing features...",rm)
	X = feat_select(X,Y,rm, n_pca = n_pca, pca = pca)
	return X,Y


###################################
############ Model ################
###################################

# binary classification
def build_base_model( dim_in ,h_sz = [170,100], fn = ["tanh",'relu','sigmoid'], weights = None):
	assert len(h_sz)+1 == len(fn)
	model = Sequential()
	model.add(Dense(h_sz[0], activation = fn[0], input_shape = (dim_in,))) # input layer
	#model.add(Dropout(0.2))
	for i in range(1,len(h_sz)):
		model.add(Dense(h_sz[i], activation = fn[i])) # hidden
	model.add(Dense(1, activation = fn[-1])) # last layer
	if weights:
		model.set_weights(weights)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model


def save(model, savename):
	# save model
    file = h5py.File(savename+'.h5py','w')
    weight = model.get_weights()
    for i in range(len(weight)):
            file.create_dataset('weight'+str(i),data=weight[i])
    file.close()


def load(model,savename):
	# load and set model
	file=h5py.File(savename+'.h5py','r')
	weight = []
	for i in range(len(file.keys())):
		weight.append(file['weight'+str(i)][:])
	model.set_weights(weight)
	return model


def train( savename = "models/ffnn_2step/bi_classif_War", rm_feat = {},n_pca = 100, downsample_dev = True):
	_p = "model_input/FFNN/"
	pca = pca_we() if n_pca else None
	#############
	## Data for First step of incremental training
	#############
	X,Y = load_data(suffix="_easy", ds=1,dp=_p, rm = rm_feat,n_pca = n_pca, pca = pca)
	if downsample_dev:
		X,Y = _downsample(X,Y) # test on downsampled distribution
	cand_len = np.load(data_path+_p+"candidate_length_rdn_easy1.npy").astype(int)
	sep = int(0.1*len(cand_len))
	cand_len_dev = cand_len[:sep]
	sep_x = int(np.sum(cand_len_dev))
	X_dev, X_train = X[:sep_x], X[sep_x:]
	Y_dev, Y_train = Y[:sep_x], Y[sep_x:]
	if not downsample_dev:
		X_train, Y_train = _downsample(X_train, Y_train) ## only downsample in training, test on true distribution
	X_train, Y_train = shuffle(X_train, Y_train)
	X_dev, Y_dev = shuffle(X_dev, Y_dev)
	#############
	## Data for second step of incremental training
	#############
	X2,Y2 = load_data(suffix="", ds=1,dp=_p,rm = rm_feat,n_pca = n_pca, pca = pca)
	cand_len2 = np.load(data_path+_p+"candidate_length_rdn1.npy").astype(int)
	if downsample_dev:
		X2,Y2 = _downsample(X2,Y2) # test on downsampled distribution
	sep = int(0.1*len(cand_len2))
	cand_len_dev2 = cand_len2[:sep]
	sep_x = int(np.sum(cand_len_dev2))
	X_dev2, X_train2 = X2[:sep_x], X2[sep_x:]
	Y_dev2, Y_train2 = Y2[:sep_x], Y2[sep_x:]
	if not downsample_dev:
		X_train2, Y_train2 = _downsample(X_train2, Y_train2) ## only downsample in training, test on true distribution
	X_train2, Y_train2 = shuffle(X_train2, Y_train2)
	X_dev2, Y_dev2 = shuffle(X_dev2, Y_dev2)

	print("X_dev.shape",X_dev.shape,"X_dev2.shape",X_dev2.shape)
	print("Removed features:",rm_feat)

	## config
	h_sz1 = np.random.random_integers(low= 100, high=2000,size=5)
	h_sz2 = np.random.random_integers(low= 100, high=1500,size=5)
	h_sz3 = np.random.random_integers(low= 100, high=1000,size=5)
	fn = ["tanh",'relu','relu','sigmoid']
	# h_sz1 = np.concatenate(h_sz1, 815)
	# h_sz2 = np.concatenate(h_sz2, 646)

	best_acc = 0
	best_mode = None
	results = None # performance
	best_config = None
	best_yp = None
	best_loss = np.inf
	
	for hiddens in zip(h_sz1,h_sz2):
	# for hiddens in h_sz1:
		print("\n>> Trying hidden size %s "%(hiddens,))
		
		model1 = build_base_model(X.shape[1],h_sz = hiddens)

		model1.fit(X_train,Y_train,epochs=20,verbose=0)
		score = model1.evaluate(X_dev,Y_dev)
		yp_raw = model1.predict(X_dev)
		## binary evaluation
		yp_ = np.copy(yp_raw)
		yp_[np.where(yp_>=0.5)]=1
		yp_[np.where(yp_<0.5)]=0
		acc = accuracy(yp_,Y_dev)
		print(yp_[:10]) 
		precision, recall, f1, _ = precision_recall_fscore_support(Y_dev,yp_)
		print("\nEasy binary Case. Accuracy: %s, precision: %s, recall: %s, F1: %s"%(acc, precision, recall, f1))
		## 1 from 11 task evaluation
		yp_raw_ = np.copy(yp_raw)
		yp_ = np.copy(yp_raw)
		acc1, yp = accuracy_at_k(yp_raw_,Y_dev,cand_len_dev,1)
		acc2,_ =  accuracy_at_k(yp_,Y_dev,cand_len_dev,2)
		acc = (acc1,acc2)
		print(yp[:10])
		precision, recall, f1, _ = precision_recall_fscore_support(Y_dev, yp)
		print("Easy choose 1 from 11 Case. Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s"%(acc, precision, recall, f1,score[0]))
		
		#############
		## second step of incremental training
		#############
		model2 = build_base_model(X.shape[1],h_sz = hiddens, weights=model1.get_weights())
		model2.fit(X_train2,Y_train2,epochs=20,verbose=0)
		score = model2.evaluate(X_dev2,Y_dev2)
		loss = score[0]
		yp_raw = model2.predict(X_dev2)
		## binary evaluation
		yp_ = np.copy(yp_raw)
		yp_[np.where(yp_>=0.5)]=1
		yp_[np.where(yp_<0.5)]=0
		acc = accuracy(yp_,Y_dev2) 
		precision, recall, f1, _ = precision_recall_fscore_support(Y_dev2,yp_)
		print("\nHard binary Case. Accuracy: %s, precision: %s, recall: %s, F1: %s, accuracy: %s"%(acc, precision, recall, f1, score[1]))
		## 1 from 11 task evaluation
		yp_raw_ = np.copy(yp_raw)
		yp_ = np.copy(yp_raw)
		acc1, yp = accuracy_at_k(yp_raw_,Y_dev2,cand_len_dev2,1)
		acc2,_ =  accuracy_at_k(yp_,Y_dev2,cand_len_dev2,2)
		acc = (acc1,acc2)
		precision, recall, f1, _ = precision_recall_fscore_support(Y_dev2, yp)
		print("Hard choose 1 from 11 Case.  Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s"%(acc, precision, recall, f1,score[0]))
		
		if loss < best_loss:
			best_model = (model1, model2)
			best_acc = acc
			results = (acc,precision,recall,f1)
			best_config = hiddens
			best_yp = yp_raw
			best_loss = loss
	print("\n\n###############")
	print("Best Model results on validation -- Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s"%(best_acc,results[1],results[2],results[3],best_loss))
	print("model hidden size:",best_config)
	print("Removed features:",rm_feat)
	print("Downsampled on validation:", downsample_dev)

	
	np.save("pred/yp_ffnn_2step/"+str(best_config),yp_raw)

	if savename:
		save(best_model[0],savename+str(best_config)+"_binary")
		save(best_model[1],savename+str(best_config)+"_nonbin")
		print("Saving directory: "+savename)
	# print("*** Testing ***")
	# test_model(best_model[1],savename = "pred/yp_ffnn_2step/ytest"+str(best_config), rm_feat = rm_feat, n_pca = n_pca, pca = pca)


def test_model(model, savename = "pred/ffnn_test/tmp", rm_feat ={}, n_pca = None, pca = None):
	# choose 1 from 11
	
	_p = "model_input/FFNN/"
	cand_len = np.load(data_path+_p+"candidate_length_rdn2.npy").astype(int)
	X,Y = load_data(suffix="", ds=2,dp=_p,downsample=True, rm = rm_feat,n_pca = n_pca, pca = pca)
	print("X.shape",X.shape)
	yp = model.predict(X)
	score = model.evaluate(X,Y)
	yp[np.where(yp>=0.5)]=1
	yp[np.where(yp<0.5)]=0
	acc = accuracy(yp,Y) 
	precision, recall, f1, _ = precision_recall_fscore_support(yp,Y)
	print("\nBinary Case. Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s, accuracy: %s"%(acc, precision, recall, f1,score[0], score[1]))
		

	X,Y = load_data(suffix="", ds=2,dp=_p,downsample=False, rm = rm_feat,n_pca = n_pca, pca = pca)
	print("X.shape",X.shape)
	score = model.evaluate(X,Y)
	yp_raw = model.predict(X)
	yp_save = np.copy(yp_raw)
	yp_raw_ = np.copy(yp_raw)
	
	acc1, yp = accuracy_at_k(yp_raw,Y,cand_len,1)
	
	acc2,_ =  accuracy_at_k(yp_raw_,Y,cand_len,2)
	acc = (acc1,acc2)
	precision, recall, f1, _ = precision_recall_fscore_support(yp,Y)
	print("\nChoose 1 from 11 Case. Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s"%(acc, precision, recall, f1,score[0]))
	
	np.save(savename,yp_save)



if __name__ == '__main__':
	
	## training
	### rm_feat: {"src_cluster","src_se","sum_cluster","sum_overlap","sum_pos","sum_posbin","sum_num","sum_se",
	# "cand_pos","cand_cluid","cand_prob","cand_M","cand_se",
	# "interac_trans","interac_pos","interac_M","interac_sim_nprev","interac_w_overlap","interac_emis"}
	train( rm_feat={"cand_se","src_se","sum_se"},n_pca = None, savename = None)
	train( rm_feat={},n_pca = 0 ,savename=None)
	# train (savename = "models/ffnn_2step_2.0/bi_classif_War_embeddings", rm_feat = {"src_cluster","sum_cluster","sum_overlap","sum_pos","sum_posbin","sum_num",
	# "cand_pos","cand_cluid","cand_prob","cand_M","interac_trans","interac_pos","interac_M","interac_sim_nprev","interac_w_overlap","interac_emis"}, downsample_dev=False)
	
	# testing
	# rm_feat={"cand_se","src_se","sum_se"}
	# model = build_base_model(64, h_sz=[815,646])
	# model = load(model, "/home/ml/jliu164/code/Summarization/models/ffnn_2step/bi_classif_War2(815, 646)_nonbin")
	# test_model(model, rm_feat = rm_feat)
	
