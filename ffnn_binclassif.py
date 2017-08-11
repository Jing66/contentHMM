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

from all_feat import load_data

data_path = "/home/ml/jliu164/code/data/"

def _accuracy(yp,Y):
	assert len(yp) == len(Y)
	return float(np.sum(yp==Y))/len(Y)


# binary classification
def build_base_model( dim_in ,h_sz = [170,100], fn = ["sigmoid",'relu','sigmoid']):
	assert len(h_sz)+1 == len(fn)
	model = Sequential()
	model.add(Dense(h_sz[0], activation = fn[0], input_shape = (dim_in,))) # input layer
	#model.add(Dropout(0.2))
	for i in range(1,len(h_sz)):
		model.add(Dense(h_sz[i], activation = fn[i])) # hidden
	model.add(Dense(1, activation = fn[-1])) # last layer
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

def train(savename = "models/bi_classif_War", downsample = True, rm_feat = {"src_Se"},n_feat = None, normalize=True):
	_p = data_path+"model_input/FFNN/"
	X,Y = load_data(normalize,ds=1,dp=_p,downsample=downsample, rm = rm_feat,n_feat = n_feat)
	
	print("X.shape",X.shape)
	print("Y.shape",Y.shape)
	# exit(0)
	pos_id = np.nonzero(Y)[0]
	neg_id = np.where(Y==0)[0]
	print("Y has %s pos label and %s neg label"%(len(pos_id),len(neg_id)))
	print("Removed features:",rm_feat)
	print("Downsample = %s; Normalize features = %s"%(downsample, normalize))
	
	N = len(X)
	X_dev, X_train = X[:int(0.1*N)],X[int(0.1*N):]
	Y_dev, Y_train = Y[:int(0.1*N)],Y[int(0.1*N):]
	pos_id = np.nonzero(Y_dev)[0]
	neg_id = np.where(Y_dev==0)[0]
	print("Y_dev has %s pos label and %s neg label"%(len(pos_id),len(neg_id)))
	# if downsample:
		# h_sz1 = np.random.random_integers(low=100 ,high=1500,size=5)
		# h_sz2 = np.random.random_integers(low=100 ,high=1000 ,size=5)
	# else:
	h_sz1 = np.random.random_integers(low= 200, high=2500,size=5)
	h_sz2 = np.random.random_integers(low= 500, high=1500,size=5)
	best_loss = np.inf
	best_mode = None
	results = None
	best_config = None
	for h1,h2 in zip(h_sz1,h_sz2):
		print("\n>> Trying hidden size [%s,%s] "%(h1,h2))
		model = build_base_model(X.shape[1],h_sz = [h1,h2])
		model.fit(X_train,Y_train,epochs=20,verbose=0)
		score = model.evaluate(X_dev,Y_dev)
		yp = model.predict(X_dev)
		yp[np.where(yp>=0.5)]=1
		yp[np.where(yp<0.5)]=0
		
		acc = _accuracy(yp,Y_dev)
		precision, recall, f1, _ = precision_recall_fscore_support(yp,Y_dev)
		print("\nOn Dev set--- Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s, binary_accuray:%s"%(acc, precision, recall, f1,score[0],score[1]))
		if score[0]<best_loss:
			best_model = model
			best_loss = score[0]
			results = (acc,precision,recall,f1)
			best_config = (h1,h2)
	print("\n\n\n###############")
	print("Best Model results on validation -- Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s"%(results[0],results[1],results[2],results[3],best_loss))
	print("model hidden size: %s; Downsample = %s; Normalize features = %s; Select n_feature: %s"%(best_config,downsample, normalize,n_feat))
	print("Removed features:",rm_feat)

	if savename:
		save(best_model,savename)
		print("Saving directory: "+savename)
	# test_model(downsample,normalize, rm_feat, model=best_model,n_feat = n_feat)


def test_model(downsample, normalize, rm_feat, model=None,n_feat = None,savename = "models/bi_classif_War",dim_in = 306,h_sz = None):
	if not model:
		model = build_base_model(dim_in,h_sz)
		load(model,savename)
	X,Y = load_data(normalize, ds=2,downsample=downsample,rm = rm_feat,n_feat = n_feat)
	pos_id = np.nonzero(Y)[0]
	neg_id = np.where(Y==0)[0]
	# print("Y_test has %s pos label and %s neg label"%(len(pos_id),len(neg_id)))
	score = model.evaluate(X,Y)

	yp = model.predict(X)
	yp[np.where(yp>=0.5)]=1
	yp[np.where(yp<0.5)]=0
	acc = _accuracy(yp,Y)
	precision, recall, f1, _ = precision_recall_fscore_support(yp,Y)
	print("\n>> Test results -- Accuracy: %s, precision: %s, recall: %s, F1: %s, loss:%s, binary_accuray:%s"%(acc, precision, recall, f1,score[0],score[1]))

#######################################
############ Execution ################
#######################################

if __name__ == '__main__':
	# test()

	### rm_feat: ['src_cluster', 'src_se', 'sum_cluster', 'sum_overlap', 'sum_pos', 'sum_num', 'sum_se', 'cand_pos', 
	#'cand_cluid', 'cand_prob', 'cand_M', 'cand_se', 'interac_trans', 'interac_sim', 'interac_pos', 'interac_overlap']
	train(downsample=True,savename = None, rm_feat={'src_se',"sum_se","cand_se"},n_feat =None )
