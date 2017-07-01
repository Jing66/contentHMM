import time
import pickle
import logging
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import scipy.cluster.hierarchy as hac
from scipy.misc import comb
from corenlpy import AnnotatedText as A
import numpy as np
import sys
import traceback
import logging
import exceptions
import re
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

from content_hmm import similarity, make_cluster_tree

START = "*START*"
END = "*END*"

# sum_root = "/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/"
sum_root = "/home/rldata/jingyun/nyt_corpus/summary_annotated/"


########################################## Set up logging #######################################
def setup_logging_to_file(filename):
    logging.basicConfig( filename=filename,
                         filemode='w',
                         level=logging.DEBUG,
                         format= '%(asctime)s - %(levelname)s - %(message)s',)


def log_exception(e):
    logging.error(
    "Function {function_name} raised {exception_class} ({exception_docstring}): {exception_message}".format(
    function_name = extract_function_name(), #this is optional
    exception_class = e.__class__,
    exception_docstring = e.__doc__,
    exception_message = e.message))

def extract_function_name():
    tb = sys.exc_info()[-1]
    stk = traceback.extract_tb(tb, 1)
    fname = stk[0][3]
    return fname



#######################################################################################################
##############################  Find all word sequences around NER  ###################################
#######################################################################################################
def gen_input(files, thread = 5, NER = ["PERSON","ORGANIZATION"], save_file = False, save_file_name = ["NER_input/person.pkl","NER_input/org.pkl"]):
	"""
	Generate representation(words around it) of a list of NER across files. Save as a dictionary {id:([w0,w1,w2, w3,w4],index position of NER word)}
	"""
	
	print("Generating representation for %s..."%(NER))
	if save_file:
		assert len(NER) == len(save_file_name)
	counter = 0
	start_time = time.time()
	results = []

	pool = ThreadPool(thread)
	results = pool.map(_gen_input, itertools.izip(files,itertools.repeat(NER)))
	pool.close()
	pool.join()

	out_raw = []
	outs = []
	for i in range(len(NER)):
		tmp = [result[i] for result in results if result]
		tmp = [i for val in tmp for i in val]
		out_raw.append(tmp)
	
	for out in out_raw:
		N = len(out)
		tmp = dict(zip(range(N),out))
		outs.append(tmp)

	if not save_file:
		return outs
	else:
		for i in range(len(outs)):
			pickle.dump(outs[i],open(save_file_name[i],"wb"))


def _gen_input((f,NER)):
	"""return `len(NER)` dictionaries in list. key: NER element, value: ([words],index)"""
	results = []

	for i in range(len(NER)):
		results.append([])
	try:
		xml = open(f).read()
	except IOError:
		print("File not exist: "+f)
	else:
		text = A(xml)
		
		for sent in text.sentences:
			for i in range(len(NER)):

				# try:
				repre = _extractor(sent,NER[i]) # list of words as context around the NER

					# print(repre)
				# except Exception as e:
				# 	print("Catch exception. Logged.")
				# 	log_exception(e)
				# 	pass
				# else:
				if repre:
					results[i].extend(repre)
		
		return results
		

def _extractor(sent, NER):
	# use lemma instead of words
	out = []
	out_indices = [] # list of [index], each element is a group of NER found in sentence
	# tokens = sent['tokens']
	# remove punctuation, keep numbers or those has NER
	
	tokens = [i for i in sent['tokens'] 
		if i['lemma'].isalpha()  or re.findall(r'^\-?[1-9][0-9]*\.?[0-9]*',i['lemma']) and i['lemma'] not in STOPWORDS
			or i['ner']]

	N = len(tokens)
	# print("Sentence has length %s"%(N))
	if N <= 9:
		for i in range(N):
			w = tokens[i]
			if w["ner"] == NER:
				idx = _group_ner(tokens,i)
				out_indices.append(idx)
		indices = _merge_tree(out_indices)
		if indices:
			word_list = [t["lemma"] for t in tokens]
			return [([START]+word_list+[END], indices)]
		else:
			return []	
	for i in range(3)+range(N-4,N):
		if tokens[i]["ner"] == NER:
			idx = _group_ner(tokens,i)
			out_indices.append(idx)

	for i in range(4, len(tokens)-4):
		# print("Checking %s : NER == %s"%(sent["tokens"][i]['word'],sent["tokens"][i]['ner']))
		if tokens[i]['ner'] == NER:
			idx = _group_ner(tokens,i)
			out_indices.append(idx)
	indices = _merge_tree(out_indices)

	for idc in indices:
		if min(idc)<4:
			wl = [t["lemma"] for t in tokens[:max(idc)+5]]
			idc_rel = [i+4-min(idc) for i in idc]
			out.append((([START]*(4-min(idc))+wl),idc_rel))
		elif max(idc)>N-4:
			wl = [t["lemma"] for t in tokens[min(idc)-4:]]
			idc_rel = [4+i-min(idc) for i in idc]
			out.append(((wl+[END]*(5-N+max(idc))),idc_rel))
		else:
			idc_rel = [i - min(idc)+4 for i in idc]
			wl = [t["lemma"] for t in tokens[min(idc)-4:max(idc)+5]]
			out.append((wl,idc_rel))
	return out


def _group_ner(tokens,idx):
	# given a sentence (list of tokens) and index where the NER starts at, find the group of words that belong to the same ner. return indices
	token = tokens[idx]
	ner = token["ner"]
	indices = [idx]
	childrens = token["children"]
	for child in childrens:
		relation, t = child
		if t["ner"] == ner:
			idx_child = _group_ner(tokens,tokens.index(t))
			indices += idx_child
	return indices

def _merge_tree(idx):
	# given a list of [index], merge them together	
	if not idx:
		return []
	idx.sort(lambda x,y: -1*cmp(len(x),len(y))) # sort according to length:long to short
	largest = [set(idx[0])]
	
	for s in idx[1:]:
		flag = False
		for l in largest:
			if set(s) < l:
				flag = True
		if not flag:
			largest.append(set(s))
	out = [list(s) for s in largest]
	return out


#######################################################################################################
############################## Create k clusters by unigram feature ###################################
#######################################################################################################
def cut_tree(dicts,seqs, linkage, k):
	"""
	Input: dictonary of {id:([sequence],idx)}, list of words sequence, cluster tree, hyperparameter k
	Output: clusters. {id: cluster_id}, {cluster_id: [word_seq]}, {id:word} is easy
	"""
	seq2id = {tuple(v[0]):k for k,v in dicts.items()} # word sequence as tuple to its id
	# word2id = {v[0][v[1]]:k for k,v in dicts.items()} # specific NER to its id
	N = len(dicts)

	cut_tree = hac.cut_tree(linkage,n_clusters = [k])
	flat = cut_tree[:,0]
	# print("Flat array is: "+str(flat))

	clus2seq = {} # map clusters id to sequence
	clus2id = {} # map clusters id to id
	for i in range(N):
		c_id = flat[i]
		seq = tuple(seqs[i])
		# print("for %sth seq, c_id = %s ,seq = %s"%(i,c_id,seq))
		if not clus2id.get(c_id):
			clus2seq[c_id] = [seq]
			clus2id[c_id] = [seq2id[seq]]
			
		else:
			clus2seq[c_id] = clus2seq.get(c_id)+[seq]
			clus2id[c_id] = clus2id.get(c_id)+[seq2id[seq]]

	id2clus = {i:flat[i] for i in range(N)}
	return clus2id, clus2seq, id2clus


def cluster(dicts, seqs, sample = 5, file_name = None, tree = None):
	"""sample different hyperparameters. sample 10 times by default"""
	N = len(dicts)
	print("Trying different #clusters on length %s ..."%(N))
	if not tree.any():
		print("No initialized linkage tree yet!")
		tree = make_cluster_tree(seqs)

	for i in range(sample):
		K = np.random.choice(range(10,int(N*0.1)))
		clus2id, clus2seq, id2clus = cut_tree(dicts,seqs,tree,K)

		clus_length = {c:len(v) for c,v in clus2id.items()}
		max_len = max(clus_length.values())
		max_clusid = clus_length.keys()[clus_length.values().index(max_len)]
		seq = clus2seq[max_clusid]
		print("\n\nk = %s"%(K))
		print("Biggest cluster is #%s"%(max_clusid))
		idx = np.random.choice(max_len, size = 10,replace =False)

		if file_name:
			with open(file_name+str(i)+".txt",'wb') as f:
				for k,v in clus2seq.items():
					f.write(">> Cluster %s:\n "%(k))
					elements = [_concat(t) for t in v]
					f.write(str(elements))
					f.write("\n\n")




#################################################################################################
#########################################   Testing    ##########################################
#################################################################################################
def _concat(wuple_w):
	return (" ").join(list(wuple_w))


def test_extractor():
	# xml = open("/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/2001summary_annotated/1355768.txt.xml").read()
	# text = A(xml)
	# print(text)
	# print(_group_ner(text.sentences[0]["tokens"],2))
	# print(_merge_tree([[0],[1],[2,0,1],[3],[4],[3,4],[7,8,9],[8,9]]))
	# print(_extractor(text.sentences[0],"ORGANIZATION"))
	# exit(0) 

	# files = ['2001summary_annotated/1355765.txt.xml','2001summary_annotated/1355764.txt.xml','2001summary_annotated/1355768.txt.xml']

	dicts = pickle.load(open("filter_results/topic2files(summary).pkl"))
	files = []
	[files.extend(i) for i in dicts.values()]
	files = [sum_root+f for f in files]
	print(len(files))
	# files = files[:10000]
	
	ners = ["PERSON","ORGANIZATION","LOCATION","TIME","DURATION","DATE","SET"]
	fnames = ["NER_input/"+i.lower()+"_all.pkl" for i in ners]
	gen_input(files,save_file=True,
		NER = ners, save_file_name=fnames)
	# gen_input(files,thread=3, save_file = True, save_file_name = ["NER_input/person_sample_wo_punct.pkl","NER_input/org_wo_punct.pkl"])
	# for f in files:
		# print(_gen_input((files[2],["NUMBER","ORGANIZATION"])))


def test_cluster():
	#dicts = pickle.load(open("NER_input/person_sample_wo_punct.pkl"))
	dicts = pickle.load(open("NER_input/person_all.pkl"))
	print(len(dicts))
	# dicts = dict(dicts.items()[:1000000])
	# print(dicts)
	# word2id = {v[0][v[1]]:k for k,v in dicts.items()} # specific NER to its id
	seqs = [i[0] for i in dicts.values()]
	# print(seqs)
	Z = make_cluster_tree(seqs)
	pickle.dump(Z, open("NER_result/linkage/Z_person_all.pkl","wb"))
	# Z = pickle.load(open("NER_result/linkage/Z_person_all_w_punct.pkl"))
	# clus2id, clus2seq, id2clus = cut_tree(dicts, seqs,Z,23)
	# print(clus2id)
	# print(clus2seq)

	cluster(dicts, seqs, tree = Z, file_name="NER_result/person_all")
	


if __name__ == '__main__':
	setup_logging_to_file("loggings/ner.log")

#	test_extractor()
	test_cluster()




