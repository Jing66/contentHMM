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
import gc
from nltk.corpus import stopwords
import pprint
from scipy.sparse import lil_matrix
from scipy.spatial.distance import squareform
import os


STOPWORDS = set(stopwords.words('english'))

from content_hmm import similarity

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
				# repre = _extractor(sent,NER[i]) # extract NER exact
				repre = _extractor_linked(sent,NER[i]) # extracting linkage
				if repre:
					results[i].extend(repre)
		
		return results


def _extractor_linked(sent,NER):
	# instead of extract the head word of the NER, find its parent in the dependency tree and return that context
	# for NER == DATE/TIME/DURATION => verb; NER = VERB => NN
	# return ([context words],head word)
	out = []
	tokens = [i for i in sent['tokens'] 
		if i['lemma'].isalpha()  or re.findall(r'^\-?[1-9][0-9]*\.?[0-9]*',i['lemma']) # and i['lemma'] not in STOPWORDS
			or i['ner']]
	N = len(tokens)

	# VERB => NN
	if NER == "VERB":
		for i in range(N):
			w = tokens[i]
			if re.findall(r'VB',w['pos']):
				# print("Find a VB*: %s"%(w))
				t_root = _link_verb(w)
				# print(t_root)
				if not t_root:
					continue
				# get the context around t_root
				idxs = [tokens.index(t) for t in t_root]

				for idx in idxs:
					if N <= 9:
						wl = [i['lemma'] for i in tokens]
					elif idx <4:
						wl = [START]+[i['lemma'] for i in tokens[:idx+4]]
						idx +=1
					elif idx >= N-4:
						wl = [i['lemma'] for i in tokens[idx-4:]]+[END]
						idx = 4
					else:
						wl = [i['lemma'] for i in tokens[idx-4:idx+4]]
						idx = 4
					out.append((tuple(wl),tuple([idx])))
	else: # TIME => VERB
		pass

	return list(set(out))



		

def _extractor(sent, NER):
	# use lemma instead of words
	out = []
	out_indices = [] # list of [index], each element is a group of NER found in sentence
	# tokens = sent['tokens']
	# remove punctuation, keep numbers or those has NER
	tokens = [i for i in sent['tokens'] 
		if i['lemma'].isalpha()  or re.findall(r'^\-?[1-9][0-9]*\.?[0-9]*',i['lemma']) # and i['lemma'] not in STOPWORDS
			or i['ner']]

	N = len(tokens)

	if N <= 9:
		for i in range(N):
			w = tokens[i]
			if w["ner"] == NER:
				idx = _group_ner(tokens,i)
				out_indices.append(idx)
		indices = _merge_tree(out_indices)
		for idc in indices:
			# num_start = 4-min(idc)
			# num_end = max(idc)+5-N
			# idc_rel = [k+num_start for k in idc]
			idc_rel = [k+1 for k in idc]
			word_list = [t["lemma"] for t in tokens]
			# return [([START]*num_start+word_list+[END]*num_end, idc_rel)]
			return [([START]+word_list+[END], idc_rel)]
		else:
			return []

	for i in range(N):
		# print("Checking %s : NER == %s"%(sent["tokens"][i]['word'],sent["tokens"][i]['ner']))
		if tokens[i]["ner"] == NER:
			idx = _group_ner(tokens,i)
			out_indices.append(idx)

	indices = _merge_tree(out_indices)
	# print(indices)
	for idc in indices:
		if min(idc)<4:
			wl = [t["lemma"] for t in tokens[:max(idc)+5]]
			idc_rel = [i+1 for i in idc]
			out.append(([START]+wl,idc_rel))
		elif max(idc)>N-4:
			wl = [t["lemma"] for t in tokens[min(idc)-4:]]
			idc_rel = [4+i-min(idc) for i in idc]
			out.append((wl+[END],idc_rel))
		else:
			idc_rel = [i - min(idc)+4 for i in idc]
			wl = [t["lemma"] for t in tokens[min(idc)-4:max(idc)+5]]
			out.append((wl,idc_rel))
	return out


def _link_verb(token):
	# return a noun token (why not index? punctuations get kicked out) that the verb is referred to
	children = token["children"]
	parents = token["parents"]
	out = []
	# go up
	for _,parent in parents:
		if re.findall(r'VB',parent['pos']):
			return _link_verb(parent)
		if re.findall(r"NN",parent['pos']):
			if not parent['ner'] or re.findall(r'PERSON|LOCATION|ORGANIZATION',parent['ner']):
				return [parent]
			else:
				return _link_verb(parent)
	# go down
	for _, child in children:
		if re.findall(r'NN',child['pos']):
			if not child['ner'] or re.findall(r'PERSON|LOCATION|ORGANIZATION',child['ner']):
				out.append(child)
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
def make_cluster_tree(dicts, force_head = True):
    """
    para text_seq: dictionary of sequences. force_head = True if force those that has same head word to have distance 0
    return linkage
    """
    # generate condensed distance matrix
    #print("Clustering "+str(len(text_seq))+" sentences...")
    text_seq_tup = dicts.values()
    N = len(dicts)
    
    # uni = [set(i) for i in text_seq]
    gc.collect()
    # cond_arr = np.ones(int(comb(N,2)))

    L = lil_matrix((N,N)) # L[i,j] = L[j,i] = similarity(Oi,Oj)
    for i in range(N):
        for j in range(i+1,N):
        	tup_i = text_seq_tup[i]
        	tup_j =  text_seq_tup[j]
        	key_words_i = set([tup_i[0][p] for p in tup_i[1]])
        	key_words_j = set([tup_j[0][p] for p in tup_j[1]])
        	if set(tup_i[0]).intersection(tup_j[0]) == set([]):
        		continue
        	
        	elif key_words_i <=key_words_j and key_words_j <= key_words_i and force_head:
        		# index = cond_arr.size - int(comb(N-i,2)) + (j-i)-1
        		# cond_arr[index] = 0
        		L[i,j] = 1
        		L[j,i] = 1
        	else:
        		sim = similarity(set(tup_i[0]),set(tup_j[0]))
        		# index = cond_arr.size - int(comb(N-i,2)) + (j-i)-1
        		# cond_arr[index] = 1 - similarity(set(tup_i[0]),set(tup_j[0]))
        		L[i,j] = sim
        		L[j,i] = sim

    cond_arr = squareform(L.toarray())
    Z = hac.linkage(1-cond_arr,method = "complete")
    return Z


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
		print("\nk = %s"%(K))
		print("Biggest cluster is #%s"%(max_clusid))
		idx = np.random.choice(max_len, size = 10,replace =False)

		if file_name:
			with open(file_name+str(i)+".txt",'wb') as f:
				pp = pprint.PrettyPrinter(indent = 4, stream = f)
				f.write("Total %s clusters. sizes:" %(K))
				f.write(str(clus_length))
				f.write("\n\n")
				for k,v in clus2seq.items():
					f.write(">> Cluster %s:\n "%(k))
					elements = [_concat(t) for t in v]
					pp.pprint(elements)
					f.write("\n\n")


def _cleanup(dicts):
	v = dicts.values()
	# print(v[0])
	pairs = [(tuple(a),tuple(b)) for a,b in v]
	uniq_pairs = list(set(pairs))
	N = len(uniq_pairs)
	return dict(zip(range(N),uniq_pairs))


def print_info():
	files = os.listdir("NER_input/")
	files = [f for f in files if f.split(".")[-1] == "pkl"]
	for f in files:
		dicts = pickle.load(open("NER_input/"+f))
		print("File %s has size %s!"%(f,len(dicts)))



#################################################################################################
#########################################   Testing    ##########################################
#################################################################################################
def _concat(wuple_w):
	return (" ").join(list(wuple_w))


def test_clean():
	dicts = pickle.load(open("NER_input/duration_all.pkl"))
	print(len(dicts))
	dicts = _cleanup(dicts)
	print(len(dicts))
	pickle.dump(dicts,open("NER_input/duration_all.pkl",'wb'))


def test_extractor():
	# xml = open("/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/2001summary_annotated/1355768.txt.xml").read()
	# xml = open('/home/rldata/jingyun/nyt_corpus/summary_annotated/1996summary_annotated/0886987.txt.xml').read()
	xml = open('/home/rldata/jingyun/nyt_corpus/summary_annotated/1996summary_annotated/0889868.txt.xml').read()
	text = A(xml)
	print(text)
	# print(_group_ner(text.sentences[0]["tokens"],2))
	# print(_extractor(text.sentences[4],"LOCATION"))
	print(_extractor_linked(text.sentences[4],'VERB'))
	# exit(0) 

	# files = ['2001summary_annotated/1355765.txt.xml','2001summary_annotated/1355764.txt.xml','2001summary_annotated/1355768.txt.xml']

	dicts = pickle.load(open("filter_results/topic2files(summary).pkl"))
	files = []
	[files.extend(i) for i in dicts.values()]
	files = [sum_root+f for f in files]
	print(len(files))
	# files = files[:10000]
	
	# ners = ["PERSON","ORGANIZATION","LOCATION"]
	# fnames = ["NER_input/"+i.lower()+"_all_rawer.pkl" for i in ners]
	# gen_input(files,save_file=True,
	# 	NER = ners, save_file_name=fnames)
	# gen_input(files,thread=5, save_file = True, save_file_name = ["NER_input/person_all-1.pkl","NER_input/org_all-1.pkl"])
	# for f in files:
	# print(_gen_input((files[13],["PERSON"])))

	gen_input(files,thread = 5,save_file = True, NER = ["VERB"],save_file_name = ["NER_input/verbs_all.pkl"])



def test_cluster():
	dicts = pickle.load(open("NER_input/org_all.pkl"))
	# dicts = pickle.load(open("NER_input/organization_all.pkl"))
	print(len(dicts))
	# word2id = {v[0][v[1]]:k for k,v in dicts.items()} # specific NER to its id
	# print(seqs)
	Z = make_cluster_tree(dicts)
	pickle.dump(Z, open("NER_result/linkage/Z_organization_all.pkl","wb"))
	# Z = pickle.load(open("NER_result/linkage/Z_person_all.pkl"))
	seqs = [i[0] for i in dicts.values()]

	# clus2id, clus2seq, id2clus = cut_tree(dicts, seqs,Z,5)
	# print(clus2id)
	# print(clus2seq)
	print("Generating clusters...")
	cluster(dicts, seqs, tree = Z, file_name="NER_result/organization_all")
	


if __name__ == '__main__':
	setup_logging_to_file('loggings/ner_'+time.strftime("%d_%m_%Y")+"_"+time.strftime("%I:%M:%S")+".log")

	test_extractor()
	# test_cluster()
	# test_clean()
	# print_info()




