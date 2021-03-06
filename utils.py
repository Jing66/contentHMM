import ast
import pickle
import json
from functools import reduce
import os
import numpy as np
import scipy
from sklearn.decomposition import IncrementalPCA as PCA
from sklearn.utils.extmath import randomized_svd as SVD # do full SVD, truncate later
# sys.path.append(os.path.abspath('..'))

vocab_path = "/home/ml/jliu164/code/data/word2idx.json"

SOS = "**START_SENT**"
SOD = "**START_DOC**"
EOS = "**END_SENT**"
SKIP_SET = set([SOD, EOS,SOS])
_PAD = 0
_UNK = 2
_TARGET = 1

def _gen_file_for_M(sentences, vocab,context_sz = 12,filename = "/home/ml/jliu164/code/data/M_tmp1.txt"):
	"""
	Given a list of sentences, convert them into data file for Aishikc's model to run on
	input: list of sentences, each as list of words, dictionary, window_size: left+right
	"""
	if filename:
		f = open(filename,"wb") # wipe out old records
		f.close()
	out = []
	f = open(filename,"a")
	for s in sentences:
		# skip start and end
		sentence = s[1:-1]
		if len(sentence) < 1:
			continue
		if sentence[0] == SOS:
			sentence = sentence[1:]
		n = len(sentence)

		for i in range(n):
			context_left = [vocab.get(word,_UNK) for word in sentence[i-int(context_sz/2):i]]
			if len(context_left) < int(context_sz/2): # needs left padding
				n_left_padding = int(context_sz/2) - i
				context_left = [_PAD]*n_left_padding + [vocab.get(word,_UNK) for word in sentence[:i]]

			context_right = [vocab.get(word,_UNK) for word in sentence[i+1:i+1+int(context_sz/2)]]
			context_right = context_right+[_PAD] * (int(context_sz/2) - len(context_right))
			line = context_left + [_TARGET]+ [vocab.get(sentence[i],_UNK)] + context_right

			assert len(line)==context_sz+2,(line,sentence,i)
			f.write(str(line)+"\r")
			out.append(line)
	return out


			
def _count2(filename = "/home/ml/jliu164/code/data/importance_input/X_train_model.txt"):
	# count the number of unknowns in a list of lists:
	from collections import Counter
	count = 0
	with open(filename) as f:
		line = f.readlines()
	for ll in line:
		l = ast.literal_eval(ll)
		count += dict(Counter(l)).get(2,0)
	print("Out of %s lines, there are %s UNK. UNK percentage:%s"%(len(line),count,float(count)/len(line)))



def make_vocab(topics, name = "word2idx.json"):
	# given list of topics, save a mapping of word to index. replace vocab.txt
	_path = "/home/ml/jliu164/code/contentHMM_input/"
	vocab = set([])
	for topic in topics:
		_,v_cont = pickle.load(open(_path+"contents/"+topic+"/"+topic+"0.pkl","rb"))
		vocab = vocab.union(v_cont)

	mapping = {k:v for k,v in zip(list(vocab), range(3,len(vocab)+3))}
	print("Saving %s word to index mapping"%(len(mapping)))
	if name:
		json.dump(mapping, open(name,"w"))
	




#################################### Stanford NLP Word Embeddings ####################################

def word2vec_file(filename,we_file = "/home/ml/jliu164/code/data/utils/we_file.json"):
	word2vec = {}
	with open(filename,encoding="utf8") as f:
		lines = f.readlines()
		for line in lines:
			word = line.split()[0]
			we = line.split()[1:]
			v = [float(i) for i in we]
			word2vec[word] = we
	print("Saving embeddings for %s words..."%(len(word2vec)))
	with open(we_file,"w") as outfile:
		json.dump(word2vec,outfile,ensure_ascii=False)

### PCA on word embeddings
class MyPCA():
	def __init__(self, X):
		## my own implementation of PCA using numpy
		X = X.T
		C = np.cov(X)
		eig_val_cov, eig_vec_cov = np.linalg.eig(C)
		eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
		eig_pairs.sort(key=lambda x: x[0], reverse=True)
		eig_vec = [p[1] for p in eig_pairs] ## sorted eigen_vec
		self.V = np.stack(eig_vec,axis=1)
		# print("self.V shape",self.V.shape)

		# self.U, self.s, self.V = scipy.linalg.svd(X, full_matrices=False)
		# self.U, self.s, self.V = SVD(X, X.shape[1])

	def transform(self, n_component, x):
		w = x.dot(self.V[:n_component].T)
		return w


def pca_we():
	### perform pca decomposition on GloVe embedding. result in a linear transformation. return the fitted PCA
	we = np.load("../data/utils/we_pca.npy")
	## sklearn PCA -- Too slow???
	# pca = PCA(n_components = n_component)
	# print("fitting PCA...")
	# pca.fit(we)
	# print("PCA fitted!")
	# return pca

	## My implementation
	pca = MyPCA(we)
	# print("PCA decomposed!")
	pickle.dump(pca, open("../data/utils/pca.pkl","wb"))
	return pca


def _freq_we():
	## find the most frequent 5000 words so that pca only needs to be done on those. Save corresponding vectors in "we_freq"
	topics = os.listdir("/home/ml/jliu164/code/contentHMM_input/contents/")
	counts = {}
	for topic in topics:
		try:
			f0 = open("/home/ml/jliu164/code/contentHMM_input/contents/"+topic+"/"+topic+"0.pkl","rb")
			f1 = open("/home/ml/jliu164/code/contentHMM_input/contents/"+topic+"/"+topic+"1.pkl","rb")
			f2 = open("/home/ml/jliu164/code/contentHMM_input/contents/"+topic+"/"+topic+"2.pkl","rb")
		except IOError:
			print("*****"+topic+" Not available!******")
		else:
			fs = [f0, f1,f2]
			for f in fs:
				docs, _ = pickle.load(f)
				for doc in docs:
					for sents in doc:
						for w in sents:
							if w in SKIP_SET:
								continue
							counts[w] = counts.get(w,0)+1
	words = sorted(counts, key=counts.get)

	n_words = 5000
	n_dim = 300
	words = words[-n_words:]# take the top 5000 frequent words for pca
	with open("/home/ml/jliu164/code/data/utils/we_file.json") as f:
		We = json.load(f) # word embedding file. {w: vec}
		unk_vec = We["UNK"]
	we_pca = np.zeros((n_words, n_dim))
	for i in range(n_words):
		v = We.get(words[i],unk_vec)
		we_pca[i] = v
	print(we_pca[:5])
	print(we_pca[-5:])
	np.save("../data/utils/we_pca.npy",we_pca)
	print("%s word vectors saved!"%(n_words))


##################################### Generate X,Y for Importance model #########################################

def generate_X(topics, filename="/home/ml/jliu164/code/data/importance_input/X.txt", ds = 1):
	# generate file for importance score
	with open(vocab_path) as f:
		vocab = json.load(f)
	print("In total %s word to index mapping"%(len(vocab)))
	
	sents_all = []
	for topic in topics:
		try:
			f = open("/home/ml/jliu164/code/contentHMM_input/contents/"+topic+"/"+topic+str(ds)+".pkl","rb")
		except IOError:
			print("*****"+topic+" Not available!******")
			exit(0)
		else:
			docs,_ = pickle.load(f)
			f.close()
		
			sentences = [i for val in docs for i in val]
			sents_all += sentences
	print("There are %s sentences in total"%len(sents_all))
	out = _gen_file_for_M(sents_all,vocab, filename =  filename)
	return out


def generate_Y(topics,filename =  "/home/ml/jliu164/code/data/importance_input/tmpY.txt"):
	sents_all = []
	Y = []
	for topic in topics:
		try:
			f1 = open("/home/ml/jliu164/code/contentHMM_input/contents/"+topic+"/"+topic+"0.pkl","rb")
			f2 = open("/home/ml/jliu164/code/contentHMM_input/summaries/"+topic+"/"+topic+"0.pkl","rb")
		except IOError:
			print("*****"+topic+" Not available!******")
			exit(0)
		else:
			docs1,_ = pickle.load(f1)
			docs2, _= pickle.load(f2)
			f1.close()
			f2.close()
			assert len(docs1) == len(docs2), "Topic %s, Document length %s, summary length %s"%(topic, len(docs1),len(docs2))
			c = 0
			for doc1,doc2 in zip(docs1,docs2):
				try:
					words = reduce(lambda a,b: a.union(b), [set(i) for i in doc2])
				except TypeError:
					print(">>>> Error info: Topic: %s, document:%s, index %s"%(topic,doc2,c))
					exit(0)
				for sent in doc1:
					for i in range(len(sent)):
						if sent[i] in set([SOS,SOD,EOS]):
							continue
						elif sent[i] in words:
							Y.append(1)
						else:
							Y.append(0)
				c+=1
	if filename:
		with open(filename,"w") as f:
			f.write(str(Y))
	return Y



#########################
## Preprocess POS tags
#########################
def word_POS_NER(ds = 1, topic ='War Crimes and Criminals'):
	"""
	For each source article, extract each word's POS, NER and present it using integer.
	Save extracted matricies: data/utils/pos_tags, pos to integer mapping: data/utils/pos2idx.json
	"""
	from corenlpy import AnnotatedText as A
	from nltk.corpus import stopwords
	STOPWORDS = set(stopwords.words('english'))
	save_path = "/home/ml/jliu164/code/filter_results/topic2files(content).pkl"
	fail_path = '/home/ml/jliu164/code/contentHMM_input/fail/'
	corpus_path =  "/home/rldata/jingyun/nyt_corpus/content_annotated/"
	files_ = pickle.load(open(save_path))[topic]
	with open(fail_path+topic+"_Failed.txt") as f:
		paths = f.readlines()
		failed = set([p.split("/")[-1].split(".")[0] for p in paths])
	files = [fs for fs in files_[:-1] if fs.split('/')[-1].split(".")[0] not in failed]
	files = files[int(-np.around(0.1*len(files))):] if ds==2 else files[int(np.around(0.1*len(files))):int(-np.around(0.1*len(files)))]
	print("length of files to process",len(files))

	pos2idx = {} #(pos tag: int)
	lemma2idx = {}	#(lemma: int)
	lemma2pos = {} # (lemma: POS tag)
	ner2idx = {}
	lemma2ner = {}
	count_ner = 0
	count_pos = 0
	count_lemma = 0
	
	for f in files:
		xml = open(corpus_path+f).read()
		annotated_text = A(xml)
		for sentence in annotated_text.sentences:
			
			tokens = sentence['tokens']
			pos_lemma = [(i['pos'],i["lemma"]) for i in tokens if i['word'].isalpha() and i['lemma'].isalpha() and not i['lemma'].lower() in STOPWORDS and i['pos']] # a list of (POS, lemma)
			ner_lemma = [(i['ner'],i["lemma"]) for i in tokens if i['word'].isalpha() and i['lemma'].isalpha() and not i['lemma'].lower() in STOPWORDS and i['ner']] # a list of (NER, lemma)
			for p,l in pos_lemma:
				if not pos2idx.get(p):
					pos2idx[p] = count_pos
					count_pos +=1
					
				if not lemma2idx.get(l):
					lemma2idx[l] = count_lemma
					count_lemma +=1
				
				lemma2pos[l] = p
			
			for n,l in ner_lemma:
				if not pos2idx.get(p):
					ner2idx[n] = count_ner
					count_ner +=1
					
				if not lemma2idx.get(l):
					lemma2idx[l] = count_lemma
					count_lemma +=1
				
				lemma2ner[l] = n
		
	json.dump(pos2idx, open("/home/ml/jliu164/code/data/utils/pos2idx"+str(ds)+".json","wb"))
	json.dump(lemma2idx, open("/home/ml/jliu164/code/data/utils/lemma2idx"+str(ds)+".json","wb"))
	json.dump(lemma2pos, open("/home/ml/jliu164/code/data/utils/lemma2pos"+str(ds)+".json","wb"))
	json.dump(lemma2ner, open("/home/ml/jliu164/code/data/utils/lemma2ner"+str(ds)+".json","wb"))
	json.dump(ner2idx, open("/home/ml/jliu164/code/data/utils/ner2idx"+str(ds)+".json","wb"))

if __name__ == '__main__':
	# word2vec_file("/home/ml/jliu164/code/data/word_embeddings.txt")
	# _count2()

	# topics_vocab = [ "Suicides and Suicide Attempts", "Police Brutality and Misconduct", 
 #       "Sex Crimes", "Drug Abuse and Traffic", "Murders and Attempted Murders", "Hijacking", 
 #      "Assassinations and Attempted Assassinations", 
 #       "War Crimes and Criminals", "Independence Movements and Secession","Tests and Testing"]
	# # make_vocab(topics_vocab)
	# topics_vocab = [ "War Crimes and Criminals"]
	
	# X = generate_X(topics_vocab, filename = "/home/ml/jliu164/code/data/importance_input/X_tmp.txt")
	# Y = generate_Y(topics_vocab, filename = "/home/ml/jliu164/code/data/importance_input/Y0.txt")
	# assert len(X)==len(Y), ("len(X)=%s, len(Y)=%s"%(len(X),len(Y)))

	# _count2()

	# _freq_we()
	# pca = pca_we()
	# v = np.random.rand(3,300)
	# v_ = pca.transform(100,v)
	# print(v_.shape)

	word_POS_NER()
