import ast
import pickle
import json
# sys.path.append(os.path.abspath('..'))

def vocab_for_M(filename = "/home/ml/jliu164/code/data/vocab.txt"):
	# loads the word2idx for Aishikc's model 
	with open(filename, "r") as cache:
		line = cache.readlines()[0]
	data = ast.literal_eval(line)
	return data

def gen_file_for_M(sentences, vocab,context_sz = 12,filename = "/home/ml/jliu164/code/data/M_tmp.txt"):
	"""
	Given a list of sentences, convert them into data file for Aishikc's model to run on
	input: list of sentences, each as list of words, dictionary, window_size: left+right
	output: 
	"""
	_PAD = 0
	_UNK = 2
	_TARGET = 1

	SOS = "**START_SENT**"
	SOD = "**START_DOC**"
	EOS = "**END_SENT**"

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

			
def _count2(filename = "/home/ml/jliu164/code/data/M_tmp.txt"):
	# count the number of unknowns in a list of lists:
	from collections import Counter
	count = 0
	with open(filename) as f:
		line = f.readlines()

	for ll in line:
		l = ast.literal_eval(ll)
		count += dict(Counter(l)).get(2,0)
	print(count)


##################################### Stanford NLP Word Embeddings ###########################################

def word2vec_file(filename,we_file = "/home/ml/jliu164/code/data/we_file.json"):
	word2vec = {}
	with open(filename) as f:
		lines = f.readlines()
		for line in lines[:10]:
			word = line.split()[0]
			we = line.split()[1:]
			v = [float(i) for i in we]
			word2vec[word] = we
	print("Saving embeddings for %s words..."%(len(word2vec)))
	with open(we_file,"w") as outfile:
		json.dump(word2vec,outfile,ensure_ascii=False)


########################################## Testing #################################################

def test_generate_file():
	vocab = vocab_for_M()
	print(len(vocab))
	
	f = open("/home/ml/jliu164/code/contentHMM_input/contents/War Crimes and Criminals/War Crimes and Criminals1.pkl","rb")
	docs,_ = pickle.load(f)
	f.close()
	
	sentences = [i for val in docs for i in val]
	print("There are %s sentences"%len(sentences))

	out = gen_file_for_M(sentences,vocab)



if __name__ == '__main__':
	# test_generate_file()
	# word2vec_file("/home/ml/jliu164/code/data/word_embeddings.txt")
	# _count2()



