SOS = "**START_SENT**"
EOS = "**END_SENT**"
UNK = "**UNK**"

import pickle

def get_topic_data(topic):
	"""
	Read from topic file, return [1] list of sentences as list of indices,[2] dictionary of word2idx
	NOTE: keep punctuation? Start document? start sentence? Restrict vocab size?
	"""
	sentences = []
	word2idx = {SOS:0, EOS:1,UNK:2}

	i = 3
	docs,_ = pickle.load(open("contentHMM_input/contents/"+topic+"/"+topic+"1.pkl")) # only do train data, skip dev & test
	docs = docs[:3]
	for doc in docs:
		for sentence in doc:
			sentence_idx = []
			for word in sentence:
				if word not in word2idx:
					word2idx[i] = word
					i += 1
			sentence_idx = [word2idx[t] for t in sentence]
			sentences.append(sentence)
	return sentences, word2idx

def test():
	s,d = get_topic_data("Hijacking")
	print(d)
	print(s)


if __name__ == '__main__':
	test()
