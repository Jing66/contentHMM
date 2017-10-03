### This script plots histogram for summary length in sentence and words in domain
import matplotlib.pyplot as plt
import numpy as np
import pickle

topics =['War Crimes and Criminals','Sex Crimes',"Drug Abuse and Traffic"]
input_path = "/home/ml/jliu164/code/contentHMM_input/summaries/"


for j,topic in enumerate(topics):
	summary_tokens = []

	for i in range(3):
		summary_token, _ = pickle.load(open(input_path+topic+"/"+topic+str(i)+".pkl","rb"),encoding="latin-1",errors="ignore")
		summary_tokens.extend(summary_token)

	n_sent = np.array([len(s) for s in summary_tokens]) # number of sentence per article
	n_word = np.array([sum([len(i) for i in s]) for s in summary_tokens ])
	
	n_word = n_word - 2*n_sent - 1 # number of words per article
	print(topic,"total # summaries: ",len(n_sent))
	print("mean #sent",np.mean(n_sent),"median #sent",np.median(n_sent), "mean #word",np.mean(n_word),"median #word",np.median(n_word))
	plt.subplot(len(topics), 2, 2*j+1)
	n_sent, bins_sent, patches_sent = plt.hist(n_sent, 10, normed=1, facecolor='green', alpha=0.75)
	if j==len(topics)-1:
		plt.xlabel('# sentences per summary')
	plt.ylabel(r'% of all summaries')
	plt.title(topic)
	# plt.axis()
	plt.grid(True)

	plt.subplot(len(topics), 2, 2*j+2)
	n_words, bins_words, patches_words = plt.hist(n_word, 10, normed=1, facecolor='green', alpha=0.75)
	if j==len(topics)-1:
		plt.xlabel('# words per summary')
	plt.ylabel(r'% of all summaries')
	plt.title(topic)
	# plt.axis()
	plt.grid(True)

plt.show()