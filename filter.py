import json
doc_dir = '/home/ml/jliu164/corpus/nyt_corpus/content_annotated/'
sum_dir = '/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/'
choice = 'content_annotated'

from corenlpy import AnnotatedText as A
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

from tagger_test import group_files

def filter_length(dicts):
	"""
	Given a dictionary of (topic : [articles]) pair, show the histogram for topic vs summary vs article length
	"""
	dict_filtered = {} #{ topic: (avg length of articles, std length of articles),(avg length of summary,std length of summary ) }
	doc_mean = []
	doc_std = []
	sum_mean = []
	sum_std = []
	sum_long = []

	for k,v in dicts.items():
		print(">> Processing topic: "+k+" with length "+str(len(v)))

		cache_doc = []
		cache_sum = []
		long_sum = 0

		for path in v:
			# check articles
			doc_path = doc_dir + path+".txt.xml"
			try:
				xml_doc = open(doc_path).read()
			except:
				print("Cannot Open doc: "+doc_path)
				pass
			else:
				annotated_text = A(xml_doc)
				cache_doc.append(len(annotated_text.sentences))

			year = re.findall(r'\d+',path)
			sum_path = sum_dir + year[0]+"summary_annotated/"+year[1]+".txt.xml"
			try:
				xml_sum = open(sum_path).read()
			except:
				print("Cannot Open summary: "+sum_path)
				continue
			else:
				annotated_sum = A(xml_sum)
		    	if len(annotated_sum.sentences) > 4:
		    		long_sum += 1
		    		cache_sum.append(len(annotated_sum.sentences))

		# get values for topic k
		cache_doc = np.array(cache_doc)
		cache_sum = np.array(cache_sum)

		doc_mean.append(np.mean(cache_doc))
		doc_std.append(np.std(cache_doc))
		sum_mean.append(np.mean(cache_sum))
		sum_std.append(np.std(cache_doc))
		sum_long.append(long_sum)

	print(doc_mean)
	print(doc_std)
	print(sum_mean)
	print(sum_std)
	
	return doc_mean, doc_std, sum_mean, sum_std

def show_graph(doc_mean, doc_std, sum_mean, sum_std, topics):
	"""
	Given 4 lists of same length, plot bar chart
	"""
	n_groups = len(topics)
	fig, ax = plt.subplots()
	bar_width = 0.1
	error_config = {'ecolor': '0.3'}
	index = np.arange(n_groups)

	rects1 = plt.bar(index, doc_mean, bar_width,
                 color='b',
                 yerr=doc_std,
                 error_kw=error_config,
                 label='Article')
	rects2 = plt.bar(index + bar_width, sum_mean, bar_width,
                 color='r',
                 yerr=sum_std,
                 error_kw=error_config,
                 label='Summary')

	plt.xlabel('Topics')
	plt.ylabel('Average length')
	plt.title('Avg/Std length of documents vs summary')
	plt.xticks(index + bar_width / 2, topics)
	plt.legend()

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	inputs = group_files(2000,2001,400,2000)
	print(inputs.keys())
	
	result = filter_length(inputs)
	pickle.dump(result, open("2000-2001result.pkl",'wb'))
