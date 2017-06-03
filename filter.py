import json
doc_dir = '/home/ml/jliu164/corpus/nyt_corpus/content_annotated/'
sum_dir = '/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/'
choice = 'content_annotated'

from multiprocessing.dummy import Pool as ThreadPool
from corenlpy import AnnotatedText as A
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import itertools

from tagger_test import group_files

def filter_length(dicts):
	"""
	Given a dictionary of (topic : [articles]) pair, show the histogram for topic vs summary vs article length
	"""
	
	topics = dicts.keys()
	pool = ThreadPool(4)

	results = pool.map(filter_topic, itertools.izip(dicts.keys(), dicts.values()))
	# results: [(topic, avg doc length, std, avg summary length, std, #long summary)]
	return results


def filter_topic((k,v)):
	"""
	For a (k,v) pair: topic k and a list of files v,
	return (avg length/std of article/summary, #long summary)
	"""
	cache_doc = []
	cache_sum = []
	long_sum = 0
	print("\n>> Processing topic: "+k)

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
		
		# check summary
		year = re.findall(r'\d+',path)
		sum_path = sum_dir + year[0]+"summary_annotated/"+year[1]+".txt.xml"
		try:
			xml_sum = open(sum_path).read()
		except:
			print("Cannot Open summary: "+sum_path)
			continue
		else:
			annotated_sum = A(xml_sum)
	    	if len(annotated_sum.sentences) >= 2:
	    		long_sum += 1
	    	cache_sum.append(len(annotated_sum.sentences))
	
	cache_doc = np.array(cache_doc)
	cache_sum = np.array(cache_sum)

	if long_sum == 0:
		print(" No summary has >= 2 sentences! ")
	
	return k, np.mean(cache_doc),np.std(cache_doc),np.mean(cache_sum),np.std(cache_sum),long_sum





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
	inputs = group_files(2000,2002,400,3000)
	print(inputs.keys())
	
	result = filter_length(inputs)
	pickle.dump(result, open("2000-2002result.pkl",'wb'))
