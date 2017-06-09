filter_path = "/home/ml/jliu164/code/filter_detail/"
doc_dir = '/home/ml/jliu164/corpus/nyt_corpus/content_annotated/'
sum_dir = '/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/'
choice = 'content_annotated'

from multiprocessing.dummy import Pool as ThreadPool
from corenlpy import AnnotatedText as A
import os
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
            #print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import pickle
import itertools
from tagger_test import group_files

def filter_length(dicts):
	"""
	Given a dictionary of (topic : [articles]) pair, show the histogram for topic vs summary vs article length
	"""
	
	topics = dicts.keys()
	pool = ThreadPool(6)

	results = pool.map(filter_topic, itertools.izip(dicts.keys(), dicts.values()))
	# results: [(topic, avg doc length, std, avg summary length, std, #long summary)...]
	pool.close()
	pool.join()
	return results

def filter_topic((k,v)):
	"""
	For a (k,v) pair: topic k and a list of files v,
	return (avg length/std of article/summary, #long summary)
	"""
	cache_doc = []
	cache_sum = []
	long_sum = 0
	print("\n>> Processing topic: "+k+" with length "+str(len(v)))

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
	# save what it is returning
	print(k+" done...saving...")
	pickle.dump((k,np.mean(cache_doc),np.std(cache_doc),np.mean(cache_sum),np.std(cache_sum),long_sum),open(filter_path+k+".pkl",'wb'))
	return k, np.mean(cache_doc),np.std(cache_doc),np.mean(cache_sum),np.std(cache_sum),long_sum


def show_graph(topics, doc_mean, doc_std, sum_mean, sum_std):
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
	plt.ylabel('Length (#sentences)')
	plt.title('Avg/Std length of documents vs. summary')
	plt.xticks(index + bar_width / 2, topics, rotation = 'vertical')
	plt.subplots_adjust(bottom = 0.3, top = 0.96)

	figure = plt.gcf() # get current figure
	figure.set_size_inches(n_groups, int(0.75*n_groups))
	
	red_patch = mpatches.Patch(color='r', label='Summary')
	blue_patch = mpatches.Patch(color='b', label='Article')
	plt.legend(handles = [blue_patch,red_patch])

	plt.savefig("Results/Length analysis.jpg")
	print(" Image saved: Length analysis.jpg")
	# plt.show()

def read_results():
	doc_means, doc_stds, sum_means, sum_stds, topics = [],[],[],[],[]
	results = os.listdir(filter_path)
	for result in results:
		path = filter_path + result
		res = pickle.load(open(path))
		topics.append(res[0])
		doc_means.append(res[1])
		doc_stds.append(res[2])
		sum_means.append(res[3])
		sum_stds.append(res[4])

	return topics, doc_means, doc_stds, sum_means, sum_stds


if __name__ == '__main__':
	# inputs = group_files(2002,2004,450,3000)
	# print(inputs.keys())
	
	# results = filter_length(inputs)
	# pickle.dump(results, open("2002-2004result.pkl",'wb'))

	topics, doc_means, doc_stds, sum_means, sum_stds = read_results()
	zipped = zip(topics, doc_means, doc_stds, sum_means, sum_stds)
	with open("Results/Length analysis.txt",'wb') as f:
		f.write("topics, document length mean, document length std, summary length mean, summary length std\n")
		for z in zipped:
			f.write(str(z))
			f.write("\n")
	show_graph(topics, doc_means, doc_stds, sum_means, sum_stds)

	
