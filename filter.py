filter_path = "/home/ml/jliu164/code/filter_detail/"
doc_dir = '/home/ml/jliu164/corpus/nyt_corpus/content_annotated/'
sum_dir = '/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/'
choice = 'content_annotated'
root_dir = '/home/ml/jliu164/corpus/nyt_corpus/'
filter_result_path = 'filter_results/'

from multiprocessing.dummy import Pool as ThreadPool
from corenlpy import AnnotatedText as A
import os
import numpy as np
import matplotlib as mpl
# if os.environ.get('DISPLAY','') == '':
#     mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import pickle
import itertools
import json
from collections import Counter
import pprint
pp = pprint.PrettyPrinter(indent=4)
import operator

from tagger_test import group_files
# from preprocess import extract_tag

def find_summary(yr):
	"""
	given a year, find the summaries that has >=3 sentences. return a list of such files.
	return: ['full_path/XXXXXXX'...]
	"""
	p = sum_dir +str(yr)+"summary_annotated/"
	files = os.listdir(p)
	out = []
	iterat = 0
	for f in files:
		f_path = p+f
		f_id = f.split(".")[0]
		try:
			xml_sum = open(f_path).read()
		except:
			pass
		else:
			annotated_sum = A(xml_sum)
			if len(annotated_sum.sentences) >= 3: 
				out.append(f_id)
				print(" Find one summary: "+f_id)
			if(iterat % 500 == 0):
				print("Scanned 500 files")
			iterat +=1
	print("There are %s summaries" %(len(out)))
	pickle.dump(out, open("filter_results/"+str(yr)+"tmp.pkl",'wb'))

	output = extract_topics(yr, out)
	pickle.dump(output, open(filter_result_path+str(yr)+"_filter_result.pkl",'wb'))
	# return output

def extract_topics(yr,file_id ) :
	"""
	Given a year and a list of files, find the corresponding topics (indexing service). 
	return: (topic, [file_id]) pair
	"""
	# file_path = root_dir+"data/"+str(yr)+"/"
	file_path = "/home/rldata/jingyun/nyt_corpus/data/"+str(yr)+"/"
	out = {}
	with open(file_path+"file_to_topics.json") as json_data:
		out_tmp = []
		d = json.load(json_data)
		for f in file_id:
			topics = d[f][0] # a list of topics given this file
			for topic in topics:
				if not out.get(topic):
					out[topic] = [f]
				else:
					out[topic].append(f)
	return out


def plot_hist(low,high):
	"""
	Plot the histogram for summary length by each year and total
	"""
	results = []
	fig = plt.figure(figsize=(16, 7))
	plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
	for i in range(low,high):

		threas = 300

		try:
			results_tmp = pickle.load(open(filter_result_path+str(i)+"_filter_result.pkl"))
			# results_tmp: {topic:[id]}
			if results_tmp == []:
				print("Nothing available in year "+str(i))
				continue
			
			# local_dict = dict(Counter(results_tmp))
			local_dict = {k:len(v) for k,v in results_tmp.items()}
			# print(local_dict)

			plt.subplot(4,4,i-low+1)
			plt.title('Year '+str(i), fontsize = 8)
			
			label_key, label_range = filter_dict(local_dict,threashold = threas)
			centers = range(len(label_key))
			plt.bar(centers, label_key.values(),align='center', color = 'g')
			# plt.xticks(label_range, label_key.keys(),fontsize = 6,rotation = 'vertical')

			print("In year %s, with threashold %s summaries, there are %s topics" %(i,threas, len(label_key)))

			merge_dict(results, results_tmp)
			# results.extend(results_tmp)
		except IOError:
			print("Haven't processed "+str(i)+"yet!")
			pass
	else:
		plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.03, hspace = 0.5,wspace = 0.15)
		# plt.show()
		plt.savefig("details.png")

		fig = plt.figure(figsize=(16, 7))
		plt.xlabel('Topics')
		plt.ylabel('# summaries (with >=3 sentences)')
		plt.title('Total Topics vs. summaries available')

		# dicts = dict(Counter(results))
		dicts = {k:len(v) for k,v in results.items()}

		threas = 500

		label_key, label_range = filter_dict(dicts,threashold = threas)
		print("\nWith threashold %s summaries, There are in total %s topics: "%(threas, len(label_key)))
		centers = range(len(label_key))
		plt.bar(centers, label_key.values(),align='center', color = 'g')
		
		# print part of the sorted filtered dictionary and sample some summaries
		sorted_x = sorted(label_key.items(), key=operator.itemgetter(1))
		sorted_x.reverse()
		pp.pprint(sorted_x[:10])
		exit(0)
		# sample 3 topics summaries, each sample 10 summaries
		topics = [label_key.keys()[i] for i in np.random.permutation(len(label_key))[:3]]
		samples = sample_summary(topics)

		i = 0
		for sample in samples:
			print("\n\n"+topics[i])
			pp.pprint(sample)
			i+=1

		# plt.xticks(label_range, range(len(label_key.keys())),rotation = 'vertical', fontsize = 6)
		plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.25)
		plt.savefig("total.png")
		# plt.show()

		
def filter_dict(input_dict, threashold = 10):
	"""
	Given a dictionary, filter out the (k:v) pairs where v<=threashold. 
	Return [0] dictionary pairs, [1] the indices of them
	"""
	out_0 = {k: v for k, v in input_dict.iteritems() if v > threashold}
	out_1 = []
	input_items = input_dict.items()
	for i in range(len(input_items)):
		if input_items[i][0] in out_0:
			out_1.append(i)

	return out_0, out_1

def sample_summary(topics, sample_num = 10):
	"""
	Given a list of topics, sample summaries for each topic
	"""
	print(topics)
	samples_all = []
	for topic in topics:
		print("Sampling from topic %s...."%(topic))
		samples = []

		for yr in range(1996,2008):
			
			try:
				ids = sum_topic(yr, topic)
				if len(ids) == 0:
					continue
				indices = np.random.choice(len(ids),size =10 if len(ids)>10 else len(ids),replace = False)
				sample_id = [ids[i] for i in indices]
			except KeyError:
				print(" %s not available in year %s... go to next year" %(topic, yr))
				continue
			else:
				if len(samples) == sample_num:
					break
				else:
					summary = [load_sum(yr, i) for i in sample_id]
					# print(summary)
					samples.extend(summary)

		samples_all.append(samples)
	return samples_all

def filter_results(low, high):
	"""
	Save a summary of {(topic, [ids])} in json
	"""
	results = {}
	for i in range(low, high):
		result_full = {}
		result = pickle.load(filter_result_path+str(i)+"_filter_result.pkl")
		for k,v in result.items():
			result_full[k] = [str(i)+"summary_annotated/"+k+".txt.xml" for k in v]
		merge_dict(results, result_full)
	# print out the first 10 pairs
	print(results.items()[:10])
	pickle.dump(results, open(filter_result_path+"_TOTAL.pkl",'wb'))



def sum_topic( yr, topic = None):
	"""
	Given a topic and a year, return all the summary id that is under given topic and have >=3 sentences
	"""
	data_path = "/home/rldata/jingyun/nyt_corpus/data/"+str(yr)+"/topics_indexing_services.json"
	try:
		with open(data_path) as json_data:
			d = json.load(json_data)
	except:
		print("Json data not available for year "+str(yr))
		return
	else:
		i = np.random.random_integers(0,len(d)-1)
		_topic = d.keys()[i] if topic is None else topic

		# print("topic %s has total %s files for all length" %(_topic, len(d[_topic])))

		a = np.arange(len(d[_topic]))
		np.random.shuffle(a)
		a = a[:10]
		ids = [d[_topic][i] for i in a.tolist() if get_length(yr,d[_topic][i]) >=3]
		return ids



def load_sum(yr, sum_id):
	file_path = root_dir+"summary/"+str(yr)+"summary/"+str(sum_id)+".txt"
	try:
		with open(file_path) as f:
			out = f.read()
			return out
	except IOError:
		print("Path: "+file_path)
		print("year %s, file id %s not available!" %(yr, sum_id))


def get_length(yr, file_id):
	file_path = sum_dir+str(yr)+"summary_annotated/"+str(file_id)+".txt.xml"
	try:
		xml_doc = open(file_path).read()
	except:
		print("Cannot Open doc: "+file_path)
		return -1
	else:
		annotated_text = A(xml_doc)
		
		return len(annotated_text.sentences)


def merge_dict(dict1, dict2):
	# merge two dictionaries, modify the first one
	for k,v in dict2.items():
		if not dict1.get(k):
			dict1[k] = v
		else:
			dict1[k].extend(v)


#################################################################################################
#########################################   Testing    ##########################################
#################################################################################################

def test_chart():
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
	plot_chart(topics, doc_means, doc_stds, sum_means, sum_stds)



if __name__ == '__main__':
	pool = ThreadPool(6)
	results = pool.map(find_summary, range(1996,2002))
	pool.close()
	pool.join()
	# find_summary(2007)


	# in_dir = ["/home/ml/jliu164/corpus/nyt_corpus/data/"+str(i)+"/" for i in range(2003,2008)]
	# pool = ThreadPool(4)
	# results = pool.map(file_to_topic,in_dir)
	# pool.close()
	# pool.join()
    # file_to_topic("/home/ml/jliu164/corpus/nyt_corpus/data/1999/02")

	# plot_hist(1996,2008)
	# samples = sample_summary(["Mental Health and Disorders"])
	# print(len(samples))
	# print(samples[0])

	# topic = 'Copyrights'
	# print('\n')
	# sums = sum_topic(2004)
	# pp = pprint.PrettyPrinter(indent=4)
	# pp.pprint(sums)

	
