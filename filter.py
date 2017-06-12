filter_path = "/home/ml/jliu164/code/filter_detail/"
doc_dir = '/home/ml/jliu164/corpus/nyt_corpus/content_annotated/'
sum_dir = '/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/'
choice = 'content_annotated'
root_dir = '/home/ml/jliu164/corpus/nyt_corpus/'

from multiprocessing.dummy import Pool as ThreadPool
# from corenlpy import AnnotatedText as A
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

from tagger_test import group_files
# from preprocess import extract_tag

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


def plot_chart(topics, doc_mean, doc_std, sum_mean, sum_std):
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
	pickle.dump(out, open(str(yr)+"tmp.pkl",'wb'))

	# out = pickle.load(open(str(yr)+"tmp.pkl"))
	output = extract_topics(yr, out)
	pickle.dump(output, open("filter_results/"+str(yr)+"_filter_result.pkl",'wb'))
	return output

def extract_topics(yr,file_id ) :
	"""
	Given a year and a list of files, find the corresponding topics (indexing service). 
	return: a list of topic tags at least as long as file_id
	"""
	file_path = root_dir+"data/"+str(yr)+"/"
	out = []
	with open(file_path+"file_to_topics.json") as json_data:
		d = json.load(json_data)
		for f in file_id:
			topics = d[f][0] # a list of topics given this file
			out.extend(topics)
	return out


def file_to_topic(root_path):
	"""
	for all the files under root_path, map file id to its topics.
	Save result: "file_to_topics.json"
	file_id:([Topics_by_indexing_service], [topics_by_online_service])
	"""
	processed = set()
	out = {}
	try:
		with open(root_path+"file_to_topics.json") as json_data:
			d = json.load(json_data)
			processed = set(d.keys())
	except:
		print("No previous work available!")

	for root, dirs, files in os.walk(root_path):
		path = [os.path.join(root,name) for name in files if len(name.split("."))==2 and name.split(".")[1]=="xml"]
		for p in path:
			tag_indexing, tag_online= extract_tag(p)
			file_id = p.split("/")[-1].split(".")[0]
			if file_id in processed:
				continue

			if tag_indexing == set([]):
				tag_indexing.update(["NO TAG"])
			if tag_online == set([]):
				tag_online.update(["NO TAG"])

			# tmp = {file_id:(list(tag_indexing), list(tag_online))}
			out[file_id] = (list(tag_indexing), list(tag_online))

	with open(root_path+'file_to_topics.json','w') as f_json:
		json.dump(out, f_json)
	
	print(">>Done for "+root_path)

def filter_dict(input_dict, threashold = 1):
	"""
	Given a dictionary, filter out the (k:v) pairs where v<=1. 
	Return [0] dictionary pairs, [1] the indices of them
	"""
	out_0 = {k: v for k, v in input_dict.iteritems() if v>1}
	out_1 = []
	input_items = input_dict.items()
	for i in range(len(input_items)):
		if input_items[i][0] in out_0:
			out_1.append(i)

	return out_0, out_1

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


def plot_hist(low,high):
	need_process = []
	results = []
	fig = plt.figure(figsize=(16, 7))
	plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
	for i in range(low,high):
		try:
			results_tmp = pickle.load(open("filter_results/"+str(i)+"_filter_result.pkl"))
			if results_tmp == []:
				print("Nothing available in year "+str(i))
				continue
			
			local_dict = dict(Counter(results_tmp))
			plt.subplot(4,4,i-low+1)
			centers = range(len(local_dict))
			plt.bar(centers, local_dict.values(),align='center', color = 'g')
			# plt.xlabel('Topics', fontsize =8)
			# plt.ylabel('# summaries', fontsize = 8)
			plt.title('Year '+str(i), fontsize = 8)
			
			label_key, label_range = filter_dict(local_dict,1)
			plt.xticks(label_range, label_key.keys(),fontsize = 6,rotation = 'vertical')

			results.extend(results_tmp)
		except IOError:
			need_process.append(i)
			print("Haven't processed "+str(i)+"yet!")
	else:
		plt.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.03, hspace = 0.9,wspace = 0.05)
		# plt.show()
		plt.savefig("details.png")

		# print("Making image...")
		dicts = dict(Counter(results))
		centers = range(len(dicts))

		fig = plt.figure(figsize=(16, 7))
		plt.bar(centers, dicts.values(),align='center', color = 'g')
		plt.xlabel('Topics')
		plt.ylabel('# summaries (with >=3 sentences)')
		plt.title('Total Topics vs. summaries available')

		# figure = plt.gcf() # get current figure
		# figure.set_size_inches(len(dicts)*3, 3*int(0.75*len(dicts)))
		label_key, label_range = filter_dict(dicts,1)
		print(label_key)
		plt.xticks(label_range, label_key.keys(),rotation = 'vertical', fontsize = 6)
		plt.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.25)
		plt.savefig("total.png")
		plt.show()

		
	



if __name__ == '__main__':
	# pool = ThreadPool(6)
	# results = pool.map(find_summary, range(2000,2007))
	# pool.close()
	# pool.join()
	# print(" 1987 - 1999 All Done!")

	# in_dir = ["/home/ml/jliu164/corpus/nyt_corpus/data/"+str(i)+"/" for i in range(2003,2008)]
	# pool = ThreadPool(4)
	# results = pool.map(file_to_topic,in_dir)
	# pool.close()
	# pool.join()
  #	file_to_topic("/home/ml/jliu164/corpus/nyt_corpus/data/1999/02")

	plot_hist(1996,2008)

	
