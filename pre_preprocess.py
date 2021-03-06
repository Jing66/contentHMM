import xml.etree.ElementTree as ET
import os
import sqlite3
import thread
import unicodedata
import threading
import json
import re

from multiprocessing.dummy import Pool as ThreadPool

########################################################################################################
############################### extract topics from raw files 	##############################
########################################################################################################
def extract_tag(in_path):
	"""
	return a set of tags the raw file has: (indexing_service, online_producer)
	"""
	tag_indexing = []
	tag_online = []
	tree = ET.parse(in_path)
	head = tree.getroot().find("head")
	for descr in head.findall('docdata/identified-content/classifier[@type="descriptor"]'):
		if descr.get('class') == 'indexing_service':
			tag_indexing.append(descr.text)
		if descr.get('class') == 'online_producer':
			tag_online.append(descr.text)
	return set(tag_indexing), set(tag_online)

def extract_tag_all(root_dir):
	"""
	save an index of all the tags and files
	"""
	index = {}
	online_prod = {}
	for root, dirs, files in os.walk(root_dir):
		path = [os.path.join(root,name) for name in files if len(name.split("."))==2 and name.split(".")[1]=="xml"]
		for p in path:
			try:
				tag_indexing, tag_online= extract_tag(p)
			except:
				tag_indexing, tag_online = set([]),set([])
				print("Both empty: "+p)
			#name = ("/").join(p.split("/")[-3:]).split(".")[0]
			name = p.split("/")[-1].split(".")[0]
			if tag_indexing == set([]):
				tag_indexing.update(["NO TAG"])
			if tag_online == set([]):
				tag_online.update(["NO TAG"])

			for tag in tag_indexing:
				item_index = index.get(tag)
				if item_index is None:
					index[tag] = [name]
				else:
					item_index.append(name)
					index.update([(tag,item_index)])
			for tag in tag_online:
				item_index = online_prod.get(tag)
				if item_index is None:
					online_prod[tag] = [name]
				else:
					item_index.append(name)
					online_prod.update([(tag,item_index)])
	f = open(root_dir+'topics_indexing_services.json','wb')
	json.dump(index,f)
	f.close()
	f = open(root_dir + 'topics_online_producer.json','wb')
	json.dump(online_prod,f)
	f.close()


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


########################################################################################################
################## extract summary, full-article, headlines from raw files 	#########################
########################################################################################################

def extract(in_path):
	tree = ET.parse(in_path)
	summary=""
	body = tree.getroot().find("body")
	for head in body.findall('body.head/abstract/p'):
		summary += head.text	
	article=""
	body = tree.getroot().find("body")
	contents = body.find('body.content')

	for content in contents.iter('block'):
		para = content.get('class')
		if para!="full_text":
			continue
		for txt in content.findall('p'):
			article+=txt.text
	
	h1 = tree.getroot().find("body/body.head/hedline/hl1")
	meta = h1.text
	
	return summary,article, meta

# extract and parse plain text into root/summary, root/meta, root/content
def extractall(root_dir):
	paths = [] # list of all files under root
	new_storage = ["summary","content","meta"]
	process_path = [root_dir+ns for ns in new_storage]
	
	# make folders under root if they don't already exist
	for ps in process_path:
		if not os.path.exists(ps):
			os.makedirs(ps)

	for root, dirs, files in os.walk(root_dir):
		# get all .xml file path under year/XX/XX
		path = [os.path.join(root,name) for name in files if len(name.split("."))==2 and name.split(".")[1]=="xml"]
		for p in path:
			try:
				extraction = extract(p)
			except Exception:
				print("Parsing error: "+p)
				continue
			name = p.split("/")[-1].split(".")[0]

			# skip if contain non-ascii or some parts missing
			if extraction[0]=="" or extraction[1]=="" or extraction[2]=="":
				print("missing part on "+name)			
				continue
			for i in range(3):
				f = open(process_path[i]+"/"+name+".txt","wb")
				try:
					f.write(extraction[i])
				except UnicodeEncodeError:
					print("Unicode encoding: "+p)
					f.write(extraction[i].encode("utf8"))
				f.close()



class myThreads(threading.Thread):
	def __init__(self,in_dir,number):
		threading.Thread.__init__(self)
		self._dir = in_dir
		self._number = number
	def run(self):
		print("Starting: "+str(self._number))
		extract_tag_all(self._dir)
		print("Existing: "+str(self._number))


def extract_multi(in_dir,start,finish):
	threads = []
	for i in range(start,finish+1):
		path= in_dir+str(i)+"/"
		thread = myThreads(path,i)
		threads.append(thread)

	for thread in threads:
		thread.start()

	for t in threads:
		t.join()
	print(str(start)+"-"+str(finish)+" Done")



########################################################################################################
################################ preprocess extracted files ###################################
########################################################################################################
def annotate(content, in_dir,start,end):
    for i in range(start,end):
        in_path = in_dir+content+'/'+str(i)+content
        out_path = in_dir+content+"_annotated/"+str(i)+content+"_annotated"
        annotate_all(in_path,out_path)

        print("Annotating "+content+"for "+str(i)+"Done!")

def annotate_all(in_dir, out_dirs):
	import corenlpy
	if not os.path.exists(out_dirs):
		os.makedirs(out_dirs)

	corenlpy.corenlp(in_dirs=in_dir, out_dir=out_dirs,
		annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse','dcoref','relation'],
		threads=10,output_format='xml')
	



########################################################################################################
############################### store paths of all files     ################################
########################################################################################################
# store all file paths under root into db
def store(root_dir):
	conn = sqlite3.connect("nyt.db")
	c = conn.cursor()
	c.execute("""CREATE TABLE IF NOT EXISTS nyt(file_id integer, orinigal_path text NOT NULL);""")
	
	# entry:[(id,path)]
	entry = []
	for root, dirs, files in os.walk(root_dir):
		# get all .xml file path under year/XX/XX
		path = [os.path.join(root,name) for name in files if len(name.split("."))==2 and name.split(".")[1]=="xml"]
		for p in path:
			name = p.split("/")[-1].split(".")[0]
			entry.append((name,p))

	c.executemany("INSERT INTO nyt VALUES (?,?)",entry)

	conn.commit()
	conn.close()

# given file id, return path to its original raw xml file
def get_path(file_id):
	conn = sqlite3.connect("nyt.db")
	c = conn.cursor()
	query = "SELECT * FROM nyt WHERE file_id=="+str(file_id)
	c.execute(query)
	out = c.fetchone()
	conn.close()
	return out[1]


#################################################################################################
################################ preprocess summary files ###################################
#################################################################################################
def substi(text):
	"""
	Find all ; and replace with '.' get rid of (M), photo(s)/drawing(s)
	"""

	text = re.sub(r'photo|drawing|map|chart|table|graph|diagram',"", text)
	text = re.sub(r'\([A-Z]\)','',text)

	text_sep = text.split(".")
	text_sep = [s.strip() for s in text_sep if len(s)>=3]
	text_new = ". ".join(text_sep)


	return text_new

def substi_all(root_dir):
	"""
	For all files under root_dir, preprocess and write back
	"""
	l = os.listdir(root_dir)
	for p in l:
		_dir= root_dir+p
		try:
			with open(_dir) as f:
				text = f.read()
				text = substi(text)
		except:
			print("Cannot modify "+ _dir)
			continue
		try:
			with open(_dir,'wb') as f:
				f.write(text)
		except:
			print("Cannot write into "+ _dir)
			continue

	print(root_dir+" Done!")




#######################################################################################################
################################################# Testing	###########################################
#######################################################################################################

# testing
def test_old():
	# test extract
	in_path ="/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2007/02/04/1823747.xml"
	
	summary, content, meta = extract(in_path)
	# print(len(summary),len(content),len(meta))
	# f = open("test1","wb")
	# try:
	# 	f.write(content)
	# except UnicodeEncodeError:
	# 	print("Encode in Unicode!")
	# 	f.write(content.encode("utf8"))
	# f.close()

	#test extractall
	# in_dir =s
	in_dir = '/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2006'
	in_dir_200701 = '/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2007/01'
	in_dir_200702 = '/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2007/02'
	tag_path = "/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2006/01/02/1729112.xml"
	#extractall(in_dir_200701)
	# thread1 = myThreads(in_dir_200701,"0701")
	# thread2 = myThreads(in_dir_200702,"0702")
	# thread1.start()
	# thread2.start()
	#extract_multi(in_dir,2006,2007)
	extract_tag_all(in_dir)


	
	#test annotate_all
	in_dir = '/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2007content'
	out_dir = '/Users/liujingyun/Desktop/NLP/nyt_corpus/summary_annotated'
	#annotate_all(in_dir, out_dir)

	for i in range(1987,2008):
		in_path = '/Users/liujingyun/Desktop/NLP/nyt_corpus/data/'+str(i)+'content'
		out_path = '/Users/liujingyun/Desktop/NLP/nyt_corpus/content_annotated/'+str(i)
		#annotate_all(in_path, out_path)


def test_substi():
	# _dir = "/home/ml/jliu164/corpus/nyt_corpus/summary/2007summary/"
	# substi_all(_dir)

	pool = ThreadPool(6)
	root_dir = '/home/ml/jliu164/corpus/nyt_corpus/summary/'
	dirs = [root_dir+str(i)+"summary/" for i in range(2002,2007)]
	results = pool.map(substi_all, dirs)
	pool.close()
	pool.join()



if __name__ == '__main__':
	# test()
	test_substi()
