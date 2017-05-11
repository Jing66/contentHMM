import xml.etree.ElementTree as ET
import os
import sqlite3

def extract(in_path):

	tree = ET.parse(in_path)
	
	summary=""
	body = tree.getroot().find("body")
	for head in body.findall('body.head/abstract/p'):
		if isinstance(head.text,unicode):
			summary = ""
			break
		summary+=head.text+"\n"
	
	
	article=""
	body = tree.getroot().find("body")
	contents = body.find('body.content')

	for content in contents.iter('block'):
		para = content.get('class')
		if para!="full_text":
			continue
		for txt in content.findall('p'):
			article+=txt.text+"\n"
	
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
		path = [os.path.join(root,name) for name in files if name.split(".")[1]=="xml"]
		for p in path:
			try:
				extraction = extract(p)
			except Exception:
				print("Parsing error: "+p)
				continue
			name = p.split("/")[-1].split(".")[0]

			# skip if any element is missing
			if extraction[0]=="" or extraction[1]=="" or extraction[2]=="":			
				continue
			for i in range(3):
				f = open(process_path[i]+"/"+name+".txt","wb")
				try:
					f.write(extraction[i])
				except UnicodeEncodeError:
					f.write(extraction[i].encode("utf16"))
				f.close()


def annotate_all(in_dir, out_dirs):
	import corenlpy
	if not os.path.exists(out_dirs):
		os.makedirs(out_dirs)

	corenlpy.corenlp(in_dirs=in_dir, out_dir=out_dirs,
		annotators=['tokenize','ssplit','pos','lemma','parse','ner','relation','dcoref','natlog'],
		threads=4,output_format='xml')
	# Can't tokenize unicode files -- FIXME

# store all file paths under root into db
def store(root_dir):
	conn = sqlite3.connect("nyt.db")
	c = conn.cursor()
	c.execute("""CREATE TABLE IF NOT EXISTS nyt(id integer PRIMARY KEY, orinigal_path text NOT NULL);""")
	
	# entry:[(id,path)]
	entry = []
	for root, dirs, files in os.walk(root_dir):
		# get all .xml file path under year/XX/XX
		path = [os.path.join(root,name) for name in files if name.split(".")[1]=="xml"]
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
	query = "SELECT * FROM nyt WHERE id=="+str(file_id)
	c.execute(query)
	out = c.fetchone()
	conn.close()
	return out[1]


# testing
def test():
	# test extract
	in_path ="/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2001/05/05/1290809.xml"
	summary, content, meta = extract(in_path)
	print(summary=="")
	# f = open("test1","wb")
	# try:
	# 	f.write(content)
	# except UnicodeEncodeError:
	# 	print("Encode in Unicode!")
	# 	f.write(content.encode("utf16"))
	# f.close()

	#test extractall
	in_dir = '/Users/liujingyun/Desktop/NLP/nyt_corpus/data/'
	#extractall(in_dir)
	
	#test annotate_all
	in_dir = '/Users/liujingyun/Desktop/NLP/nyt_corpus/data/summary'
	out_dir = '/Users/liujingyun/Desktop/NLP/nyt_corpus/data/summary_annotated'
	#annotate_all(in_dir, out_dir)


if __name__ == '__main__':
	test()

