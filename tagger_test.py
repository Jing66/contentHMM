from corenlpy import AnnotatedText as A
import pickle
import random
import os
import re
import json

from content_hmm import *

########################################################################################################
######################################### Group Input Test  ############################################
########################################################################################################

def preprocess(file_path):
    """
    insert **START_DOC** before the start of each article; **START/END_SENT** before the start/end of each sentence;
    remove punctuation; replace numbers with 5, #digits are the same
    input a file directory
    return [0]: a document, in the form of a list of sentences
            [1]: a set of all vocabulary
    """
    docs = []
    vocab = set()
    try:
        xml = open(file_path).read()
    except:
        print("Cannot Open File "+file_path)
        return docs,vocab
    annotated_text = A(xml)
    for sentence in annotated_text.sentences:
        tokens = sentence['tokens']
        tokens = [i['lemma'].lower() if i['pos'] not in punctuation else numfy(i['lemma']) for i in tokens]
        vocab = vocab.union(set(tokens))
        tokens.insert(0,START_SENT)
        tokens.append(END_SENT) 
        docs.append(tokens)
    
    docs[0] = [START_DOC]+docs[0]   #['**START_DOC**', '**START_SENT**', u'the', u'head', u'of', u'the', u'united', u'nations', u'election', u'agency', u'say', u'sunday', u'that', u'she', u'would', u'resist', u'a', u'report', u'action', u'to', u'oust', u'she', u'from', u'she', u'position', u',', u'a', u'move', u'that', u'would', u'come', u'a', u'week', u'before', u'crucial', u'election', u'she', u'office', u'be', u'oversee', u'in', u'iraq.secretary', u'general', u'kofi', u'annan', u'plan', u'to', u'deliver', u'a', u'dismissal', u'letter', u'to', u'carina', u'perelli', u',', u'head', u'of', u'the', u'united', u'nations', u"'", u'electoral', u'assistance', u'division', u',', u'the', u'associated', u'press', u'report', u'and', u'two', u'united', u'nations', u'official', u'confirm', u'.', '**END_SENT**']]
    return docs, vocab


def numfy(word):
    if not bool(re.search(r'\d', word)):
        return word
    else:
        return re.sub(r'[0-9]','5',word)


def group_files(start,end, low, high):
    # return (topic, files) pairs
    in_dir = '/home/ml/jliu164/corpus/nyt_corpus/data/'
    dicts = {}
    for i in range(start,end):
        topic_dir = in_dir +str(i)+'/'
        file_dir = str(i)+'content_annotated/'
        # topics = os.walk(topic_dir).next()[2] # ["XXX.json","XXX.json",'XXX.txt']
        topic = "topics_indexing_services.json"  # indexing_service.json
        with open(topic_dir+topic) as data_file:
            data = json.load(data_file)
            keys = data.keys()
            for key in keys:
                files = data[key]
                files = [file_dir+fpt for fpt in files] #[u'200Xcontent_annotated/XXXXXXX',...]
                if(dicts.get(key)):
                    lzt = dicts.get(key)
                    lzt.extend(files)
                    dicts[key] = lzt
                else:
                    dicts[key] = files
    dicts = {k: v for k, v in dicts.items() if len(v) > low and len(v)<high}
    return dicts


def save_input(start,end,low,high):
    """
    from year start to end, get all the documents and vocabularies stored by topic.
    every file is [[[words]sentence]docs]
    /home/ml/code/contentHMM_input
    """
    files = group_files(start,end,low,high)
    root_dir = "/home/ml/jliu164/corpus/nyt_corpus/content_annotated/"
    print({k:len(v) for k,v in files.items()})
    for topic in files.keys():
        print(" Processing topic: "+topic)
        subdir = "/home/ml/jliu164/code/contentHMM_input/"+topic+'/'
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        else:
            continue
        file_path = files[topic]
        M = int(0.1*len(file_path))
        # [0]: dev set;[ 1]:train set; [2]: test set
        data_set = file_path[:M],file_path[M:-M],file_path[-M:]
        
        for i in range(3):
            print(" Saving data set "+str(i))
            docs= []
            vocabs= set()
            for f in data_set[i]:
                path = root_dir + f + ".txt.xml"
                doc,vocab = preprocess(path)
                if doc!=[]:
                    docs.append(doc)
                vocabs = vocabs.union(vocab)
            output = open(subdir+topic+str(i)+'.pkl','wb')
            pickle.dump((docs,vocabs),output)
            print(" All {} articles saved! " .format(len(docs)))
            output.close()

def train_single(dev_path, train_path):
    try:
        docs_dev,vocab_dev = pickle.load(open(dev_path,'rb'))
    except:
        print("Cannot load input develop set: "+dev_path)
        return None
    delta_1, delta_2, k, T = 0.001, 0.2, 40, 4
    
    myTagger = ContentTagger(docs_dev,vocab_dev, k, delta_1, delta_2, T)
    

    print("======= Finding Hyper parameters ======== ")
    delta_1, delta_2, k, T = hyper(myTagger)

    print("====== Training with hyper-parameters ============")
    try:
        docs_train, vocab_train = pickle.load(open(train_path,'rb'))
    except:
        print("Cannot load input training set: "+train_path)
        return None
    myTagger2 = ContentTagger(docs_train,vocab_train,k,delta_1,delta_2,T)
    myTagger2.train_unsupervised()
    
    return myTagger2


def train_all():
    """
    Train taggers on all topics and store the tagger in: /home/ml/jliu164/code/contentHMM_tagger/
    """
    input_dir = '/home/ml/jliu164/code/contentHMM_input/'
    tagger_dir = '/home/ml/jliu164/code/contentHMM_tagger/'
    inputs = os.listdir(input_dir)
    # get all topics data input
    for topic in inputs:
        # skip the trained models
        if os.path.exists(tagger_dir+topic+".pkl"):
            continue
        print("=============================================================")
        print("Training the model for topic "+topic)
        print("=============================================================")
        dev_path = input_dir+topic+'/'+topic+'0.pkl'
        train_path = input_dir+topic+'/'+topic+"1.pkl"
        test_path = input_dir+topic+'/'+topic+'2.pkl'
        myTagger = train_single(dev_path,train_path)
        
        # save the tagger
        if myTagger:
            pickle.dump(myTagger, open(tagger_dir+topic+'.pkl','wb'))


########################################################################################################
######################################### Permutation Test  ############################################
########################################################################################################
def permutation_test(tagger, test_path):
    """
    Given a tagger, test on test set and the permutation of test set
    """
    doc, vocab = pickle.load(open(test_path))
    alpha = tagger.forward_algo(doc)
    logprob = logsumexp(alpha[-1])
    mistake = 0
    for i in range(10):
        # shuffle doc, shuffle sentence in doc
        np.random.shuffle(doc)
        [np.random.shuffle(i) for i in doc]
        perm_logprob = tagger.viterbi(doc)
        if logprob < perm_logprob:
            mistake +=1
    print("Out of 10 permutation, recall is "+str(mistake))

