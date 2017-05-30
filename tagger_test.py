from corenlpy import AnnotatedText as A
import pickle
import random
import os
import re
import json
import time

from content_hmm import *

input_dir = '/home/ml/jliu164/code/contentHMM_input/summaries/'
tagger_dir = '/home/ml/jliu164/code/contentHMM_tagger/summaries/'
root_dir = "/home/ml/jliu164/corpus/nyt_corpus/content_annotated/"
para_path = tagger_dir+'hyper-para.txt'
image_dir = '/home/ml/jliu164/code/contentHMM_tagger/Transition Image/summaries/'
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
        tokens = [numfy(i['lemma']) for i in tokens if i['word'].isalpha() and i['lemma'].isalpha() and not i['lemma'].lower() in STOPWORDS]
       
        vocab = vocab.union(set(tokens))
        tokens.insert(0,START_SENT)
        tokens.append(END_SENT) 
        docs.append(tokens)

    if len(docs)>0:    
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
    print({k:len(v) for k,v in files.items()})
    for topic in files.keys():
        print(" Processing topic: "+topic)
        subdir = input_dir +topic+'/'
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

def train_single(dev_path, train_path,topic):
    try:
        docs_dev,vocab_dev = pickle.load(open(dev_path,'rb'))
    except:
        print("Cannot load input develop set: "+dev_path)
        return None
    delta_1, delta_2, k, T = 0.001, 0.2, 40, 4
    myTagger = ContentTagger(docs_dev,vocab_dev, delta_1, delta_2, T = T,k = k)
    delta_1, delta_2, k, T = hyper(myTagger)

    print("====== Training.... ============ delta 1, delta 2, k, T:")
    print(delta_1,delta_2,k,T)
    para = {topic:(delta_1,delta_2,k,T)}
    with open(para_path,'a') as f:
        f.write(str(para)+'\n')
    try:
        docs_train, vocab_train = pickle.load(open(train_path,'rb'))
    except:
        print("Cannot load input training set: "+train_path)
        return None
    myTagger2 = ContentTagger(docs_train,vocab_train,delta_1,delta_2,T = T, k = k)
    myTagger2.train_unsupervised()
    
    return myTagger2


def train_all():
    """
    Train taggers on all topics and store the tagger in: /home/ml/jliu164/code/contentHMM_tagger/
    """
    inputs = os.listdir(input_dir)
    # get all topics data input
    for topic in inputs:
        start_time = time.time()
        # skip the trained models
        if os.path.exists(tagger_dir+topic+".pkl"):
            continue
        print("=============================================================")
        print("Training the model for topic "+topic)
        print("=============================================================")
        dev_path = input_dir+topic+'/'+topic+'0.pkl'
        train_path = input_dir+topic+'/'+topic+"1.pkl"
        test_path = input_dir+topic+'/'+topic+'2.pkl'

        try:
            myTagger = train_single(dev_path,train_path,topic)
        except:
	    print("!!!!!!!Training the model for {} failed! ".format(topic))
            continue
        # save the tagger
        if myTagger:
            pickle.dump(myTagger, open(tagger_dir+topic+'.pkl','wb'))
            dur = time.time() - start_time
            hours, rem = divmod(dur, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Model trained in {} hours, {} minutes, {} seconds".format(int(hours),int(minutes),int(seconds)))
            print


########################################################################################################
######################################### Permutation Test  ############################################
########################################################################################################
def permutation_test_single(tagger, test_doc, num):
    """
    Given a tagger, test on a document/article
    """
    vocabs = [set(i) for i in test_doc]
    vocab = reduce(lambda a,b:a.union(b),vocabs)
    vocab.discard(START_SENT)
    vocab.discard(START_DOC)
    vocab.discard(END_SENT)
    
    test_tagger = ContentTagger([test_doc],vocab, tagger._delta_1, tagger._delta_2, emis = tagger._emis, trans = tagger._trans, prior = tagger._priors)
    # pickle.dump(test_tagger,open("test.pkl",'wb'))
    
    test_tagger._clusters, test_tagger._flat = test_tagger.viterbi()
    # print(test_tagger._flat)
    alpha = test_tagger.forward_algo()
    logprob = logsumexp(alpha[-1])
    # print("Original logprob: "+str(logprob))
    mistake = 0 
    for i in range(num):
        # shuffle sentence in doc
        np.random.shuffle(test_doc)
        test_tagger._clusters, test_tagger.flat = test_tagger.viterbi(test_doc, flat = True)

        alpha_shuffle= test_tagger.forward_algo(test_doc)
        perm_logprob = logsumexp(alpha_shuffle[-1])
        # print("After shuffle the log prob is:"+str(perm_logprob))
        if logprob < perm_logprob:
            mistake +=1
    print("Out of {} permutation, #better permutation: {} ".format(str(num),str(mistake)))
    return mistake


def permutation_test(doc_num, test_num):
    """
    perform permutation test. 
    doc_num: the number of documents to be sampled to be tested on.
    test_num: the number of times each document is shuffled to be test
    """
    inputs = os.listdir(input_dir)
    taggers = os.listdir(tagger_dir)
    mistake = 0
    for topic in inputs:

        print("=============================================================")
        print("Testing the model for topic "+topic)
        print("=============================================================")
        test_path = input_dir+topic+'/'+topic+'2.pkl'
        try:
            test_docs,vocabs = pickle.load(open(test_path))
        except:
            print("Test files not available!")
            continue

        try:
            myTagger = pickle.load(open(tagger_dir+topic+'.pkl'))
        except:
            print("Model isn't available!")
            continue

        trans_image(myTagger,topic)

        for _ in range(doc_num):
            i = np.random.random_integers(len(test_docs)-1)
            mistake += permutation_test_single(myTagger, test_docs[i], test_num)
        
    f = open("Content permutation test result.txt",'a')
    f.write("For topic "+topic+", mistake rate is" + str(float(mistake)/(test_num*doc_num)))
    f.close()
        


def trans_image(tagger,topic):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    plt.imshow(tagger._trans, cmap = cm.Greys,interpolation = 'nearest')
    plt.title(topic)
    plt.savefig(image_dir + topic+'.jpg')



if __name__ == '__main__':
    # docs,vocab = pickle.load(open("contentHMM_input/contents/News and News Media/News and News Media2.pkl"))
    # tagger = pickle.load(open('test tagger.pkl'))
    # mistake = 0.0
    # for _ in range(10):
    #     i = np.random.random_integers(len(docs)-1)
    #     print("Test on doc # "+str(i))
    #     print("Test doc has # sentences: "+str(len(docs[i])))
    #     mistake = permutation_test_single(tagger,docs[i],20)
    # print("Final: "+str(mistake/(len(docs)*20)))
     
    train_all()

   # permutation_test(10,10)

   # dev_path = 'contentHMM_input/summaries/News and News Media/News and News Media0.pkl'
   # train_path = 'contentHMM_input/summaries/News and News Media/News and News Media1.pkl'

   # tag = train_single(dev_path,train_path,"MondayAfternoon3")
   # pickle.dump(tag,open("News random.pkl",'wb'))
