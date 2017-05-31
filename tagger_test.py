
import pickle
import random
import os
import re
import json
import time
import sys

from content_hmm import *

input_dir = '/home/ml/jliu164/code/contentHMM_input/contents/'
tagger_dir = '/home/ml/jliu164/code/contentHMM_tagger/contents/'
root_dir = "/home/ml/jliu164/corpus/nyt_corpus/summary_annotated/"
choice = 'summary_annotated'
para_path = tagger_dir+'hyper-para.txt'
image_dir = '/home/ml/jliu164/code/contentHMM_tagger/Transition Image/contents/'
fail_path = '/home/ml/jliu164/code/contentHMM_input/fail/'


########################################################################################################
######################################### Group Input Test  ############################################
########################################################################################################

def preprocess(file_path,fail_topic):
    """
    insert **START_DOC** before the start of each article; **START/END_SENT** before the start/end of each sentence;
    remove punctuation; replace numbers with 5, #digits are the same
    input a file directory
    return [0]: a document, in the form of a list of sentences
            [1]: a set of all vocabulary
    """
    from corenlpy import AnnotatedText as A

    docs = []
    vocab = set()
    try:
        xml = open(file_path).read()
    except:
        print("Cannot Open File "+file_path)
        with open(fail_path+fail_topic+"_Failed.txt",'a') as f:
            f.write(file_path+"\n")
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
        file_dir = str(i)+choice
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

        failed = set()
        try:
            with open(fail_path+topic+"_Failed.txt") as f:
                failed = set(f.readlines())
        except:
            pass

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
                if path in failed:
                    continue

                doc,vocab = preprocess(path, topic)
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
    delta_1, delta_2 = 0.001, 1
    emis = range(15)
    tagger = ContentTagger(vocab_dev, emis, None,None, delta_1,delta_2)
    delta_1,delta_2,k,T = tagger.hyper_train(docs_dev,vocab_dev)


    print("=============== Training.... ============ delta 1, delta 2, k, T:")
    print(delta_1,delta_2,k,T)
    para = {topic:(delta_1,delta_2,k,T)}
    with open(para_path,'a') as f:
        f.write(str(para)+'\n')
    try:
        docs_train, vocab_train = pickle.load(open(train_path,'rb'))
    except:
        print("Cannot load input training set: "+train_path)
        return None
    new_tagger = tagger.train(docs_train, vocab_train, k, T, delta_1, delta_2)
    
    return new_tagger


def train_all():
    """
    Train taggers on all topics and store the tagger in: /home/ml/jliu164/code/contentHMM_tagger/
    """
    if not os.path.exists(tagger_dir):
        os.makedirs(tagger_dir)

    inputs = os.listdir(input_dir)
    # get all topics data input
    for topic in inputs:
        start_time = time.time()
        # skip the trained models
        if os.path.exists(tagger_dir+topic+".pkl"):
            print( topic +" Model already exist! ")
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
            with open("error_log.txt",'a') as f:
                f.write(topic+"\n")
                f.write(sys.exc_info()[0]+"\n")
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

def permutation_test_single(test_tagger, test_doc, num):
    """
    Given a tagger, test on a document/article
    """
    # vocabs = [set(i) for i in test_doc]
    # vocab = reduce(lambda a,b:a.union(b),vocabs)
    # vocab.discard(START_SENT)
    # vocab.discard(START_DOC)
    # vocab.discard(END_SENT)
    # tagger.print_info()
    
    alpha = test_tagger.forward_algo(test_doc, flat = True)
    logprob = logsumexp(alpha[-1])
    # print("Original logprob: "+str(logprob))
    mistake = 0 
    for i in range(num):
        # shuffle sentence in doc
        np.random.shuffle(test_doc)
        # test_tagger._clusters, test_tagger.flat = test_tagger.viterbi(test_doc, flat = True)
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

            print("Testing with doc {} of length {}...".format(str(i),str(len(test_docs[i]))))
            try:
                mistake += permutation_test_single(myTagger, test_docs[i], test_num)
            except:
                print("Cannot test this model!")
                pass
        
    f = open("Content permutation test result.txt",'a')
    f.write("For topic "+topic+", mistake rate is" + str(float(mistake)/(test_num*doc_num)))
    f.close()
        


def trans_image(tagger,topic):
    """
    Given a tagger, generate its transition probability graph
    """
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    plt.imshow(tagger._trans, cmap = cm.Greys,interpolation = 'nearest')
    plt.title(topic)
    plt.savefig(image_dir + topic+'.jpg')



########################################################################################################
######################################### Extractive Summary  ##########################################
########################################################################################################

def extract_summary_train(tagger,summary_sent):
    """
    summary_sent: a list of documents, each is a list of sentences, each sentence is a list of words.
    return: an array of log probability for each topic to appear in summary
    """
    article_sent = tagger._docs[:]
    summary_length = [len(i) for i in summary_sent]
    train_length = [len(i) for i in article_sent]
   
    ptr = min(len(summary_sent),len(article_sent))
    summary_sent = summary_sent[:ptr]
    article_sent = article_sent[:ptr]

    summary_length = [len(i) for i in summary_sent]
    train_length = [len(i) for i in article_sent]

    _, summary_flat = tagger.viterbi(summary_sent)
    _, article_flat = tagger.viterbi(article_sent)

    summary_flat = np.array(summary_flat)
    article_flat = np.array(article_flat)

    states = set([i for i in range(np.max(article_flat)) if np.count_nonzero(article_flat == i)>=3])
    
    state_prob = np.zeros(max(states)+1)
    cache_pair = np.zeros(max(states)+1)
    cache_doc = np.zeros(max(states)+1)
    j = 0
    k = 0
    for i in range(len(train_length)):
        summary_part = summary_flat[k:k+summary_length[i]]
        article_part = article_flat[j:j+train_length[i]]
        state_intersec = set(article_part).intersection(set(summary_part)).intersection(states)
        # print(set(article_part))
        for s in state_intersec:
            cache_pair[s] += 1
        for s in set(article_part).intersection(states):
            cache_doc[s] += 1
        k += summary_length[i]
        j += train_length[i]
    
    state_prob = cache_pair/cache_doc
    
    return state_prob

def extract_summary(tagger, article_sent, l, state_prob = None, summary_train = None):
    """
    Given an article, produce length-l summary
    return: a list of sentences as summary
    """
    if state_prob is None and summary_train:
        state_prob = extract_summary_train(tagger, summary_train)
    elif state_prob is None and not summary_train:
        print(" Please either give summaries to train on or a existing topic probability ndarray!")
        exit(1)

    _ , flat = tagger.viterbi(article_sent, flat = True)
    flat = np.array(flat)
    print(flat)
    largest = np.nanargmax(state_prob)
    print(largest)
    indices = np.where(flat == largest)[0]
    out_l = np.sort(indices)
    summary = []
    j=0
    for i in out_l:
        if j > l:
            break
        summary.append(article_sent[i])
        j +=1
    return summary





#################################################################################################
#########################################   Testing    ##########################################
#################################################################################################

if __name__ == '__main__':

    # docs,vocab = pickle.load(open("contentHMM_input/contents/Olympic Games (2000)/Olympic Games (2000)2.pkl"))
    # tagger = pickle.load(open('contentHMM_tagger/contents/Olympic Games (2000).pkl'))
    # summaries, _ = pickle.load(open("contentHMM_input/summaries/Olympic Games (2000)/Olympic Games (2000)1.pkl"))
    # mistake = 0.0
    # for _ in range(10):
    #     i = np.random.random_integers(len(docs)-1)
    #     print("Test on doc # "+str(i))
    #     print("Test doc has # sentences: "+str(len(docs[i])))
    #     mistake = permutation_test_single(tagger,docs[i],20)
    # print("Final: "+str(mistake/(len(docs)*20)))
     
    train_all()

    # permutation_test(15,15)

    # print(dict(Counter(tagger._flat)))
    # length = [len(docs[i]) for i in range(len(docs))]
    # print(length)
    # summary = extract_summary(tagger,docs[1],3, summary_train = summaries)
    # print(summary)


   # dev_path = "contentHMM_input/contents/Olympic Games (2000)/Olympic Games (2000)0.pkl"
   # train_path = "contentHMM_input/contents/Olympic Games (2000)/Olympic Games (2000)1.pkl"
   # tag = train_single(dev_path,train_path,"WednesdayAfternoon3")
   # delta1, delta2, k, T = 5.7774778572996404e-05, 6.155334807363642, 38, 4
   # # docs_train, vocab_train = pickle.load(open(train_path,'rb'))
   
   # pickle.dump(tag,open("Olympic Games random.pkl",'wb'))
