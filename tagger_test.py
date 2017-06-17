
import pickle
import random
import os
import re
import json
import time
import sys
import traceback
import logging
import exceptions
import itertools

from content_hmm import *
from multiprocessing.dummy import Pool as ThreadPool

input_dir = '/home/ml/jliu164/code/contentHMM_input/contents/'
tagger_dir = '/home/ml/jliu164/code/contentHMM_tagger/contents/'

choice = 'content_annotated/'
root_dir = "/home/rldata/jingyun/nyt_corpus/" + choice
para_path = tagger_dir+'hyper-para.txt'
image_dir = '/home/ml/jliu164/code/contentHMM_tagger/Transition Image/contents2/'
fail_path = '/home/ml/jliu164/code/contentHMM_input/fail/'
filter_result_path = 'filter_results/'


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
    # return (topic, files_path) pairs
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


def save_input(start = 2000,end = 2001,low = 400,high = 600, dicts = None, content = True):
    """
    from year start to end, get all the documents and vocabularies stored by topic. dicts: {topic: [file_id_path]}. content: save contents (true) or summary
    every file is [[[words]sentence]docs]
    save dir: /home/ml/code/contentHMM_input/contents
    NOTE: clear fail and saving directory before processing new dataset
    """
    if not dicts:
        files = group_files(start,end,low,high)
    else:
        files = dicts
    print({k:len(v) for k,v in files.items()})

    # save which input
    if content:
        for k,v in files.items():
            files[k] = [re.sub("summary","content",i) for i in v]
    else:
        for k,v in files.items():
            files[k] = [re.sub("content","summary",i) for i in v]

    # process each topic
    for topic in files.keys():

        failed = set()
        try:
            with open(fail_path+topic+"_Failed.txt") as f:
                failed = set(f.readlines())
        except Exception as e:
            print("No fail record for topic "+topic)
            pass
        print("\n=====================================================")
        print(" Processing topic: "+topic)
        print("=====================================================")
        subdir = input_dir +topic+'/'
        # skip existing folders
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
                path = root_dir + f

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


########################################################################################################
#########################################   Model Training  ############################################
########################################################################################################
def hyper_train(docs_train, vocab_train, docs_dev, topic, trainer,sample_size = 30):
    """
    Given the development set of doc and vocab, return the best set of hyper parameter for the trainer of this topic
    """
    init_path = "contentHMM_tagger/topic_init/"+topic+"_trainer_init.pkl"
    # initialize with random numpy arrays of size sample_size
    delta_1 = np.random.uniform(low = 0.0000001, high = 0.001)
    k = np.random.randint(10,high =50)
    t = np.random.randint(2,high =7)
    delta_2 = np.random.uniform(low = 0.1, high = 10) # (0.1,10)

    if trainer:
        tree = trainer._tree
    else:
        # initialize and save the model
        trainer = ContentTaggerTrainer(docs_train, vocab_train, k, t, delta_1, delta_2)
        tree = trainer._tree
        pickle.dump(trainer, open(init_path,'wb'))

    delta_1 = np.random.uniform(low = 0.0000001, high = 0.001, size = sample_size)
    k = np.random.randint(10,high =50,size = sample_size)
    t = np.random.randint(2,high =7,size = sample_size)
    delta_2 = np.random.uniform(low = 0.1, high = 10, size = sample_size) # (0.1,10)

    # create thread pool
    pool = ThreadPool(5)
    
    results = pool.map(_train, itertools.izip(delta_1,delta_2,k,t,itertools.repeat(tree), itertools.repeat(trainer),range(sample_size),itertools.repeat(docs_dev)))
    # results = pool.map(_train, itertools.izip(delta_1,delta_2,itertools.repeat(k),itertools.repeat(t),itertools.repeat(tree), itertools.repeat(trainer),range(sample_size),itertools.repeat(docs_dev)))
    pool.close()
    pool.join()
    
    pairs = dict(results)
    best_score = max(pairs.keys()) 
    out = pairs[best_score]  
    return out

def _train((delta_1,delta_2,k,t,tree,trainer,j,docs_dev)):
    # return (score,model) of a trained model tested on dev set
    print("++++++++++++++ Sampling #"+str(j)+"+++++++++++++++")
    print(" Training model with hyper parameters: ", delta_1, delta_2 , k, t)
    new_trainer = trainer.adjust_tree(k, tree, t,delta_1,delta_2)
    print(">> Initial Clustering for hyper_train:")
    print(dict(Counter(new_trainer._flat)))
    
    model = new_trainer.train_unsupervised()

    # test on dev set
    alpha = model.forward_algo(docs_dev)
    logprob = logsumexp(alpha[-1])
    print(">>>>>>>>>>>>>log prob on dev set"+str(logprob))
    
    return logprob,model


def train_all():
    """
    Train taggers on all topics and store the tagger in: /home/ml/jliu164/code/contentHMM_tagger/
    """
    if not os.path.exists(tagger_dir):
        os.makedirs(tagger_dir)

    # inputs = os.listdir(input_dir)
    inputs = ['Freedom and Human Rights']
    
    # get all topics data input
    for topic in inputs:
        start_time = time.time()
        # skip the trained models
        if os.path.exists(tagger_dir+topic+".pkl"):
            print( topic +" Model already exist! ")
            continue

        print("=============================================================")
        print(">>   Training the model for topic "+topic)
        print("=============================================================")

        dev_path = input_dir+topic+'/'+topic+'0.pkl'
        train_path = input_dir+topic+'/'+topic+"1.pkl"

        try:
            dev_docs, _ = pickle.load(open(dev_path))
            docs_train, vocab_train = pickle.load(open(train_path))
        except:
            print("   Training or development data not available!")
            continue

        trainer = None
        try:
            trainer = pickle.load(open("contentHMM_tagger/topic_init/"+topic+"_trainer_init.pkl"))
            print(" Initialized trainer available yay!")
        except:
            print(" No available Initialized trainer...")
            pass

        try:
            myTagger = hyper_train(docs_train, vocab_train, dev_docs, topic,trainer)
        except Exception as e:
            print("!Training the model for {} failed! ".format(topic))
            log_exception(e)
            pass
        # save the tagger
        else:
            dur = time.time() - start_time
            hours, rem = divmod(dur, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Model trained in {} hours, {} minutes, {} seconds".format(int(hours),int(minutes),int(seconds)))

            pickle.dump(myTagger, open(tagger_dir+topic+'.pkl','wb'))
            print


########################################################################################################
######################################### Permutation Test  ############################################
########################################################################################################

def permutation_test_single(test_tagger, test_doc, num):
    """
    Given a tagger, test on a document/article
    """
    alpha = test_tagger.forward_algo(test_doc, flat = True)
    logprob = logsumexp(alpha[-1])
    # print("Original logprob: "+str(logprob))
    mistake = 0 
    for i in range(num):
        np.random.shuffle(test_doc)
        alpha = test_tagger.forward_algo(test_doc, flat = True)
        perm_logprob = logsumexp(alpha[-1])

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
    
    
    for topic_file in taggers:
        if topic_file.split(".")[-1] != 'pkl':
            continue
        topic = topic_file.split(".")[0]
        mistake = 0

        print("\n=============================================================")
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

        for _ in range(doc_num):
            i = np.random.random_integers(0,len(test_docs)-1)

            print("Testing with doc {} of length {}...".format(str(i),str(len(test_docs[i]))))
            try:
                mistake += permutation_test_single(myTagger, test_docs[i], test_num)
            except Exception as e:
                print("Cannot test this model!")
                log_exception(e)
                pass
        
        f = open("permutation test result.txt",'a')
        f.write(topic+", mistake #: " + str(mistake)+'\n')
        f.close()
        


def trans_image(tagger,topic):
    """
    Given a tagger, generate its transition probability graph
    """
    import matplotlib as mpl
    if os.environ.get('DISPLAY','') == '':
        #print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    plt.imshow(tagger._trans, cmap = cm.Greys,interpolation = 'nearest')
    plt.title(topic)
    plt.savefig(image_dir + topic+'.jpg')



########################################################################################################
######################################### Extractive Summary  ##########################################
########################################################################################################

def extract_summary_train(tagger,summary_sent, article_sent):
    """
    summary_sent: a list of documents, each is a list of sentences, each sentence is a list of words.
    return: an array of log probability for each topic to appear in summary
    """
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
    
    # print(cache_pair)
    # print(cache_doc)
    state_prob = cache_pair/cache_doc
    
    return state_prob

def extract_summary(tagger, article_sent, l, summary_train , article_train):
    """
    Given an article(a list of article), produce length-l summary
    return: a list of sentences as summary
    """
    if (len(summary_train) != len(article_train)):
        print(" Number of summaries have to match number of documents!")

    state_prob = extract_summary_train(tagger, summary_train, article_train)
    print(state_prob)
    _ , flat = tagger.viterbi(article_sent, flat = True)

    flat = np.array(flat)
    print("Article clustering: "+str(flat))

    indices = np.where(flat == np.nanargmax(state_prob))[0]

    summary = []
    j=0
    for i in indices:
        if j > l:
            break
        summary.append(article_sent[i])
        j +=1
    return summary



#################################################################################################
#########################################   Testing    ##########################################
#################################################################################################

def token_to_sent(tokens):
    """
    convert a list of words into sentence
    """
    return (" ").join(tokens)

def sent_to_article(sents):
    """
    convert a list of sentences into a document
    """
    return (".").join(sents)

# print all the tagger information under path
def print_all(path):
    taggers = os.listdir(path)
    out = []
    for tagger_path in taggers:
        p= path + tagger_path
        if p.split(".")[-1] != 'pkl':
            continue
        print("\n=====================================================")
        print("   Tagger Info for "+tagger_path)
        print("=====================================================")
        tagger = pickle.load(open(p))
        tagger.print_info()
        trans_image(tagger,tagger_path.split("/")[-1])
        out.append(tagger)
    return out


# test the permutation test
def test_permutation():
    docs,vocab = pickle.load(open("contentHMM_input/contents/Olympic Games (2000)/Olympic Games (2000)2.pkl"))
    tagger = pickle.load(open('contentHMM_tagger/contents/Olympic Games (2000).pkl'))
    for _ in range(10):
        i = np.random.random_integers(len(docs)-1)
        print("Test on doc # "+str(i))
        print("Test doc has # sentences: "+str(len(docs[i])))
        permutation_test_single(tagger,docs[i],20)



def test_extract_summary(topic):
    docs,_ = pickle.load(open("contentHMM_input/contents/"+topic+"/"+topic+"2.pkl"))
    tagger = pickle.load(open('contentHMM_tagger/contents_2/'+topic+".pkl"))
    summaries_train, _ = pickle.load(open("contentHMM_input/summaries/"+topic+"/"+topic+"1.pkl"))
    contents_train, _ = pickle.load(open("contentHMM_input/contents/"+topic+"/"+topic+"1.pkl"))
    valid, _ = pickle.load(open("contentHMM_input/summaries/"+topic+"/"+topic+"2.pkl"))
    
    length = [len(docs[i]) for i in range(len(docs))]
    # print(tagger._m)
    print(length)
    summary = extract_summary(tagger,docs[4],5, summary_train = summaries_train, article_train = contents_train)
    
    print(sent_to_article([token_to_sent(i) for i in summary]))

    # print(">>>> Actual Summary:")
    # print(sent_to_article([token_to_sent(i) for i in valid[2]]))



             ###################### Set up logging ###################
def setup_logging_to_file(filename):
    logging.basicConfig( filename=filename,
                         filemode='w',
                         level=logging.DEBUG,
                         format= '%(asctime)s - %(levelname)s - %(message)s',)


def log_exception(e):
    logging.error(
    "Function {function_name} raised {exception_class} ({exception_docstring}): {exception_message}".format(
    function_name = extract_function_name(), #this is optional
    exception_class = e.__class__,
    exception_docstring = e.__doc__,
    exception_message = e.message))

def extract_function_name():
    tb = sys.exc_info()[-1]
    stk = traceback.extract_tb(tb, 1)
    fname = stk[0][3]
    return fname

def test_hyper_train():

    start_time = time.time()

    dev_path = "contentHMM_input/contents/Olympic Games/Olympic Games0.pkl"
    dev_docs,_ = pickle.load(open(dev_path))
    trainer = pickle.load(open("trainer_init.pkl"))    
    train_docs, train_vocab = pickle.load(open("contentHMM_input/contents/Olympic Games/Olympic Games1.pkl"))
    new_tagger = hyper_train(train_docs, train_vocab, dev_docs, "Olympic Games",trainer)

    dur = time.time() - start_time
    hours, rem = divmod(dur, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Model trained in {} hours, {} minutes, {} seconds".format(int(hours),int(minutes),int(seconds)))
    
    pickle.dump(new_tagger, open("Olympic Games.pkl",'wb'))



if __name__ == '__main__':    
    setup_logging_to_file("main.log")

    # test_extract_summary("Awards, Decorations and Honors")
    # test_permutation()
    # test_hyper_train()

    # train_all()

    # permutation_test(25,10)

    # print_all('/home/ml/jliu164/code/contentHMM_tagger/contents/')

    # tagger = pickle.load(open("Olympic Games random.pkl"))
    # tagger.print_info()
    # docs,vocab = pickle.load(open("contentHMM_input/contents/Olympic Games (2000)/Olympic Games (2000)2.pkl"))
    # c,f = tagger.viterbi(docs[3],flat = True)
    # print(dict(Counter(f)))

    topics_to_train = set(["Assualts", "Suicides and Suicide Attempts", "Police Brutality and Misconduct", 
        "Sex Crimes", "Drug Abuse and Traffic", "Murders and Attempted Murders", "Hijacking", 
        "Assassinations and Attempted Assassinations", 
     "War Crimes and Criminals", "Independence Movements and Secession"])

    dicts = pickle.load(open(filter_result_path+"TOTAL.pkl"))
    dicts_to_train = {k:v for k,v in dicts.items() if k in topics_to_train}

    save_input(dicts = dicts_to_train)

    
