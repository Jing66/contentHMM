
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
import pprint
pp = pprint.PrettyPrinter(indent=4)

from content_hmm import *
from multiprocessing.dummy import Pool as ThreadPool

input_dir = '/home/ml/jliu164/code/contentHMM_input/contents/'
extract_dir = "/home/ml/jliu164/code/contentHMM_extract/contents/"
tagger_dir = '/home/ml/jliu164/code/contentHMM_tagger/contents/'
root_dir = "/home/rldata/jingyun/nyt_corpus/"
choice = "content_annotated"

para_path = tagger_dir+'hyper-para.txt'
image_dir = '/home/ml/jliu164/code/contentHMM_tagger/Transition Image/contents/'
fail_path = '/home/ml/jliu164/code/contentHMM_input/fail/'
filter_result_path = 'filter_results/'


########################################################################################################
######################################### Group Input Test  ############################################
########################################################################################################

def preprocess(file_path,fail_topic, for_extract = False):
    """
    insert **START_DOC** before the start of each article; **START/END_SENT** before the start/end of each sentence;
    remove punctuation; replace numbers with 5, #digits are the same
    input a file directory
    return [0]: a document, in the form of a list of sentences
            [1]: a set of all vocabulary
    """
    docs = []
    vocab = set()
    origin = []
    from corenlpy import AnnotatedText as A
    xml = open(file_path).read()
    annotated_text = A(xml)
    for sentence in annotated_text.sentences:
        tokens = sentence['tokens']
        # get original text
        if for_extract:
            out = ""
            for token in tokens:
                out += token['word'] + " "
            origin.append(out)

        # tokens = [numfy(i['lemma']) for i in tokens if i['word'].isalpha() and i['lemma'].isalpha() and not i['lemma'].lower() in STOPWORDS]
        tokens = [i['lemma'] for i in tokens]

        assert len(tokens)>0, (file_path,sentence)
        if len(tokens)>0:
            tokens.insert(0,START_SENT)
            tokens.append(END_SENT) 
            docs.append(tokens)
            vocab = vocab.union(set(tokens))

    if len(docs)>0:    
        docs[0] = [START_DOC]+docs[0]   #['**START_DOC**', '**START_SENT**', u'the', u'head', u'of', u'confirm', u'.', '**END_SENT**']]
        docs[-1].append(EOD)
    if not for_extract:
        return docs, vocab
    else:
        return docs, origin


def numfy(word):
    if not bool(re.search(r'\d', word)):
        return word
    else:
        return re.sub(r'[0-9]','5',word)


def group_files(start,end, low, high):
    # return (topic, files_path) pairs
    in_dir = root_dir+"data/"
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


def save_input(start = 2000,end = 2001,low = 400,high = 600, dicts = None, content = True, redo = False):
    """
    from year start to end, get all the documents and vocabularies stored by topic. dicts: {topic: [file_id_path]}. 
    every file is [[[words]sentence]docs]
    para: content: True if saving content, false is saving summary; redo: True if processing the second time, false if processing the first time
    save dir: /home/ml/code/contentHMM_input/
    NOTE: clear fail and saving directory before processing new dataset
    """
    if not dicts:
        files = group_files(start,end,low,high)
    else:
        files = dicts
    print({k:len(v) for k,v in files.items()})

    # save which input
    if content:
        root_dir = "/home/rldata/jingyun/nyt_corpus/content_annotated/"
        # input_dir ='/home/ml/jliu164/code/contentHMM_input/contents/'
        input_dir = '/home/ml/jliu164/code/Summarization/seq_input/contents/'
        for k,v in files.items():
            files[k] = [re.sub("summary","content",i) for i in v]
    else:
        root_dir = "/home/rldata/jingyun/nyt_corpus/summary_annotated/"
        input_dir = '/home/ml/jliu164/code/Summarization/seq_input/summaries/'
        # input_dir = '/home/ml/jliu164/code/contentHMM_input/summaries/'
        for k,v in files.items():
            files[k] = [re.sub("content","summary",i) for i in v]

    # process each topic
    for topic in files.keys():
        failed = set()
        try:
            with open(fail_path+topic+"_Failed.txt") as f:
                paths = f.readlines()
                failed_l = [p.split("/")[-1].split(".")[0] for p in paths]
                failed = set(failed_l)
                # print(failed)
        except IOError:
            print("No fail record for topic "+topic)
            pass
        print("\n=====================================================")
        print(" Processing topic: "+topic)
        print("=====================================================")
        subdir = input_dir +topic+'/'
        print("Processing directory: "+root_dir)
        print("Saving directory: "+subdir)

        # skip existing folders if for the first time
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        elif os.path.exists(subdir) and not redo:
            continue

        file_path = files[topic]
        
        M = len(file_path) - len(failed)
        print("Will process in total %s files "%(M))

        # [0]: dev set;[ 1]:train set; [2]: test set
        # data_set = file_path[:M],file_path[M:-M],file_path[-M:]
        
        # save dataset
        for i in range(3):
            docs = []
            vocabs = set()
            origins = []
            
            # separate files
            sep = int(np.round(0.1*M))
            print(" Saving data set %s: %s files"%(i,sep))
            if i==0:
                fs = file_path[:sep]
            elif i==1:
                fs = file_path[sep:-sep]
            else:
                fs = file_path[-sep:]

            for f in fs:
                path = root_dir + f

                if path.split("/")[-1].split(".")[0] in failed:
                    continue

                try:
                    if i==2:
                        doc, origin = preprocess(path,topic,for_extract = True)
                    else:
                        doc,vocab = preprocess(path, topic)
                except IOError:
                    print("Cannot Open File "+path)
                    with open(fail_path+topic+"_Failed.txt",'a') as f:
                        f.write(path+"\n")
                else:
                    docs.append(doc)
                    if i==2:
                        # print(origin)
                        origins.append(origin)
                    else:
                        vocabs = vocabs.union(vocab)

            output = open(subdir+topic+str(i)+'.pkl','wb')
            if i==2:
                pickle.dump((docs,origins),output)
            else:
                pickle.dump((docs,vocabs),output)
            print(" All {} articles saved! " .format(len(docs)))
            output.close()
    
        # # make directory
        # if not os.path.exists(extract_dir):
        #     os.makedirs(extract_dir)

        # docs = []
        # vocabs = set()
        # # save dataset for testing
        # print(" Saving testing data set")
        # docs= []
        # vocabs= set()
        # for f in data_set[2]:
        #     path = root_dir+f
        #     if path.split("/")[-1].split(".")[0] in failed:
        #         continue
           
        #     doc,vocab = preprocess(path, topic)
        #     docs.append(doc)
        #     vocabs = vocabs.union(vocab)

        # print(extract_dir+topic+"2.pkl")
        # output = open(extract_dir+topic+str(i)+'.pkl','wb')
        # pickle.dump((docs,vocabs),output)
        # print(" All {} articles saved! " .format(len(docs)))
        # output.close()


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


def train_all(inputs = None):
    """
    Train taggers on all topics and store the tagger in: /home/ml/jliu164/code/contentHMM_tagger/
    """
    if not os.path.exists(tagger_dir):
        os.makedirs(tagger_dir)
    if not inputs:
        inputs = os.listdir(input_dir)
   
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
            print("%s articles available for training " %(len(docs_train)))
        except:
            print("   Training or development data not available!")
            print("looking for: "+dev_path+" and "+train_path)
            continue

        trainer = None
        try:
            trainer = pickle.load(open("contentHMM_tagger/topic_init/"+topic+"_trainer_init.pkl"))
            print(" Initialized trainer available yay!")
        except:
            print("No available Initialized trainer...")
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

def extract_summary(topic,tagger, article_sent, origin, l, summary_train , article_train):
    """
    Given an article(a list of article), produce length-l summary
    return: a list of sentences as summary
    """
    assert len(summary_train)==len(article_train),"There are %s articles but there are %s summaries! " %(len(article_train),len(summary_train))

    # load/save the trained extraction model
    try:
        state_prob = pickle.load(open(extract_dir+topic+".pkl"))
    except IOError:
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
        state_prob = extract_summary_train(tagger, summary_train, article_train)
        pickle.dump(state_prob, open(extract_dir+topic+".pkl",'wb'))
    # print(state_prob)

    _ , flat = tagger.viterbi(article_sent, flat = True)

    flat = np.array(flat)
    # print("Article clustering: "+str(flat))

    indices =  heapq.nlargest(l, range(len(flat)), flat.take)
    # print(indices)

    summary = []
    for i in indices:
        summary.append(origin[i])
    # print(summary)
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
def print_all(path, separate = False):
    taggers = os.listdir(path)
    out = []
    for tagger_path in taggers:
        p= path + tagger_path
        if p.split(".")[-1] != 'pkl':
            continue
        print("\n=====================================================")
        print("   Loading Tagger Info for "+tagger_path)
        print("=====================================================")
        tagger = pickle.load(open(p))
        if not separate:
            print(tagger.info())
            
        else:
            tagger_info_dir = "Results/Tagger Info/"
            if not os.path.exists(tagger_info_dir):
                os.makedirs(tagger_info_dir)
            tagger = pickle.load(open(p))
            name = tagger_path.split(".")[0]
            info_path = tagger_info_dir+name+".txt"
            with open(info_path,"wb") as f:
                f.write(tagger.info())
            print("%s info saved!"%(name))

        trans_image(tagger,tagger_path.split("/")[-1])



# test the permutation test
def test_permutation():
    docs,vocab = pickle.load(open("contentHMM_input/contents/Olympic Games (2000)/Olympic Games (2000)2.pkl"))
    tagger = pickle.load(open('contentHMM_tagger/contents/Olympic Games (2000).pkl'))
    for _ in range(10):
        i = np.random.random_integers(len(docs)-1)
        print("Test on doc # "+str(i))
        print("Test doc has # sentences: "+str(len(docs[i])))
        permutation_test_single(tagger,docs[i],20)



def test_extract_summary(topic, times = 1, l = 3):
    docs, origin = pickle.load(open("contentHMM_input/contents/"+topic+"/"+topic+"2.pkl"))

    tagger = pickle.load(open('contentHMM_tagger/contents/'+topic+".pkl"))
    summaries_train, _ = pickle.load(open("contentHMM_input/summaries/"+topic+"/"+topic+"1.pkl"))
    contents_train, _ = pickle.load(open("contentHMM_input/contents/"+topic+"/"+topic+"1.pkl"))
    valid, valid_origin = pickle.load(open("contentHMM_input/summaries/"+topic+"/"+topic+"2.pkl"))
    
    length = [len(docs[i]) for i in range(len(docs))]
    print(length)
    choices = np.random.permutation(len(docs))[:times]
    for i in choices:
        if len(docs[i]) <= l:
            continue
        summary = extract_summary(topic, tagger,docs[i],origin[i],l, summary_train = summaries_train, article_train = contents_train)
        print("\n\n summary:")
        pp.pprint(summary)
        print("\n origin document:")
        pp.pprint(origin[i])

        print("Actual Summary:")
        pp.pprint(valid_origin[i])



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


def checkLength(topics):
    """
    Sanity check: if the length of test set in contents and summaries are the same.
    """
    for topic in topics:
        content_path = '/home/ml/jliu164/code/contentHMM_input/contents/'+topic+"/"+topic+"1.pkl"
        # content2_path = '/home/ml/jliu164/code/contentHMM_input/contents2/'+topic+"/"+topic+"1.pkl"
        summary_path = '/home/ml/jliu164/code/contentHMM_input/summaries/'+topic+"/"+topic+"1.pkl"

        cont, v_cont = pickle.load(open(content_path))
        # cont2,_ = pickle.load(open(content2_path))
        summary, v_sum = pickle.load(open(summary_path))
        print("Topic %s: training set: Content has %s files,%s vocab, summary has %s files, %s vocab."%(topic,len(cont),len(v_cont),len(summary),len(v_sum)))

        content_path = '/home/ml/jliu164/code/contentHMM_input/contents/'+topic+"/"+topic+"2.pkl"
        # content2_path = '/home/ml/jliu164/code/contentHMM_input/contents2/'+topic+"/"+topic+"2.pkl"
        summary_path = '/home/ml/jliu164/code/contentHMM_input/summaries/'+topic+"/"+topic+"2.pkl"

        cont,o = pickle.load(open(content_path))
        # cont2,_ = pickle.load(open(content2_path))
        summary,_ = pickle.load(open(summary_path))
        print("testing set: Content has %s files,  summary has %s files."%(len(cont),len(summary)))


def summarize_topics(topics_to_train):
    """
    Given a list of topics, find all files of this topic and return a dictionary
    return: (topic,[2000content_annotated/XXXXXX.txt.xml])
    """
    # dicts = pickle.load(open(filter_result_path+"TOTAL.pkl"))
    # dicts_to_train = {k:v for k,v in dicts.items() if k in topics_to_train}
    save_path = "filter_results/topic2files(content).pkl"
    try:
        dicts_to_train_all = pickle.load(open(save_path))
        print("Summary already available!")
        dicts_to_train = {k:v for k,v in dicts_to_train_all.items() if k in topics_to_train}
        return dicts_to_train
    except IOError:
        print("Summary of target topics not available!")
        
        dicts_to_train = {}
        for i in range(1996,2008):
            d = json.load(open("/home/rldata/jingyun/nyt_corpus/data/"+str(i)+"/topics_indexing_services.json"))
            for topic in topics_to_train:
                if not d.get(topic):
                    print("%s not available in year %s" %(topic,i))
                    continue
                tmp = [str(i)+"content_annotated/"+v+".txt.xml" for v in d[topic]]
                if not dicts_to_train.get(topic):
                    dicts_to_train[topic] = tmp
                else:
                    dicts_to_train[topic].extend(tmp)

        # print(dicts_to_train.items()[0])
        pickle.dump(dicts_to_train,open(save_path,"wb"))
        return dicts_to_train


if __name__ == '__main__':    
    # setup_logging_to_file('loggings/tagger_test_'+time.strftime("%d_%m_%Y")+"_"+time.strftime("%I:%M:%S")+".log")

    # test_extract_summary("Police Brutality and Misconduct", times = 7)
    # test_permutation()
    # test_hyper_train()

    # train_all(inputs = ["Police Brutality and Misconduct"])

    # permutation_test(25,10)

    # print_all('/home/ml/jliu164/code/contentHMM_tagger/contents/',separate = True)

    # tagger = pickle.load(open("Olympic Games random.pkl"))
    # tagger.print_info()
    # docs,vocab = pickle.load(open("contentHMM_input/contents/Olympic Games (2000)/Olympic Games (2000)2.pkl"))
    # c,f = tagger.viterbi(docs[3],flat = True)
    # print(dict(Counter(f)))

    topics_to_train = set([ "Suicides and Suicide Attempts", "Police Brutality and Misconduct", 
       "Sex Crimes", "Drug Abuse and Traffic", "Murders and Attempted Murders", "Hijacking", 
      "Assassinations and Attempted Assassinations", 
       "War Crimes and Criminals", "Independence Movements and Secession","Tests and Testing"])
    topics_to_train = set(["War Crimes and Criminals"])
    dicts_to_train = summarize_topics(topics_to_train) 
    save_input(dicts = dicts_to_train, content =False, redo =True)

    # checkLength(topics_to_train)

    
