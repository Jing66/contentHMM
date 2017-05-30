
from nltk import bigrams
import math
import numpy as np

import scipy.cluster.hierarchy as hac
from collections import Counter
from scipy.misc import comb
from scipy.sparse import dok_matrix
#import matplotlib.pyplot as plt

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
START_SENT = "**START_SENT**"
START_DOC = "**START_DOC**"
END_SENT = "**END_SENT**"
UNK = "**UNK**"

########################################################################################################
################################# Create k clusters to initialize ######################################
########################################################################################################
# bigram:
def similarity(bigrams1,bigrams2):
    """
    input text bigrams as list
    """
    numer = len(bigrams1.intersection(bigrams2))
    denom = math.sqrt(len(bigrams1)*len(bigrams2))
    if not denom:
        return 0.0
    else:
        return 1 - float(numer)/denom

def make_cluster_tree(text_seq):
    """
    para text_seq: list of sentences. each sentence is list of words
    return linkage
    """
    # generate condensed distance matrix
    #print("Clustering "+str(len(text_seq))+" sentences...")
    N = len(text_seq)
    #dicts = [word_tokenize(i) for i in text_seq]
    dicts = text_seq
    bigr = [set(bigrams(i)) for i in dicts]

    cond_arr = np.empty(int(comb(N,2)))
    cond_arr.fill(1.0)
    for i in range(N):
        for j in range(i+1,N):
            
            if bigr[i].intersection(bigr[j]) == set([]):
                continue
            else:
                index = cond_arr.size - int(comb(N-i,2)) + (j-i)-1
                cond_arr[index] = similarity(bigr[i],bigr[j])
    Z = hac.linkage(cond_arr,method = "complete")
    return Z
    
def make_clusters(text_seq, linkage, k):
    """
    return (clustered text, flat cluster)
    clustered text: list of clusters, each cluster is a list of sentence string
    flat cluster: [i] indicates text_seq[i] is in cluster flat[i]
    """
    cut_tree = hac.cut_tree(linkage,n_clusters = [k])
    flat = cut_tree[:,0]
    out = []
    for i in range(k):
        tmp = [text_seq[j] for j in range(len(text_seq)) if flat[j]==i]
        if tmp != []:
            out.append(tmp)

    return out, flat


# separate k clusters into etcetera and the rest
def filter_etc(clustered_text, flat, T):
    """
    return [0]: the cluster after filtering:
        [1]: a list of size T (#sentences in all documents). A = flat[i] assigns sentence i to cluster A
    """
    k = len(clustered_text)
    etcetera_list = [clustered_text[i] for i in range(len(clustered_text)) if len(clustered_text[i])<T]

    lzt = [[val for sub in etcetera_list for val in sub]]
    rest = [c for c in clustered_text if not c in etcetera_list]
    out = rest + lzt if lzt != [[]] else rest
    m = max(flat)-len(etcetera_list)+1

    flat_etc = [i for i in range(len(clustered_text)) if np.count_nonzero(flat==i) < T ] 
    flat_etc = np.array(flat_etc)
    flat_out = [flat[i]-np.where(flat_etc<flat[i])[0].size if not flat[i] in flat_etc else m for i in range(len(flat))]
    return out, flat_out


def logsumexp(arr): 
    max_ = arr.max()
    return np.log(np.sum(np.exp(arr - max_))) + max_

########################################################################################################
############################################ Content Tagger Class  #####################################
########################################################################################################

class ContentTagger():
    def __init__(self, docs, vocab,  delta_1, delta_2, T = None ,k = None, emis = None, trans = None,prior = None):
        self._docs = docs # a list of documents, each is a list of sentences
        self._delta_1 = delta_1
        self._delta_2 = delta_2
        flat_docs = [i for di in self._docs for i in di]

        
        self._tree = make_cluster_tree(flat_docs)  
        # print the linkage tree
        # plt.figure()
        # dn = hac.dendrogram(self._tree)
        # plt.show()

        if T and k:
            cluster ,flat_c = make_clusters(flat_docs,self._tree, k)
            # self._clusters is a list of clusters, each containing a list of sentences, each containing a list of words
            self._clusters,self._flat = filter_etc(cluster,flat_c, T) 
        
            self._m = len(self._clusters) # cluster[m-1] is the etc cluster
        
        self._vocab = vocab # vocab doesn't contain START/END
        self._V = len(self._vocab)
        # map all words into string from 1 to |V|+1. #0 is for START_SENT
        self._map = dict(zip(list(self._vocab),range(3,self._V+3)))
        self._map[START_SENT] = 0
        self._map[END_SENT] = 1
        self._map[UNK] = 2


        # print(self._vocab)
        # print("Vocabulary size: "+str(self._V))
        #print(self._clusters)
        #print(self._flat)
        # print(self._m)
       
        self._priors = self.prior() if prior is None else prior
        self._trans = self.trans_prob() if trans is None else trans
        self._emis = self.emission_prob_all() if emis is None else emis

        if not T and not k:
            self._m = len(self._emis) +1

        # print("After initialization, total #cluster: "+str(self._m))
    ############################################ Emission Probability for sentences  #####################################
    def emission_prob_all(self):
        # create cache for countings 
        emis = []
        # fill in cache
        for si in range(self._m - 1):
            cache = dok_matrix((self._V+3, self._V+3))

            sents = self._clusters[si]
            
            #seqs = [word_tokenize(i) for i in sents]
            bigram_seqs = [list(bigrams(j)) for j in sents]

            bigram_seq = [val for li in bigram_seqs for val in li if val[0]!= START_DOC]
            big_count_all = dict(Counter(bigram_seq)) #{('a', 'b'): 3, ('b', 'e'): 1, ('b', 'c'): 1, ('**start**', 'a'): 1, ('b', 'd'): 1}
            for tup in big_count_all:
                if(tup[0] == END_SENT): 
                    continue
                cache[self._map[tup[0]],self._map[tup[1]]] += big_count_all[tup]
            emis.append(cache)
        return emis


    def sent_logprob(self, sents_all):
        """
        input a list of sentences, each sentence is a list of words
        """
        cache = self._emis
        cache_max = np.zeros((self._V+3,self._V+3))
        emis_prob = np.zeros((self._m-1, len(sents_all))) #[i]: [[s1,s2...],[]] until state m-1
        
        # prob for emis_etc
        emis_etc = np.zeros(len(sents_all))
        
        # unigram cache
        uni_cache = [i.sum(axis=1).ravel().getA().flatten() for i in cache] # [i,j] = f_ci(map[j])
        uni_cache = np.array(uni_cache)

        
        for j in range(len(sents_all)):
            seq = sents_all[j]
            bigrams_seq = list(bigrams(seq))

            
            if bigrams_seq[0][0] == START_DOC:
                bigrams_seq = bigrams_seq[1:]
            for bigr in bigrams_seq:
                
                w = bigr[0] if bigr[0] in self._vocab else UNK
                w_prime = bigr[1] if bigr[1] in self._vocab else UNK
                # print(w+", "+w_prime)
                # print(self._map[w])
                # print(self._map[w_prime])

                state_cache = [ci[self._map[w],self._map[w_prime]] for ci in cache]
                state_cache = np.array(state_cache)
                # print(state_cache)

                word_prob = self.cal_prob(state_cache, uni_cache[:,self._map[w]])
                # print(word_prob)
                
                emis_prob[:,j] += np.log(word_prob)
                cache_max[self._map[w],self._map[w_prime]] = max(np.max(word_prob), cache_max[self._map[w],self._map[w_prime]])

        # normalize
        # for i in range(self._m-1):
        #     emis_prob[i,:] = emis_prob[i,:] - logsumexp(emis_prob[i,:])
        
        max_sum = np.sum(cache_max,axis=1)
        
        for j in range(len(sents_all)):
            seq = sents_all[j]
            #bigrams_seq = list(bigrams(word_tokenize(seq)))
            bigrams_seq = list(bigrams(seq))
            if bigrams_seq[0][0] == START_DOC:
                bigrams_seq = bigrams_seq[1:]
            for bigr in bigrams_seq:
                numer = 1 - cache_max[self._map[w],self._map[w_prime]]
                denom = self._V - max_sum[self._map[w]]
                emis_etc[j] += math.log(numer/denom)
        # emis_etc = emis_etc - logsumexp(emis_etc)
        
        emis_prob = np.vstack((emis_prob,emis_etc))
        return emis_prob

    def cal_prob(self, count_big ,count_uni):
        return (count_big+self._delta_1)/(count_uni+self._delta_1* self._V)

    ############################################ Transition Probability  ###################################

    def trans_prob(self):
        """
        docs: a list of document, each document is a list of text strings
        return a m*m matrix. a_ij = log(p(sj|si))
        """
        #D_c = np.zeros(self._m) # [i] = # docs containing sentences from cluster i
        D_c_ij = np.zeros((self._m, self._m)) # D[i,j] = D(c_i,c_j) = number of documents in which a sentence from c_i immed prcedes a sent from c_j
        ptr = 0
        for i in range(len(self._docs)):
            length = len(self._docs[i])
            c_index = self._flat[ptr:ptr+length]
            # for count in set(c_index):
            #     D_c[count] += 1
            for m,n in set(bigrams(c_index)):
                D_c_ij[m,n]+=1
            ptr+=length
        norm = np.sum(D_c_ij, axis = 1)
        
        out = (D_c_ij.T+self._delta_2)/(norm+self._delta_2*self._m)
        return np.log(out.T)

    ############################################ Prior Probability  ###################################
    def prior(self):
        """
        return the priori log2 probability of this cluster in this doc
        """
        count = np.zeros(self._m) # [i]: number of document whose head is in cluster i
        ptr = 0
        for doc in self._docs:
            length = len(doc)
            count[self._flat[ptr]]+=1
            ptr+=length
        pi = (count+self._delta_2)/(np.sum(count)+self._delta_2*self._m)
        return np.log(pi)

    ############################################ Forward/Backword/Viterbi###################################
    
    def viterbi(self,docs=None, flat = False):
        """
        docs: a list of articles, each article is a list of sentences
        return: [0] a new arrangement of clusters
                [1] a new flat clusters
        """     
        
        if docs and not flat:
            sents = [i for val in docs for i in val]
        elif not docs and not flat:
            sents = [di for i in self._docs for di in i]
        else:
            sents = docs
        
        T = len(sents)
        N = self._m
        V = np.zeros((T, N), np.float64)
        B = {}
        output_logprob = self.sent_logprob(sents)
        trans_logprob = self._trans
        prior_logprob = self._priors

        #start = self._flat[0] # which cluster the sentence 1 belongs to
        V[0,:] = prior_logprob + output_logprob[:,0]
        for i in range(N):
            B[0, i] = None

        for t in range(1, T):
                for sj in range(N):
                    best = None
                    for si in range(N):
                        va = V[t-1, si] + trans_logprob[si,sj]
                        if not best or va > best[0]:
                            best = (va, si)
                    V[t, sj] = best[0] + output_logprob[sj, t]
                    B[t, sj] = best[1]
        best = None
        for i in range(N):
            val = V[T-1, i]
            if not best or val > best[0]:
                best = (val, i)

        # traverse the back-pointers B to find the state sequence
        current = best[1]
        sequence = [current]
        for t in range(T-1, 0, -1):
            last = B[t, current]
            sequence.append(last)
            current = last      
        # seq is a flatterned cluster of length T
        sequence = list(reversed(sequence))
        out = []
        for i in range(N):
            tmp = [sents[j] for j in range(len(sents)) if sequence[j]==i]
            # if tmp!=[]:
            out.append(tmp)
        
        return out, sequence


    def forward_algo(self,docs = None, flat = False):
        if docs and not flat:
            sents = [i for val in docs for i in val]
        elif not docs and not flat:
            sents = [di for i in self._docs for di in i]
        else:
            sents = docs

        T = len(sents)
        alpha = np.zeros((T, self._m)) # [t,i]: P(E_1:t, X_t = i)
        output_logprob = self.sent_logprob(sents)
        # Initialization
        # start = self._flat[0] 
        alpha[0,:] = self._priors+output_logprob[:,0]

        # Induction
        for t in range(1, T):
            for j in range(self._m):
                alpha[t,j] = logsumexp(alpha[t-1]+self._trans[:,j])+ output_logprob[j, t]

        return alpha #doesn't calculate P(E|theta): last step

    ############################################## Train model ############################################
    def train_unsupervised(self, max_inter = 30, converg_logprob = 1e-8):
        converged = False
        iteration= 0
        log_prob = 0.0

        if len(self._emis) == 0:
                raise Exception("Number of clusters cannot be 0!")

        alpha = self.forward_algo()
        last_logprob = logsumexp(alpha[-1])
        print(" >> Initial log prob by forward algo: ", last_logprob)

        while not converged and iteration < max_inter:
            if len(self._emis) == 0:
                raise Exception("Number of clusters cannot be 0!")
            self._clusters ,self._flat= self.viterbi()
            self._emis = self.emission_prob_all()
            self._trans = self.trans_prob()
            
            alpha = self.forward_algo()
            log_prob = logsumexp(alpha[-1])

            print(">>>> At Iteration "+str(iteration))
            print("current log P(E|theta) = "+str(log_prob))
            # print("Local flat cluster: ")
            # print(self._flat)

            if iteration>0 and abs(log_prob - last_logprob) < converg_logprob:
                converged = True
            iteration += 1
            last_logprob = log_prob
        print(">> Total iteration: "+str(iteration))
        return log_prob


    def adjust_tree(self, k, tree,T, delta_1 = None, delta_2 = None): 
        #Given tree, k,T, adjust probabilities according to new cluster 
        self._tree = tree
        flat_docs = [i for di in self._docs for i in di]
        cluster ,flat_c = make_clusters(flat_docs,self._tree, k)
        self._clusters,self._flat = filter_etc(cluster,flat_c, T)
        self._m = len(self._clusters)
        if delta_1:
            self._delta_1 = delta_1
        if delta_2:
            self._delta_2 = delta_2

        self._priors = self.prior()
        self._trans = self.trans_prob()
        self._emis = self.emission_prob_all()

    ############################################## Print Information ############################################
    def print_info(self):
        print("=============== Probabilities Info =====================")
        print("Total clusters:"+str(self._m))
        print("++++++ Most probable prior: +++++++")
        print(np.argmax(self._priors))
        print("++++++ Most probable Emission for every cluster( etcetera excluded): ++++++")
        max_emis = [si.tocsr().toarray().argmax() for si in self._emis]
        max_index = [divmod(i,self._V+3) for i in max_emis] # [k]=(i,j): most probable bigram is index i,j for cluster k
        max_words = [(self._map.keys()[self._map.values().index(i)],self._map.keys()[self._map.values().index(j)]) for i,j in max_index]
        print(max_words)
        print("++++++ Most probable Transition for all clusters: ++++++")
        print(np.argmax(self._trans, axis = 1))


############################################## Try hyperpara ############################################
def hyper(tagger):
    """
    return delta1, delta2, k, T as hyper parameter tested on development data using grid search/ Sampling
    """
    tree = tagger._tree
    delta1,delta2,K,T,M = 0.0,0.0,0,0,0
    last_log = -np.inf
    # Grid search
    # for k in range(30,50,2): # 10
    #     print("Setting t...")
    #     for t in range(3,10,2): # 4
    #         print(" Setting delta_1...")
    #         for delta_1 in np.arange(1,0.00001,-0.2): #5
    #             print(" Setting delta_2...")
    #             for delta_2 in np.arange(1,0.00001, -0.2): #5
    #                 tagger.adjust_tree(k, tree, t,delta_1,delta_2)
    #                 print("Training model with hyper parameters ", delta_1, delta_2 , k, t)
    #                 tagger.train_unsupervised()

    #                 alpha = tagger.forward_algo()
    #                 log_prob = logsumexp(alpha[-1])

    #                 if log_prob > last_log:
    #                     delta1,delta2,K,T = delta_1, delta_2 , k, t
    #                     print(">>Improve hyperparameter to: ",delta1,delta2,K,T)
    #                     print("log probability ", log_prob)
    #                     last_log = log_prob

    # delta_1 = np.random.uniform(0.00001,1)
    # delta_2 = np.random.uniform(0.00001,1)
    # k = np.random.random_integers(25,50)
    # t = np.random.random_integers(3,10)

    # Sampling 30 times
    i = 0
    while i<30:
        delta_1 = np.random.uniform(0.0000001,1)
        delta_2 = np.random.uniform(0.0000001,1)
        k = np.random.random_integers(20,50)
        t = np.random.random_integers(3,10)
        print("++++++++++++++ Sampling #"+str(i)+"+++++++++++++++")
        
        tagger.adjust_tree(k, tree, t,delta_1,delta_2)
        
        print(" Training model with hyper parameters ", delta_1, delta_2 , k, t)
        try:
            log_prob = tagger.train_unsupervised()
        except Exception:
            print(" This hyperparameter does not work!")
            pass

        else:
            i += 1
            if log_prob > last_log:
                delta1,delta2,K,T = delta_1, delta_2 , k, t
                print(">>>>Improve hyperparameter to: ",delta1,delta2,K,T)
                last_log = log_prob

        # delta_1 = delta_1 + np.random.uniform(-0.1,0.1)
        # delta_1 = max(0.0001,delta_1)
        # delta_2 =  delta_2 + np.random.uniform(-0.1,0.1)
        # delta_2 = max(0.0001,delta_2)
        # k = k + np.random.random_integers(-5,5)
        # t = t + np.random.random_integers(-1,2)
        # k = max(25,k)
        # t = max(3,t)
    print(">>>>>>Best hyperparameters: ",delta1,delta2,K,T)
    return delta1,delta2,K,T




if __name__ == '__main__':
   # train_all()
    # import json
    # root_dir = "/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2006content/"
    # docs = []
    # topic_file = "/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2006topics_indexing_services.json"
    # with open(topic_file) as data_file:
    #     topics = json.load(data_file)
    # files = topics["Books and Literature"]
    # for name in files:
    #     try:
    #         f = open(root_dir+name+".txt")
    #         docs.append(f.read().decode('utf8'))
    #     except:
    #         pass
    # print("======= original docs ======")
    # print(">> total file number: "+str(len(docs)))
    # N = int(0.1*len(docs))
    # dev_set = docs[:N]
    # test_set = docs[-N:]
    # test_flat = [sent_tokenize(i) for i in test_set]
    # train_set = docs[N:-N]
    # print("develop set, train set, test set length:",len(dev_set),len(train_set),len(test_set))
    
    # delta_1, delta_2, k, T = 0.001, 0.2, 30, 3
    # myTagger = ContentTagger(train_set, k, delta_1, delta_2, T)
    #delta_1,delta_2,K,T,M = hyper(myTagger)
    
    
    #myTagger = ContentTagger(train_set, K, delta_1, delta_2, T)

    # print("========Testing Viterbi==========")
    # v ,f= myTagger.viterbi(myTagger._docs)
    # print(f)

    # from nltk import word_tokenize,sent_tokenize
    # test = ["So how about this ohhhh whaaa? This is a big foo bar right, is a. This is a large apple bar right.\
    # That must be something else right.","That can't ohhhh whaaa be something else right. This is a foo bar again right. is a"]
    
    # docs = [[['**START_DOC**','**START_SENT**', 'so', 'how', 'about', 'this', 'one', 'ohhhh', 'whaaa', '**END_SENT**'],\
    # [ '**START_SENT**', 'this', 'is', 'a', 'big', 'foo', 'bar', 'right', 'is', 'a', '**END_SENT**'], \
    # ['**START_SENT**', 'this', 'is', 'a', 'large', 'apple', 'bar', 'right', '**END_SENT**'], \
    # ['**START_SENT**', 'that', 'must', 'be', 'something', 'else', 'right', '**END_SENT**']], \
    # [['**START_DOC**', '**START_SENT**', 'that', 'ca', "n't", 'ohhhh', 'whaaa', 'be', 'something', 'else', 'right', '**END_SENT**'],
    # ['**START_SENT**', 'this', 'is', 'a', 'foo', 'bar', 'again', 'right', '**END_SENT**'],\
    #  ['**START_SENT**', 'is', 'a', '**END_SENT**']]]

    # vocab = set(['right', 'apple', 'is', 'one', 'something', 'ohhhh', 'again', 'large', 'how', 'foo', 'ca',  'be', 'that', 'big', 'else', 'must', 'a', 'about', 'bar', 'this', 'whaaa', "n't", 'so'])
    # myTagger = ContentTagger(docs,vocab, 1,1,0,3)
    
    import pickle
    train_path = "contentHMM_input/contents/News and News Media/News and News Media0.pkl"
    doc,vocab = pickle.load(open(train_path))
    delta_1,delta_2,k,T = 0.021665726605398998, 0.5116140306128781, 20, 4
    tagger = ContentTagger(doc,vocab,delta_1,delta_2,T = T,k=k)
    tagger.train_unsupervised()
    pickle.dump(tagger, open("test tagger.pkl",'wb'))


    # myTagger.print_info()
    # test_list = [val for i in myTagger._docs for val in i]
    # sent_all = [val for doc in docs for val in doc]

    # print("========Testing Sentence prob ==========")
    # sent = myTagger.sent_logprob(sent_all)
    # print(sent)
    
    # print("========Testing Viterbi==========")
    # v ,f= myTagger.viterbi()
    # print(v)
    # print(f)

    # print("========Testing Foward Algo ==========")
    # print(myTagger.forward_algo())

    print("========Testing Training ==========")
    myTagger.train_unsupervised()
    # print("Final Clusters and flat: ")
    # print(myTagger._clusters)
    # print(myTagger._flat)
