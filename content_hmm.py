from nltk import bigrams
import math
import numpy as np
import heapq
import gc

import scipy.cluster.hierarchy as hac
from collections import Counter
from scipy.misc import comb
from scipy.sparse import dok_matrix
# import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from tagger_test import *

STOPWORDS = set(stopwords.words('english'))
START_SENT = "**START_SENT**"
START_DOC = "**START_DOC**"
END_SENT = "**END_SENT**"
UNK = "**UNK**"
GAMMA = 0.2

########################################################################################################
################################# Create k clusters to initialize ######################################
########################################################################################################
# n-grams:
def similarity(seq1,seq2):
    """
    input text bigrams as list
    """
    numer = len(seq1.intersection(seq2))
    denom = math.sqrt(len(seq1)*len(seq2))
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
    
    # dicts = text_seq
    # bigr = [set(bigrams(i)) for i in dicts]

    uni = [set(i) for i in text_seq]
    gc.collect()

    cond_arr = np.empty(int(comb(N,2)))
    cond_arr.fill(1.0)
    for i in range(N):
        for j in range(i+1,N):
            
            # Bigram feature
            # if bigr[i].intersection(bigr[j]) == set([]):
            #     continue
            # else:
            #     index = cond_arr.size - int(comb(N-i,2)) + (j-i)-1
            #     cond_arr[index] = similarity(bigr[i],bigr[j])

            # Unigram feature
            if uni[i].intersection(uni[j]) == set([]):
                continue
            else:
                index = cond_arr.size - int(comb(N-i,2)) + (j-i)-1
                cond_arr[index] = 1 - similarity(uni[i],uni[j])

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
    def __init__(self, vocab, emis, trans, prior, delta_1, delta_2):
        """
        vocab: set of words in the corpus, excluding start/end sentence/doc indicators
        """
        self._delta_1 = delta_1
        self._delta_2 = delta_2
       
        self._priors = prior
        self._trans =  trans
        self._emis = emis

        self._m = len(self._emis) +1

        self._vocab = vocab
        self._V = len(vocab)
        # map all words into string from 1 to |V|+1. #0 is for START_SENT
        self._map = dict(zip(list(self._vocab),range(3,self._V+3)))
        self._map[START_SENT] = 0
        self._map[END_SENT] = 1
        self._map[UNK] = 2


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

    ############################################ Forward/Backword/Viterbi###################################
    
    def viterbi(self,docs, flat = False):
        """
        docs: a list of articles, each article is a list of sentences
        return: [0] a new arrangement of clusters
                [1] a new flat clusters
        """     
        
        if not flat:
            sents = [i for val in docs for i in val]
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


    def forward_algo(self,docs, flat = False):
        if not flat:
            sents = [i for val in docs for i in val]
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

    def update(self, emis = None, trans = None, priors = None):
        if emis:
            self._emis = emis
            self._m = len(emis)+1
        if trans.any():
            self._trans = trans
        if priors.any():
            self._priors = priors


    ######################################### Print Information ########################################
    def info(self, n_state = 5, n_emis = 15):
        out = ""
        out+="\n>> Total cluster #:"+str(self._m)
        out+="\n>> Vocabulary size: "+str(self._V)
        # Prior info print
        out+="\n\n>> Top 5 probable prior clusters and prob:"
        top = heapq.nlargest(n_state, range(self._m), self._priors.take)
        for t,p in zip(top, np.exp(self._priors[top]).tolist()):
           out+="\nP(s_%s) = %s" %(t,str(p))

        # Transition info print
        out+="\n\n>> Top 5 Most probable Transition for all clusters:"
        for i in range(self._m):
            top3 = heapq.nlargest(n_state, range(self._m), self._trans[i,:].take)
  
            out+="\nTopic "+str(i)+": top 5 transtions are topics "+str(top3)
            for j in top3:
                out+="\nP(s_%s|s_%s) = %s" %(j,i,np.exp(self._trans[i,j]))

        # Emission info print
        out+="\n\n>> Top 15 Bigram Emission for every cluster/topic (etcetera excluded) ranking from highest to lowest:"
        # max_emis = [si.tocsr().toarray().argmax() for si in self._emis]
        # max_index = [divmod(i,self._V+3) for i in max_emis] # [k]=(i,j): most probable bigram is index i,j for cluster k
        # max_words = [(self._map.keys()[self._map.values().index(i)],self._map.keys()[self._map.values().index(j)]) for i,j in max_index]
        # print(max_words)
        for i in range(len(self._emis)):
            emis = self._emis[i].tocsr().toarray()
            ind = emis.flatten().argsort()[-n_emis:]
            X,Y = np.unravel_index(ind, emis.shape)
            
            words = [(self._map.keys()[self._map.values().index(x)],self._map.keys()[self._map.values().index(y)]) for x,y in zip(X,Y)]
            words.reverse()
            counts = [emis[x][y] for x,y in zip(X,Y)]
            counts.reverse()
            out+="\n\n>> Topic "+str(i)+" emission and counts: "
            for w,c in zip(words,counts):
                out+="\nBigram emission %s, counts = %s" %(w,str(c))

            uni = np.sum(emis, axis =1)
            top = heapq.nlargest(10, range(len(uni)), uni.take)
            words_uni = [self._map.keys()[self._map.values().index(x)] for x in top]
            counts_uni = [uni[i] for i in top]
            for w,c in zip(words_uni,counts_uni):
                out+="\nUnigram emission: '%s'. Counts = %s" %(w,str(c))
            
            end_emis = emis[:,1]
            if not np.all(end_emis):
                end_top = heapq.nlargest(n_state, range(len(end_emis)), end_emis.take)
                end_words = [self._map.keys()[self._map.values().index(x)] for x in end_top]
                end_counts = [end_emis[i] for i in end_top]
                out+="\nTop 5 words followed by END_SENT: "
                out+=str(end_words)
                out+="counts = "+str(end_counts)

        return out
        


   
########################################################################################################
############################################ Content Tagger Trainer ####################################
########################################################################################################

class ContentTaggerTrainer():
    def __init__(self, docs, vocab, k, T, delta_1, delta_2, tree = None):
        self._docs = docs
        self._vocab = vocab
        self._k = k
        self._T = T
        self._delta_1 = delta_1
        self._delta_2 = delta_2

        flat_docs = [i for di in docs for i in di]
        if tree is None:
            self._tree = make_cluster_tree(flat_docs) 
        else:
            self._tree = tree
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

        self._priors = self.prior() 
        self._trans = self.trans_prob() 
        self._emis = self.emission_prob_all()

 ############################################ Emission Probability for Bigrams  #####################################
    def emission_prob_all(self):
        # create cache for countings 
        emis = []
        # fill in cache
        for si in range(self._m - 1):
            cache = dok_matrix((self._V+3, self._V+3))

            sents = self._clusters[si]
            
            bigram_seqs = [list(bigrams(j)) for j in sents]

            bigram_seq = [val for li in bigram_seqs for val in li if val[0]!= START_DOC]
            big_count_all = dict(Counter(bigram_seq)) #{('a', 'b'): 3, ('b', 'e'): 1, ('b', 'c'): 1, ('**start**', 'a'): 1, ('b', 'd'): 1}
            for tup in big_count_all:
                if(tup[0] == END_SENT): 
                    continue
                if(tup[0] == START_SENT and tup[1] == END_SENT):
                    continue
                cache[self._map[tup[0]],self._map[tup[1]]] += big_count_all[tup]
            emis.append(cache)
            # sanity check
            with open("war_record_emis.txt",'a') as f:
                if cache[0,1]!= 0:
                    f.write(str(cache[0,1]))
                    f.write(str(sents))
                f.write('\n')
        return emis

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
            for m,n in set(bigrams(c_index)):
                D_c_ij[m,n]+=1
            ptr+=length
        norm = np.sum(D_c_ij, axis = 1)
        
        out = (D_c_ij.T+self._delta_2)/(norm+self._delta_2*self._m)
        out = np.log(out.T)

        # add stickiness: [i,j] = gamma*1(i=j) + (1-gamma) * P(Si|Sj)
        # sticky = np.identity(self._m)* GAMMA
        # out = (1 - GAMMA) * out + sticky

        return out

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

    ############################################## Train model ############################################

    def train_unsupervised(self, max_iter = 30 , converg_logprob = 1e-8):
        converged = False
        iteration= 0
        log_prob = 0.0
        model = ContentTagger(self._vocab, self._emis, self._trans, self._priors, self._delta_1, self._delta_2)
        if len(self._emis) == 0:
                print("Number of clusters cannot be 0!")
                raise Exception("Number of clusters cannot be 0!")

        alpha = model.forward_algo(self._docs)
        last_logprob = logsumexp(alpha[-1])
        print(" >> Initial log prob by forward algo: ", last_logprob)
        
        while not converged and iteration < max_iter:
            # if len(self._emis) == 0:
            #     print("Number of clusters cannot be 0!")
            #     raise Exception("Number of clusters cannot be 0!")
            self._clusters ,self._flat= model.viterbi(self._docs)

            self._emis = self.emission_prob_all()
            self._trans = self.trans_prob()
            self._priors = self.prior()

            model.update(self._emis, self._trans, self._priors)
            alpha = model.forward_algo(self._docs)

            log_prob = logsumexp(alpha[-1])

            print(">>>> At Iteration "+str(iteration))
            print("current log P(E|theta) = "+str(log_prob))
            # print(self._flat)
            # print(np.exp(self._priors))

            if iteration>0 and (abs(log_prob - last_logprob) < converg_logprob or log_prob < last_logprob):
                converged = True
            
            iteration += 1
            last_logprob = log_prob
        print(">> Total iteration: "+str(iteration))
        return model

    def adjust_tree(self, k, tree,T, delta_1, delta_2 ): 
        #Given tree, k,T, adjust probabilities according to new cluster, return a new trainer
       
        new_trainer = ContentTaggerTrainer(self._docs, self._vocab, k, T, delta_1, delta_2,tree)
        return new_trainer

 
########################################################################################################
############################################   Testing    ####################################
########################################################################################################

if __name__ == '__main__':
   
    # from nltk import word_tokenize,sent_tokenize
    # test = ["So how about this ohhhh whaaa? This is a big foo bar right, is a. This is a large apple bar right.\
    # That must be something else right.","That can't ohhhh whaaa be something else right. This is a foo bar again right. is a"]
    
    docs = [[['**START_DOC**','**START_SENT**', 'so', 'how', 'about', 'this', 'one', 'ohhhh', 'whaaa', '**END_SENT**'],\
    [ '**START_SENT**', 'this', 'is', 'a', 'big', 'foo', 'bar', 'right', 'is', 'a', '**END_SENT**'], \
    ['**START_SENT**', 'this', 'is', 'a', 'large', 'apple', 'bar', 'right', '**END_SENT**'], \
    ['**START_SENT**', 'that', 'must', 'be', 'something', 'else', 'right', '**END_SENT**']], \
    [['**START_DOC**', '**START_SENT**', 'that', 'ca', "n't", 'ohhhh', 'whaaa', 'be', 'something', 'else', 'right', '**END_SENT**'],
    ['**START_SENT**', 'this', 'is', 'a', 'foo', 'bar', 'again', 'right', '**END_SENT**'],\
     ['**START_SENT**', 'is', 'a', '**END_SENT**']]]

    vocab = set(['right', 'apple', 'is', 'one', 'something', 'ohhhh', 'again', 'large', 'how', 'foo', 'ca',  'be', 'that', 'big', 'else', 'must', 'a', 'about', 'bar', 'this', 'whaaa', "n't", 'so'])
    trainer = ContentTaggerTrainer(docs,vocab, 3,0,1,1)

    print("========Testing Training ==========")
    myTagger = trainer.train_unsupervised(10,1e-8)
    print(myTagger._map)
    # print(trainer._flat)
    myTagger.print_info()

    

    # test_list = [val for i in myTagger._docs for val in i]
    # sent_all = [val for doc in docs for val in doc]

    # print("========Testing Sentence prob ==========")
    # sent = myTagger.sent_logprob(sent_all)
    # print(sent)
    
    # print("========Testing Viterbi==========")
    # v ,f= myTagger.viterbi(docs)
    # print(f)

    # print("========Testing Foward Algo ==========")
    # print(myTagger.forward_algo(docs))
