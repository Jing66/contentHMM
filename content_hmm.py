
from nltk import bigrams
import math
import numpy as np
import os

from nltk import word_tokenize,sent_tokenize
import scipy.cluster.hierarchy as hac
from collections import Counter
from scipy.misc import comb
import matplotlib.pyplot as plt

START_SENT = "**START_SENT**"
START_DOC = "**START_DOC**"
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

# how do we determine k? -- we don't know yet
def make_cluster_tree(text_seq):
    """
    para text_seq: list of sentences
    return linkage
    """
    # generate condensed distance matrix
    print("length of text_seq: "+str(len(text_seq)))
    N = len(text_seq)
    dicts = [word_tokenize(i) for i in text_seq]
    bigr = [set(bigrams(i)) for i in dicts]

    cond_arr = np.empty(int(comb(N,2)))
    cond_arr.fill(-1.0)
    for i in range(N):
        for j in range(i+1,N):
            index = cond_arr.size - int(comb(N-i,2)) + (j-i)-1
            if bigr[i].intersection(bigr[j]) == set([]):
                cond_arr[index] = 1.0
            else:
                cond_arr[index] = similarity(bigr[i],bigr[j])
    print("Out of "+str(cond_arr.size)+ " pairs of sentences, #pairs has distance < 1: " + str(np.count_nonzero(cond_arr!=1.0)))
    
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
    m = max(flat)-len(etcetera_list)+1

    flat_etc = [i for i in range(len(clustered_text)) if np.count_nonzero(flat==i) < T ] 
    flat_etc = np.array(flat_etc)
    flat_out = [flat[i]-np.where(flat_etc<flat[i])[0].size if not flat[i] in flat_etc else m for i in range(len(flat))]
    return rest+lzt, flat_out

def insert_starts(docs):
    """
    insert **START_DOC** before the start of each article; **START_SENT** before the start of each sentence Return the list of articles
    return a list of documents, each is a list of sentences
    """
    out = []
    sents = []

    for doc in docs:
        try:
            sents = sent_tokenize(doc)
        except:
            continue
        start_sents = [START_SENT+" "+sent.lower() for sent in sents]
        #start_sents.insert(0,START_DOC)
        start_sents[0] = START_DOC+" "+start_sents[0]
        out.append(start_sents)

    return out

def vocab(docs):
    # return a set of all words in corpus 
    sents = [word_tokenize(i) for i in docs]
    
    vocab = [set(si) for si in sents]
    sets = reduce(lambda a,b:a.union(b),vocab)
    return sets

def logsumexp(arr): 
    max_ = arr.max()
    return np.log(np.sum(math.e**(arr - max_))) + max_
########################################################################################################
############################################ Content Tagger Class  #####################################
########################################################################################################

class ContentTagger():
    def __init__(self, docs, k, delta_1, delta_2, T):
        self._docs = insert_starts(docs) # omit **Start Doc**
        self._delta_1 = delta_1
        self._delta_2 = delta_2
        flat_docs = [i for di in self._docs for i in di]
        # clustered sentences have **start**
        self._tree = make_cluster_tree(flat_docs)

        # print the linkage tree
        # plt.figure()
        # dn = hac.dendrogram(self._tree)
        # plt.show()

        cluster ,flat_c = make_clusters(flat_docs,self._tree, k)
        self._clusters,self._flat = filter_etc(cluster,flat_c, T)
        self._m = len(self._clusters)
        self._vocab = vocab(docs)
        self._V = len(self._vocab)
        print("After filter, total #cluster: "+str(self._m))
        #print(self._vocab)
        print("Vocabulary size: "+str(self._V))
        print("=============== Clusters =====================")
        #print(self._clusters)
        print(self._flat)
   
        self._priors = self.prior()
        self._trans = self.trans_prob()
        self._emis = self.emission_prob_all()
        print("=============== Probability =====================")
        print("++++++Prior probability:")
        print(self._priors)
        print("++++++transition probability:")
        print(self._trans)
        print("++++++Emission probability:")
        print(self._emis)


    ############################################ Emission Probability  #####################################

    def emission_prob_i(self, seq_j, bigram_count,uni_count):
        """
        seq_j: a list of words
        return: the probability of seeing sentence j in state si
        """
       
        # print(">>>>>>>>>Emitting sequence: "+seq_j)
        uni_dict = uni_count[-1]
        bigram_dict = bigram_count[-1]
        bigram_j = list(bigrams(word_tokenize(seq_j))) #[('**START_SENT**', 'So'), ('So', 'how'), ('how', 'about'), ('about', 'this'), ('this', 'one'), ('one', 'ohhhh'), ('ohhhh', 'whaaa'), ('whaaa', '?')]
        # ignore start of document
        if bigram_j[0] == (START_DOC,START_SENT):
            bigram_j = bigram_j[1:]
        emi_prob = 0.0
        for w,w_prime in set(bigram_j):
            uni_count = uni_dict.get(w,0)
            big_count = bigram_dict.get((w,w_prime),0)
            
            w_prob = self.cal_prob(big_count, uni_count)
            #print(w, uni_count,w_prime, big_count,w_prob)
            emi_prob += math.log(w_prob)
        return emi_prob


    def emission_prob_etc(self,seq_j, bigram_count,uni_count):
        """
        return the log probability of sentence j in cluster etcetera
        """
        # max p_si(w'|w)
        tokens = list(bigrams(word_tokenize(seq_j)))

        if tokens[0] == (START_DOC,START_SENT):
            tokens = tokens[1:]
        etc_logprob = 0.0
        

        for w,w_prime in tokens:
            
            word_prob = 0.0 # p_si(w'|w) for state i at iteration i
            #for state 1...m-1
            for i in range(self._m-1):
                new_prob= self.prob_si_word(w_prime,w,i,bigram_count,uni_count)
                if new_prob > word_prob:
                    word_prob = new_prob
            numer = 1 - word_prob # 1- max(P_si(w'|w))      
            denom = 0.0
            for u in self._vocab:
                # max p_si(u|w) for all u in V
                all_prob = 0.0

                for i in range(self._m-1):
                    new_prob = self.prob_si_word(u,w,i,bigram_count,uni_count)
                    #print(" In cluster "+str(i)+", P("+u+" | "+w+") = "+str(new_prob))
                    if new_prob > all_prob:
                        all_prob = new_prob
                
                denom += 1 - all_prob
                #print(u, denom)
            
        etc_logprob += math.log(numer/denom)
        return etc_logprob


    def emission_prob_all(self):
        """
        return a m*T matrix. [i,k] is the log emission probabilities of sentence k by cluster_i. log(P(sent_k | cluster_i))
        
        """
        #f_ci(w'|w) for all states. [i]:{(w,w'):counts,...} for state i
        bigram_count = []
        uni_count = [] #f_ci(w) for all states. 

        sents_all = [i for ci in self._docs for i in ci] # all sentences as a list
        emis_prob = np.empty((self._m, len(sents_all))) #[i]: [[s1,s2...],[]]

        # clusters 1...m-1
        for s_i in range(self._m - 1):
            sents = self._clusters[s_i]    #['a b c', '**start** a b d', 'a b e']
            seqs = [word_tokenize(i) for i in sents] #[['a', 'b', 'c'], ['**start**','a', 'b', 'd'], ['a', 'b', 'e']]
            E_k_i = [] # P(sentence k | state i)
            #f_seq = [val for li in seqs for val in li]
            bigram_seqs = [list(bigrams(j)) for j in seqs] 
            bigram_seq = [val for li in bigram_seqs for val in li] #[('a', 'b'), ('b', 'c'), ('**start**', 'a'), ('a', 'b'), ('b', 'd'), ('a', 'b'), ('b', 'e')]
            f_seq = [tup[0] for tup in bigram_seq]
            #uni_count_all = dict(Counter(f_seq)) #{'a': 3, 'c': 1, 'b': 3, 'e': 1, 'd': 1, '**start**': 1}
            big_count_all = dict(Counter(bigram_seq)) #{('a', 'b'): 3, ('b', 'e'): 1, ('b', 'c'): 1, ('**start**', 'a'): 1, ('b', 'd'): 1}
            uni_count_all = dict(Counter(f_seq))
            # add the stats into dictionary 
            bigram_count.append(big_count_all)
            uni_count.append(uni_count_all)
            
           # sentence 1...T
            for seq in sents_all:
                emis_ik = self.emission_prob_i(seq, bigram_count, uni_count) # P(sentence j| state i)
                #print("Sentence ["+seq + "] prob is "+str(math.e**emis_ik))
                E_k_i.append(emis_ik)

            E_k = np.array(E_k_i)
            emis_prob[s_i,:] = E_k - logsumexp(E_k)
        # etcetera
        prob_sm = []
        for seq in sents_all:
            prob_mj = self.emission_prob_etc(seq,bigram_count,uni_count)
            prob_sm.append(prob_mj)
        E_etc = np.array(prob_sm)
        emis_prob[self._m-1,:] = E_etc - logsumexp(E_etc)
        return emis_prob

    def prob_si_word(self, w_prime,w,si,bigram_count,uni_count):
        """
        calculate p_si(w'|w) given counts at s_i
        """
        dicts_big = bigram_count[si] #{(w',w):counts}
        dicts_uni = uni_count[si]
        emis_prob = 0.0
        
        count_big = dicts_big.get((w,w_prime),0)
        count_uni = dicts_uni.get((w),0)
        emis_prob = self.cal_prob(count_big,count_uni)
        return emis_prob

    def cal_prob(self, count_big ,count_uni):
        return float(count_big+self._delta_1)/(count_uni+self._delta_1* self._V)


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
        # print(" >> Transition mode counts: ")
        # print(D_c_ij)
        # print(out)
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
      
        pi = (count+self._delta_2)/(count.sum()+self._delta_2*self._m)
        return np.log(pi)

   
    ############################################ Forward/Backword/Viterbi###################################
    
    def viterbi(self,docs):
        """
        docs: a list of articles, each article is a list of sentences
        return a new arrangement of clusters
        TODO: also return new flat clusters
        """
        
        sents = [di for i in self._docs for di in i]
        
        T = len(sents)
        N = self._m
        V = np.zeros((T, N), np.float64)
        B = {}
        output_logprob = self._emis
        trans_logprob = self._trans
        prior_logprob = self._priors

        start = self._flat[0] # which cluster the sentence 1 belongs to
        for i in range(N):
            V[0, i] = prior_logprob[i] + output_logprob[i, start]   
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
            out.append(tmp)
        # sequence start with 1
        sequence = [i for i in sequence]
        return out, sequence


    def forward_algo(self):
        sents = [di for i in self._docs for di in i] # All sentences in docs: ['a b c', 'a b d', 'a b e', 'c f g', 'y,r,z']
        T = len(sents)
        alpha = np.zeros((T, self._m))
 
        
        for i in range(self._m): #state i
            alpha[0, i] = self._priors[i] + self._emis[i, 0]

        # Induction
        for t in range(1, T):
            cluster_index = -1
            for i in range(self._m):
                if sents[t] in self._clusters:
                    cluster_index = i # which cluster sentence t belongs to
                    break

            output_logprob = self._emis[:,cluster_index-1]

            for i in range(self._m):
                summand = alpha[t-1] + self._trans[i]
                alpha[t, i] = logsumexp(summand) + output_logprob[i]

        return alpha #doesn't calculate P(E|theta): last step

    ############################################## Train ############################################
    def train_unsupervised(self, max_inter = 100, converg_logprob = 1e-6):
        converged = False
        iteration= 0
        log_prob = 0.0
        last_logprob = 0.0

        while not converged and iteration < max_inter:
            self._clusters ,self._flat= self.viterbi(self._docs)
            self._emis = self.emission_prob_all()
            self._trans = self.trans_prob()
            
            alpha = self.forward_algo()
            
            log_prob = logsumexp(alpha[-1])

            print(">> At Iteration "+str(iteration))
            print("log P(E|theta) = "+str(log_prob))
            print("New emission log prob: ")
            print(self._emis)
            print("New transition log prob: ")
            print(self._trans)

            if iteration>0 and abs(log_prob - last_logprob) < converg_logprob:
                converged = true
            iteration += 1
            last_logprob = log_prob


########################################################################################################
################################################## Testing  ############################################
########################################################################################################
if __name__ == '__main__':
    import json
    root_dir = "/Users/liujingyun/Desktop/NLP/nyt_corpus/data/2006content/"
    docs = []
    topic_file = "/Users/liujingyun/Desktop/NLP/nyt_corpus/data/topics.txt"
    with open(topic_file) as data_file:
        topics = json.load(data_file)
    files = topics["Restaurants"]
    for name in files:
        try:
            f = open(root_dir+name+".txt")
            docs.append(f.read().decode('utf8'))
        except:
            pass
    print("======= original docs ======")
    print(">> total file number: "+str(len(docs)))
    delta_1, delta_2, k, T = 0.000001, 0.0002, 30, 5
    myTagger = ContentTagger(docs, k, delta_1, delta_2, T)

    # test = ["So how about this one ohhhh whaaa? This is a big foo bar right, is a. This is a large apple bar right.\
    # That must be something else right.","That can't ohhhh whaaa be something else right. This is a foo bar again right. is a"]
    # myTagger = ContentTagger(test, 3,1,1,3)
    # test_list = myTagger._docs
    
    # print("========Testing Viterbi==========")
    # v ,f= myTagger.viterbi(test_list)
    # print(v)
    # print(f)

    # print("========Testing Foward Algo ==========")
    # print(myTagger.forward_algo())

    # print("========Testing Training ==========")
    # myTagger.train_unsupervised()
    # print("Final Clusters and flat: ")
    # print(myTagger._clusters)
    # print(myTagger._flat)








    


