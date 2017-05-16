
from nltk import bigrams
import math
import numpy as np

from nltk import word_tokenize
import scipy.cluster.hierarchy as hac
from collections import Counter

########################################################################################################
################################# Create k clusters to initialize ######################################
########################################################################################################
# bigram:
def similarity(text1,text2):
    """
    input text as string
    """
    t1 = word_tokenize(text1)
    t2 = word_tokenize(text2)
    l1 = list(bigrams(t1)) # nltk bigrams doesn't have start/end of a sentence symbol
    l2 = list(bigrams(t2))
    
    dic1=dict.fromkeys(set(l1).union(set(l2)),0)
    dic2=dict.fromkeys(set(l1).union(set(l2)),0)
    for l in l1:
        item = dic1.get(l)+1
        dic1.update([(l,item)])
    for l in l2:
        item = dic2.get(l)+1
        dic2.update([(l,item)])
    v1 = dic1.values()
    v2 = dic2.values()
    numer = np.dot(v1,v2)
    denom = math.sqrt(np.dot(v1,v1)*np.dot(v2,v2))
    if not denom:
        return 0.0
    else:
        return 1 - float(numer)/denom



# how do we determine k? -- we don't know yet
def cluster(text_seq, k):
    """
    para text_seq: list of text strings
    return a list of k clusters, each containing a list of text sequences
    """
    # generate condensed distance matrix
    cond_arr = []
    for i in range(len(text_seq)):
        for j in range(i+1,len(text_seq)):
            dist = similarity(text_seq[i],text_seq[j])
            #print("Distance between "+text_seq[i]+" and "+text_seq[j] + " is "+str(dist))
            cond_arr.append(dist)

    #print cond_arr
    Z = hac.linkage(cond_arr,method = "complete")
    clust = hac.fcluster(Z, k, criterion='maxclust')
    print("flat cluster is ",clust)
    out = []
    for i in range(1,k):
        tmp = [text_seq[j] for j in range(len(text_seq)) if clust[j]==i]
        out.append(tmp)

    return out

# separate k clusters into etcetera and the rest
def filter(clustered_text, T):
    etcetera_list = [clustered_text[i] for i in range(len(clustered_text)) if len(clustered_text[i])<T]
    lzt = [[val for sub in etcetera_list for val in sub]]
    rest = [c for c in clustered_text if not c in etcetera_list]
    return rest+lzt


########################################################################################################
############################################ Emission Probability  #####################################
########################################################################################################
#f_ci(w'|w) for all states. [i]:{(w,w'):counts,...} for state i
BIGRAM_COUNT = []
UNI_COUNT = [] #f_ci(w) for all states
VOCAB_SIZE = [] # global list of vocabulary sizes for state[i]

def emission_prob_etc(clusters, w_prime, w, delta_1,f_seq):
    # max p_si(w'|w)
    word_prob = [] # p_si(w'|w) for all si
    for i in range(len(clusters)-1):
        emis_prob = prob_si_word(w_prime,w,i,delta_1)
        word_prob.append(emis_prob)
    numer = 1 - max(word_prob)
    
    denom = 0.0
    for u in set(f_seq):
        # max p_si(u|w) for all u in V
        all_prob = []
        for i in range(len(clusters)-1):
            emis_prob = prob_si_word(u,w,i,delta_1)
            all_prob.append(emis_prob)
        max2 = max(all_prob)
        denom += 1-max2
    return numer/denom

def prob_si_word(w_prime,w,si,delta_1):
    """
    calculate p_si(w'|w)
    """
    dicts_big = BIGRAM_COUNT[si] #{(w',w):counts}
    dicts_uni = UNI_COUNT[si]
    emis_prob = 0.0
    count_big = 0
    count_uni = 0
    if(dicts_big.has_key((w_prime,w))):
        count_big = dicts_big[(w,w_prime)]
    if dicts_uni.has_key(w):
        count_uni = dicts_uni[(w)]
    emis_prob = cal_prob(count_big,delta_1,count_uni,VOCAB_SIZE[si-1])
    return emis_prob


def cal_prob(count_big , delta_1,count_uni , size):
    return (count_big+delta_1)/(count_uni+delta_1*size)


def emission_prob_i(flat_cluster, si, seq_j, delta_1):
    """
    seq_j: a list of words
    return: the probability of seeing sequence j in state si
    """
    bigram_i = list(bigrams(flat_cluster))
    bigram_j = list(bigrams(seq_j))  #set([('b', 'c'), ('c', 'a'), ('d', 'a'), ('a', 'b'), ('b', 'e'), ('b', 'd')])
    dict_big = {}
    emi_prob = 0.0
    for w,w_prime in set(bigram_j):
        uni_count = UNI_COUNT[si-1][w]
        big_count = BIGRAM_COUNT[si-1][(w,w_prime)]
        w_prob = cal_prob(big_count, delta_1,uni_count,VOCAB_SIZE[si-1])
        emi_prob += math.log(w_prob)
    return emi_prob


def emission_prob_all(clusters, delta_1):
    """
    return a list of list, [i]:[p1...pk] is the log emission probabilities of sentence k by cluster_i
    
    """
    emis_prob = [] #[i]: [[s1,s2...],[]]
    for s_i in range(len(clusters)-1):
        sents = clusters[s_i]    #['a b c', 'a b d', 'a b e']
        seqs = [word_tokenize(i) for i in sents] #[['a', 'b', 'c'], ['a', 'b', 'd'], ['a', 'b', 'e']]
        f_seq = [k for si in seqs for k in si] #['a', 'b', 'c', 'a', 'b', 'd', 'a', 'b', 'e']
        E_k_i = [] # P(sentence k | state i)
        bigram = list(bigrams(f_seq))
        uni_count_all = dict(Counter(f_seq))
        big_count_all = dict(Counter(bigram))
        VOCAB_SIZE.append(len(f_seq))
        BIGRAM_COUNT.append(big_count_all)
        UNI_COUNT.append(uni_count_all)
       
        for seq in seqs:
            emis_ik = emission_prob_i(f_seq,s_i,seq,delta_1)
            E_k_i.append(emis_ik)
        
        emis_prob.append(E_k_i)

    # etcetera
    sm = clusters[-1] # ['c f g','y,r,z']
    token_seq = [word_tokenize(i) for i in sm] # [['c', 'f', 'g'], ['y', ',', 'r', ',', 'z']]
    f_seq = [k for si in token_seq for k in si] #['c', 'f', 'g', 'y', ',', 'r', ',', 'z']
    #bigram_l = list(bigrams(f_seq))
    prob_sm = []
    for sent_token in token_seq:
        prob_mj = 0.0
        for w,w_prime in list(bigrams(sent_token)):
            prob_mj += math.log(emission_prob_etc(clusters, w_prime, w, delta_1, f_seq))
        # prob_sm+= logprob_etc
        prob_sm.append(prob_mj)
    emis_prob.append(prob_sm)
    return emis_prob


########################################################################################################
############################################ Transition Probability  ###################################
########################################################################################################
def trans_prob(docs, clusters,delta_2):
    """
    docs: a list of document, each document is a list of text strings
    return a m*m matrix. a_ij = p(sj|si)
    """
    m = len(clusters)
    D_c = [] #[i] = # docs containing sentences from cluster i
    D_c_ij = [] # D[i,j] = D(c_i,c_j) = number of documents in which a sentence from c_i immed prcedes a sent from c_j
    for i in range(m):
        doc_count = 0
        for sent in clusters[i]:
            for doc in docs:
                doc_count += doc.count(sent)
        D_c.append(doc_count)

        doc_count_ij = []
        for j in range(m):
            count = 0
            if i==j:
                doc_count_ij.append(0)
                continue
            else:
                for doc in docs:
                    for k in range(1,len(doc)):
                        if doc[k-1] in clusters[i] and doc[k] in clusters[j]:
                            count+=1
                doc_count_ij.append(count)
        D_c_ij.append(doc_count_ij)

    D_ij = np.array(D_c_ij)
    D_i = np.array(D_c)
    out = (D_ij.T+delta_2)/(D_i+delta_2*m)
    return out.T

########################################################################################################
############################################ Viterbi decoding  ###################################
########################################################################################################


########################################################################################################
################################################## Testing  ############################################
########################################################################################################
if __name__ == '__main__':
    A = ["a b c","a b d","a b e",'c f g','y,r,z']
    out = cluster(A,4)
    print("after filter:")
    filtered = filter(out,2)
    print(filtered)
    emis = emission_prob_all(filtered, 0.2)
    print("Emission probability: ")
    print(emis)
