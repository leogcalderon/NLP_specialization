import re
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string

def preprocess_tweet(tweet):
    '''
    tweet = string
    returns a list with the tweet words preprocessed
    '''

    '''Cleaning tweet'''
    tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet2 = re.sub(r'#', '', tweet2)
    tweet2 = re.sub(r'\$\w*', '', tweet2)
    tweet2 = re.sub(r'^RT[\s]+', '', tweet2)

    '''Tokenize'''
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet2)

    '''Stop words'''
    stopwords_english = stopwords.words('english')
    tweet_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english:
            tweet_clean.append(word)

    '''Stemming'''
    stemmer = PorterStemmer()
    tweet_stemmed = []
    for word in tweet_clean:
        stem_word = stemmer.stem(word)
        tweet_stemmed.append(stem_word)

    return tweet_stemmed
    
def get_freq(tweets, labels):
    '''
    tweets = list of tweets
    labels = list of tweets labels
    returns a dictionary with frequencies of each word for each label
    '''
    freq = {}
    labels = labels.tolist()
    
    for tweet, y in zip(tweets, labels):
        for word in preprocess_tweet(tweet):
            pair = (word, y)
            freq[pair] = freq.get(pair, 0) + 1
    
    return freq

def plot_vectors(M):
    
    rows,cols = M.T.shape
    maxes = 1.1*np.amax(abs(M), axis = 0)
    
    ax = plt.axes()

    for i,l in enumerate(range(0,cols)):
        ax.arrow(0,0,M[i,0],M[i,1],head_width=0.1,head_length=0.3,color = "k")
        ax.annotate("Vector {}".format(i),(M[i,0],M[i,1]))

    plt.plot(0,0,'ok')
    plt.xlim([-maxes[0],maxes[0]])
    plt.ylim([-maxes[1],maxes[1]])
    plt.grid(b=True, which='major')
    plt.show()

def cosine(v,w):
    dot_product = np.dot(v,w)
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    cos = dot_product/(norm_v*norm_w)
    return cos

def euclidean(v,w):
    distance = np.linalg.norm(v - w)
    return distance

def assign_unk(word):
    
    punct = set(string.punctuation)
    digit = set(string.digits)
    upper = set(string.ascii_uppercase)
    
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]
    
    for letter in word:
        if letter in punct:
            return "--unk_punc--"
        
        if letter in digit:
            return "--unk_digit--"
        
        if letter in upper:
            return "--unk_upper--"
        
    for noun in noun_suffix:
        if word.endswith(noun):
            return "--unk_noun--"
        
    for verb in verb_suffix:
        if word.endswith(verb):
            return "--unk_verb--"
        
    for adj in adj_suffix:
        if word.endswith(adj):
            return "--unk_adj--"
        
    for adv in adv_suffix:
        if word.endswith(adv):
            return "--unk_adv--"
        
        else:
            return "--unk--"
        
def get_word_tag(line, vocab):
    if not line.split():
        word = "--n--"
        tag = "--s--"
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
    return word, tag

def preprocess(vocab, data_fp):
    """
    Preprocess data
    """
    orig = []
    prep = []

    # Read data
    with open(data_fp, "r") as data_file:

        for cnt, word in enumerate(data_file):

            # End of sentence
            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue

            # Handle unknown words
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unk(word)
                prep.append(word)
                continue

            else:
                orig.append(word.strip())
                prep.append(word.strip())

    assert(len(orig) == len(open(data_fp, "r").readlines()))
    assert(len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep