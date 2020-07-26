import re
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


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