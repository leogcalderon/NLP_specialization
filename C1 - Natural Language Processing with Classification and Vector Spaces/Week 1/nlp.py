import re
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
