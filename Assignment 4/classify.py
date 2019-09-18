from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pickle
from textblob import TextBlob
import re
import numpy as np
import requests
from scipy.sparse import lil_matrix
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def afinn_download():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
            
    return afinn

def read_tweets():
    tweets = pickle.load(open('tweets.pkl', 'rb'))
    tweets = [t['text'] for t in tweets][:200]
    return tweets

def tokenize1(tweets):
    return re.sub('\W+', ' ', tweets.lower()).split()

def afinn_sentiment(terms, afinn):
    total = 0.
    for t in terms:
        if t in afinn:
            print('\t%s=%d' % (t, afinn[t]))
            total += afinn[t]
    return total

def afinn_sentiment2(terms, afinn):
    pos = 0
    neg = 0
    for w in terms:
        if w in afinn:
            #print('\t%s=%d' % (t, afinn[t]))
            if afinn[w] > 0:
                pos += afinn[w]
            else:
                neg += -1 * afinn[w]
    return pos, neg

def tweets_score(tweets, afinn):
    positive_tweets = []
    negative_tweets = []
    neutral_tweets = []
    
    tokens = [tokenize1(t) for t in tweets]
    for token_list, tweet in zip(tokens, tweets):
        pos, neg = afinn_sentiment2(token_list, afinn)
        if pos > neg:
            positive_tweets.append((tweet, pos, neg))
        elif neg > pos:
            negative_tweets.append((tweet, pos, neg))
        elif pos == neg:
            neutral_tweets.append((tweet, pos, neg))
    
    print('Here are the positive tweets.')
    pos_len = len(positive_tweets)
    print(pos_len)
    for tweet, pos, neg in sorted(positive_tweets,key=lambda x: x[1],reverse=True):
        print(pos, neg, tweet)
    print('\n')
    print('\n')
    
    print('Here are the negative tweets.')
    neg_len = len(negative_tweets)
    print(neg_len)
    for tweet, pos, neg in sorted(negative_tweets, key=lambda x: x[2], reverse=True):
        print(pos, neg, tweet)
        
    print('\n')
    print('\n')
    
    neu_len = len(neutral_tweets)
    print(neu_len)    
    print('Here are the neutral tweets.')
    for tweet, pos, neg in neutral_tweets:
       print(pos, neg, tweet)
    return pos_len, neg_len, neu_len, positive_tweets, negative_tweets, neutral_tweets

def get_census_names():
    """ Fetch a list of common male/female names from the census.
    For ambiguous names, we select the more frequent gender."""
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                  females_pct[f] > males_pct[f]])    
    return male_names, female_names

def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):
    """ Split a tweet into tokens."""
    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()
    if prefix:
        tokens = ['%s%s' % (prefix, t) for t in tokens]
    return tokens

def tweet2tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):
    """ Convert a tweet into a list of tokens, from the tweet text and optionally the
    user description. """
    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                       collapse_urls, collapse_mentions)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens

def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))  # If term not present, assign next int.
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  # looking up a key; defaultdict takes care of assigning it a value.
    print('%d unique terms in vocabulary' % len(vocabulary))
    return vocabulary

def make_feature_matrix(tokens_list, vocabulary, tweets):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()  # convert to CSR for more efficient random access.

def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()


def get_gender(tweet, male_names, female_names):
    name = get_first_name(tweet)
    if name in female_names:
        return 1
    elif name in male_names:
        return 0
    else:
        return -1

def do_cross_val(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(n_splits=nfolds, random_state=42, shuffle=True)
    accuracies = []
    for train_idx, test_idx in cv.split(X):
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    print(np.std(accuracies))
    print(accuracies)
    return avg

def main():
    positive = []
    negative = []
    neutral = []
    males = []
    females = []
    male_tweets = []
    female_tweets = []
    unknown_tweets = []
    
    
    c = open('classify.txt','w',encoding = 'utf-8')
    
    print('Lets do Sentiment Analysis Based on Afinn')
    afinn = afinn_download()
    tweets = read_tweets()
    #print (tweets)
    print('Found %d Tweets' % (len(tweets)))
    c.write('Total number of tweets: %d' %len(tweets))
    print("Sentiment analysis based on AFINN...")
    c.write('\n\n==================Sentiment analysis based on AFINN=======================')
    pos_len, neg_len, neu_len, positive_tweets, negative_tweets, neutral_tweets = tweets_score(tweets, afinn)
    print('\n\nNumber of Positive Tweets: %d' %pos_len)
    c.write('\n\nNumber of Positive Tweets: %d' %pos_len)
    print('\n\nNumber of Negative Tweets: %d' %neg_len)
    c.write('\n\nNumber of Negative Tweets: %d' %neg_len)
    print('\n\nNumber of Neutral Tweets: %d' %neu_len)
    c.write('\n\nNumber of Neutral Tweets: %d' %neu_len)
    
    pos = ((pos_len)/(len(tweets)))*100
    neg = ((neg_len)/(len(tweets)))*100
    neu = ((neu_len)/(len(tweets)))*100
    
    print('\nPercentage of tweets with positive sentiment: %0.2f%%' %pos)
    c.write('\n\nPercentage of tweets with positive sentiment: %0.2f%%' %pos)
    print('\nPercentage of tweets with negative sentiment: %0.2f%%' %neg)
    c.write('\n\nPercentage of tweets with negative sentiment: %0.2f%%' %neg)
    print('\nPercentage of tweets with neutral sentiment: %0.2f%%' %neu)
    c.write('\n\nPercentage of tweets with neutral sentiment: %0.2f%%' %neu)
    
    print('\nPositive Tweet Sample:')
    c.write('\n\nPositive Tweet Sample:')
    print(positive_tweets[0][0])
    c.write(positive_tweets[0][0])
    print('\nNegative Tweet Sample:')
    c.write('\n\nNegative Tweet Sample:')
    print(negative_tweets[0][0])
    c.write(negative_tweets[0][0])
    print('\nNeutral Tweet Sample:')
    c.write('\n\nNeutral Tweet Sample:')
    print(neutral_tweets[0][0])
    c.write(neutral_tweets[0][0])
    
    print('\n')
    print('========================Gender Analysis of tweets=======================')
    c.write('\n\n\n\n========================Gender Analysis of tweets=======================')
    tweets1 = pickle.load(open('tweets.pkl', 'rb'))
    tweets2 = [t for t in tweets1]
    
    male_names, female_names = get_census_names()
    print('found %d female and %d male names' % (len(female_names), len(male_names)))
    tokens_list = [tweet2tokens(t, use_descr=True, lowercase=True,
                            keep_punctuation=False, descr_prefix='d=',
                            collapse_urls=True, collapse_mentions=True)
              for t in tweets2]
    vocabulary = make_vocabulary(tokens_list)
    
    for t in tweets2:
        g= get_gender(t, male_names, female_names)
        if g ==0:
            male_tweets.append(t['text'])
        elif g == 1:
            female_tweets.append(t['text'])
        elif g == -1:
            unknown_tweets.append(t['text'])
            
    
            
    
    X = make_feature_matrix(tokens_list, vocabulary, tweets2)
    print('shape of X:', X.shape)
    y = np.array([get_gender(t, male_names, female_names) for t in tweets2])
    cnt = Counter(y)

    for ct in cnt.keys():
        if ct == -1:
            ucnt = cnt[ct]
        elif ct == 0:
            mcnt = cnt[ct]
        elif ct == 1:
            fcnt = cnt[ct]
    
    print('Number of tweets by Males: %d \nNumber of tweets by Females: %d \nNumber of tweets by unknown: %d' %(mcnt,fcnt,ucnt))
    c.write('\n\nNumber of tweets by Males: %d'%mcnt)
    c.write('\n\nNumber of tweets by Females: %d' %fcnt) 
    c.write('\n\nNumber of tweets by unknown: %d' %ucnt)
    print('\n\nSample tweet by male:')
    print(male_tweets[0])
    c.write('\n\nSample tweet by male:')
    c.write(male_tweets[0])
    
    print('\n\nSample tweet by female:')
    print(female_tweets[0])
    c.write('\n\nSample tweet by female:')
    c.write(female_tweets[0])
    print('\n\nSample tweet by unknown:')
    print(unknown_tweets[0])
    c.write('\n\nSample tweet by unknown:')
    c.write(unknown_tweets[0])
        
#     print('gender labels:', Counter(y))
    print('avg accuracy', do_cross_val(X, y, 5))

if __name__ == '__main__':
    main()