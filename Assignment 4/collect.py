"""
collect.py
"""

import networkx as nx
from TwitterAPI import TwitterAPI
import sys
import time
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import json

consumer_key = 'kZpmC8xJBOY65K6BjVsgsRbyz'
consumer_secret = 'gUStSCswnlu00PMU8pheIcDKDccKm2n4XRKKgbmX8ZtBx1H3G7'
access_token = '1010522881405652993-tTYzliBv7BTsB0E1VoM7uqiHbw0D0Q'
access_token_secret = 'JsOeymj1wm7YV62s4vXB0C3GfQUHiW9Dm4jNIuTCYkFoW'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.
    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.
    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    candidates = []
    cad = []
    with open(filename,"r") as f:
        for column in [line.split() for line in f]:
            cad.append(column)      
        #print (cad)
        
        for i in cad:
            for j in i:
                candidates.append(j)
    #print (candidates)
    return candidates

def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)
    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup
    In this example, I test retrieving two users: twitterapi and twitter.
    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    file = open('names.txt', 'w')
    result = dict()
    for user in screen_names:
        request = robust_request(twitter, "friends/list", {'screen_name': user, 'count' :10})
        names = list()
        for friend in request:
            names.append(friend['screen_name'])
        result[user] = names
    
    for user in screen_names:
        for friend in result[user]:
            #print(friend)
            request1 = robust_request(twitter, "friends/list", {'screen_name': friend, 'count' :20})
            friend_names = list()
            for name in request1:
                friend_names.append(name['screen_name'])
            result[friend] = friend_names
    print(result)
    for k in result:
        for v in result[k]:
            file.write(k + ":" + v+ "\n")
    file.close() 
    
    return result

def get_tweets(twitter, screen_names):
    tweets = list()
    limit = 200
    request = robust_request(twitter, 'statuses/filter', {'track':'F1','language':'en','locations':'-74,40,-73,41'})
    for tweet in request:
        tweets.append(tweet)
        if len(tweets) == limit:
            break
    pickle.dump(tweets, open('tweets.pkl', 'wb'))    
    return tweets

def main():
    #print("Data Imported")
    twitter = get_twitter()
    #print("Twitter Information fetched")
    screen_names = ['F1']
    user = []
    c = open('collect.txt', 'w')
    #print('Established Twitter connection.')
    #print('Read screen names:\n%s' % screen_names)
    data_user = get_users(twitter, screen_names)
    for k in data_user:
        for v in data_user[k]:
            user.append(v)
    print(len(user))
    c.write('Number of users collected: %d' %len(user))
    c.write('\n')
    
    #print (data_user)
    #save_obj(data_user, 'twit_user')
    #print("users info saved.")
    tweets = get_tweets(twitter, screen_names[0])
    print (len(tweets))
    c.write('\n\nNumber of messages collected: %d' %len(tweets))
    c.close()
    for t in tweets:
        print(t)
    
    
    #save_obj(tweets, 'tweets')
    #print("%d tweets saved." % (len(tweets)))


if __name__ == '__main__':
    main()

