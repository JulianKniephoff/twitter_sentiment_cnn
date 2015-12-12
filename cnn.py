import sys
import csv

try:
    tweets_path = sys.argv[1]
except IndexError:
    print("Usage: cnn.py <tweets_file>")
    sys.exit(1)

with open(tweets_path) as tweets_file:
    tweets = list()
    vocabulary = set()
    for row in csv.reader(tweets_file):
        tweet = row[1:]
        tweets.append(tweet)
        for word in tweet:
            vocabulary.add(word)

    print(len(tweets), 'Tweets')
    print(len(vocabulary), 'words')
