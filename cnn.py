# TODO docopt

import sys
import csv

import numpy as np
from gensim.models import Word2Vec


def parse_tweets(path):
    with open(path) as tweets_file:
        tweets = list()
        vocabulary = set()
        for row in csv.reader(tweets_file):
            tweet = row[1:]
            tweets.append(tweet)
            for word in tweet:
                vocabulary.add(word)

        return tweets, vocabulary


def main():
    try:
        positive_tweets_path = sys.argv[1]
        negative_tweets_path = sys.argv[2]
        embeddings_path = sys.argv[3]
    except IndexError:
        print('Usage: cnn.py <positive_tweets_file> <negative_tweets_file> <word2vec_model>')
        sys.exit(1)

    # Load tweets and vocabulary
    positive_tweets, positive_vocabulary = parse_tweets(positive_tweets_path)
    negative_tweets, negative_vocabulary = parse_tweets(negative_tweets_path)

    tweets = positive_tweets + negative_tweets
    # The vocabulary has to become a list since we need to keep track of the order
    vocabulary = list(positive_vocabulary | negative_vocabulary)

    # Extract initial weights for the embedding layer
    word2vec_embeddings = Word2Vec.load(embeddings_path)
    weights = np.array(map(lambda word: word2vec_embeddings[word], vocabulary))
    print(weights)


if __name__ == '__main__':
    main()
