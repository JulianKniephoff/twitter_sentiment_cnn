# TODO docopt?

# TODO Create and test the embedding layer

import sys
import csv

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding

from gensim.models import Word2Vec


class CNN:
    def __init__(self, model=None, vocab=None):
        self.model = model
        self.vocab = vocab or self.extract_vocab_from_model()
        self.index = self.create_word_to_vocab_index_mapping()
        if model:
            self.initial_weights = self.extract_initial_weights_from_model()

    def extract_vocab_from_model(self):
        assert self.model
        return list(self.model.vocab)

    def tweet_to_indices(self, tweet):
        return [self.index[word] for word in tweet if word in self.vocab]

    def create_word_to_vocab_index_mapping(self):
        return {word: i for (i, word) in enumerate(self.vocab)}

    def extract_initial_weights_from_model(self):
        try:
            self.initial_weights = np.array([self.model[word] for word in self.vocab])
        except KeyError:
            raise AssertionError("The given vocabulary and that of the model disagree")


def parse_tweets(path):
    with open(path) as tweets_file:
        return [row[1:] for row in csv.reader(tweets_file)]


def main():
    try:
        positive_tweets_path = sys.argv[1]
        negative_tweets_path = sys.argv[2]
        embeddings_path = sys.argv[3]
    except IndexError:
        print('Usage: cnn.py <positive_tweets_file> <negative_tweets_file> <word2vec_model>')
        sys.exit(1)

    # Load tweets and vocabulary
    positive_tweets = parse_tweets(positive_tweets_path)
    negative_tweets = parse_tweets(negative_tweets_path)

    cnn = CNN(Word2Vec.load(embeddings_path))
    print(cnn.initial_weights)

    # Extract initial weights for the embedding layer
    # word2vec_embeddings = Word2Vec.load(embeddings_path)
    # weights = list()
    # for word in vocabulary:
    #    try:
    #        weights.append(word2vec_embeddings[word])
    #    except KeyError:
    #        pass

    ## TODO There might be stuff in here that we do not have embeddings for
    # weights = np.array(map(lambda word: word2vec_embeddings[word], vocabulary))
    # print(weights)


if __name__ == '__main__':
    main()
