import sys
import csv

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding

from gensim.models import Word2Vec


class WordEmbeddings:
    def __init__(self, model):
        self.model = model
        self.vocab = list(model.vocab)

    def matrix(self):
        # TODO Memoize this?
        return np.array([self.model[word] for word in self.vocab])


class CNN:
    def __init__(self, vocab, initial_embeddings=None, embedding_dimension=None):
        if initial_embeddings is not None:
            embedding_dimension = initial_embeddings.shape[1]

        # TODO Variable naming ...
        self.index = self.create_word_to_vocab_index_mapping(vocab)

        self.network = self.build_network(initial_embeddings, embedding_dimension)

    def create_word_to_vocab_index_mapping(self, vocab):
        return {word: i for (i, word) in enumerate(vocab)}

    def build_network(self, initial_embeddings, embedding_dimension):
        embedding = Embedding(input_dim=len(self.index), output_dim=embedding_dimension, weights=[initial_embeddings] if initial_embeddings is not None else None)

        model = Sequential([embedding])
        # TODO Don't use strings here
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        return model

    def tweet_to_indices(self, tweet):
        return [self.index[word] for word in tweet if word in self.index]


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

    # TODO Make it so that this does not need to be kept in memory
    embeddings = WordEmbeddings(Word2Vec.load(embeddings_path))
    cnn = CNN(embeddings.vocab, initial_embeddings=embeddings.matrix())
    print(cnn.network.predict(np.array([cnn.tweet_to_indices(positive_tweets[0])])).shape)


if __name__ == '__main__':
    main()
