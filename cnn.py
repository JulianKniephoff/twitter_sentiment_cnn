# TODO docopt?

# TODO Create and test the embedding layer

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
        return np.array([self.model[word] for word in self.vocab])


class CNN:
    def __init__(self, vocab, initial_word_embeddings='uniform', embedding_dimension=None):
        self.vocab = vocab
        self.network = self.build_network(initial_word_embeddings=initial_word_embeddings, embedding_dimension=embedding_dimension)

        self.index = self.create_word_to_vocab_index_mapping()

    def create_word_to_vocab_index_mapping(self):
        return {word: i for (i, word) in enumerate(self.vocab)}

    def build_network(self, initial_word_embeddings, embedding_dimension):
        embedding = Embedding(input_dim=len(self.vocab), output_dim=embedding_dimension, init=initial_word_embeddings)

        model = Sequential([embedding])
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        return model

    def tweet_to_indices(self, tweet):
        return [self.index[word] for word in tweet if word in self.vocab]


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

    embeddings = WordEmbeddings(Word2Vec.load(embeddings_path))
    cnn = CNN(embeddings.vocab, initial_word_embeddings=embeddings.matrix())
    cnn.network.predict([cnn.tweet_to_indices(positive_tweets[0])t ])

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
