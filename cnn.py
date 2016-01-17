import sys
from os import path
import csv

import numpy as np

from keras.models import Graph
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D

from gensim.models import Word2Vec


def create_index(vocabulary):
    return {word: i for (i, word) in enumerate(vocabulary)}


def one_max_pooling(X):
    import theano.tensor as T  # TODO Is it really necessary to reimport this here?
    return T.max(X, 1)


def build_network(vocabulary, initial_embeddings, embedding_dimension, filter_sizes_and_counts):
    # TODO Put name first always
    network = Graph()
    network.add_input(name='input', input_shape=(None,), dtype='int')  # TODO 'int' should not be a string
    network.add_node(layer=Embedding(input_dim=len(vocabulary),
                                     output_dim=embedding_dimension,
                                     weights=[initial_embeddings] if initial_embeddings is not None else None ),
                     name='embedding',
                     input='input')

    # TODO Ensure that there is at least one element in filter_sizes_and_counts
    filters = []
    for size, count in filter_sizes_and_counts:
        # TODO Use sequential containers here?
        network.add_node(layer=Convolution1D(count, size),
                         name='convolution-%d' % size,
                         input='embedding')
        network.add_node(layer=Lambda(one_max_pooling,
                                      # TODO We should not have to specify the output shape
                                      output_shape=(count,)),
                         name='max-pooling-%d' % size,
                         input='convolution-%d' % size)
        filters.append('max-pooling-%d' % size)

    # TODO Use sequential containers here, too
    network.add_node(layer=Dense(2, activation='softmax'),  # TODO Don't use strings here, either
                     name='softmax',
                     inputs=filters)  # TODO What if we only have one input? This does not work then.

    network.add_output(name='output',
                       input='softmax')

    # TODO Don't use strings here
    # TODO Are these actually the parameters we want?
    network.compile(optimizer='sgd', loss={'output': 'categorical_crossentropy'})
    return network


class CNN:
    def __init__(self, vocabulary, initial_embeddings=None, embedding_dimension=None, filter_sizes_and_counts=[]):
        if initial_embeddings is not None:
            embedding_dimension = initial_embeddings.shape[1]

        self.index = create_index(vocabulary)

        self.network = build_network(vocabulary, initial_embeddings, embedding_dimension, filter_sizes_and_counts)

    def tweet_to_indices(self, tweet):
        return [self.index[word] for word in tweet if word in self.index]

    def save(self, basedir):
        with open(path.join(basedir, 'model.json'), 'w') as model_file:
            model_file.write(self.network.to_json())
        self.network.save_weights(path.join(basedir, 'weights.h5'))

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
    # embeddings = WordEmbeddings(Word2Vec.load(embeddings_path))
    # cnn = CNN(embeddings.vocab, initial_embeddings=embeddings.matrix())
    # print(cnn.network.predict(np.array([cnn.tweet_to_indices(positive_tweets[0])])).shape)


if __name__ == '__main__':
    main()
