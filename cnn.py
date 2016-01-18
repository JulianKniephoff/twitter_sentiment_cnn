import sys
import os.path
import csv

import numpy as np
from theano.tensor.nnet import softmax

from keras.models import Graph
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD
from keras.objectives import categorical_crossentropy

from gensim.models import Word2Vec


def create_index(vocabulary):
    return {word: i for (i, word) in enumerate(vocabulary)}


class CNN:
    def __init__(self, vocabulary):
        self.index = create_index(vocabulary)
        self.network = None

    def tweet_to_indices(self, tweet):
        return [self.index[word] for word in tweet if word in self.index]

    # TODO Make the argument list better
    def build_network(self, initial_embeddings, embedding_dimension, filter_sizes_and_counts):

        def one_max_pooling(x):
            from theano.tensor import max
            return max(x, 1)

        self.network = Graph()
        self.network.add_input(name='input', input_shape=(None,), dtype='int')  # TODO 'int' should not be a string
        self.network.add_node(name='embedding',
                              layer=Embedding(input_dim=len(self.index),
                                              output_dim=embedding_dimension,
                                              weights=[initial_embeddings] if initial_embeddings is not None else None ),
                              input='input')

        # TODO Ensure that there is at least one element in filter_sizes_and_counts
        filters = []
        for size, count in filter_sizes_and_counts:
            # TODO Use sequential containers here?
            # The question is then: Do we need to access them later on and how do we do that?
            self.network.add_node(name='convolution-%d' % size,
                                  layer=Convolution1D(count, size),
                                  input='embedding')
            self.network.add_node(name='max-pooling-%d' % size,
                                  layer=Lambda(one_max_pooling,
                                               # TODO We should not have to specify the output shape
                                               output_shape=(count,)),
                                  input='convolution-%d' % size)
            filters.append('max-pooling-%d' % size)

        # TODO Use sequential containers here, too
        if len(filters) == 1:
            inputs = {'input': filters[0]}
        else:
            inputs = {'inputs': filters}
        self.network.add_node(name='softmax',
                              layer=Dense(2, activation=softmax),
                              **inputs)

        self.network.add_output(name='output',
                                input='softmax')

        # TODO Are these actually the parameters we want?
        self.network.compile(optimizer=SGD(), loss={'output': categorical_crossentropy})

    def save(self, basedir):
        with open(os.path.join(basedir, 'model.json'), 'w') as model_file:
            model_file.write(self.network.to_json())
        self.network.save_weights(os.path.join(basedir, 'weights.h5'))


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
