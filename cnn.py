import sys
import os.path
import csv

import numpy as np

from theano.tensor.nnet import softmax

from keras.models import Graph, model_from_yaml
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD
from keras.objectives import categorical_crossentropy


def create_index(vocabulary):
    return {word: i + 1 for (i, word) in enumerate(vocabulary)}


# TODO Does this live in the right scope?
def one_max_pooling(x):
    # TODO This import business is wonky
    from theano.tensor import max
    return max(x, 1)


class CNN:
    def __init__(self, vocabulary):
        self.index = create_index(vocabulary)
        self.network = None

    def tweet_to_indices(self, tweet):
        return [self.index[word] for word in tweet if word in self.index]

    # TODO Make the argument list better
    def build_network(self, initial_embeddings, embedding_dimension, filter_sizes_and_counts):

        # TODO Is this idiomatic?
        assert len(filter_sizes_and_counts) > 0


        self.network = Graph()
        self.network.add_input(name='input', input_shape=(None,), dtype='int')  # TODO 'int' should not be a string
        self.network.add_node(name='embedding',
                              layer=Embedding(input_dim=len(self.index),
                                              output_dim=embedding_dimension,
                                              weights=[initial_embeddings] if initial_embeddings is not None else None ),
                              input='input')

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
        with open(os.path.join(basedir, 'model.yml'), 'w') as model_file:
            model_file.write(self.network.to_json())
        self.network.save_weights(os.path.join(basedir, 'weights.h5'))

    def load(self, basedir):
        # TODO What if the index does not match the vocabulary in the model files?
        with open(os.path.join(basedir, 'model.yml'), 'r') as model_file:
            self.network = model_from_yaml(model_file.read())
            # TODO Do we have to compile the model again, here?
            self.network.load_weights(os.path.join(basedir, 'weights.h5'))

