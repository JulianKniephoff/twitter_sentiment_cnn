import os.path
import json

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


# Wrap a `Lambda` layer with a specific function
# NOTE This is necessary to deserialize this layer
class OneMaxPooling(Lambda):
    def __init__(self, count, **kwargs):
        # Count represents the output shape
        # TODO `count` is not really a good name
        # NOTE This has to live in a different attribute, though, since `output_shape` is not properly deserialized
        self.count = count
        # TODO Why do we have to specify the `output_shape` at all?
        super(OneMaxPooling, self).__init__(function=one_max_pooling, output_shape=(self.count,), **kwargs)

    def get_config(self):
        config = super(OneMaxPooling, self).get_config()
        # Add `count` to the config so that it gets serialized alongside the rest of the configuration
        config['count'] = self.count
        return config


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
        self.embedding_layer = Embedding(input_dim=len(self.index) + 1,
                                         output_dim=embedding_dimension,
                                         weights=[initial_embeddings] if initial_embeddings is not None else None)
        self.network.add_node(name='embedding',
                              layer=self.embedding_layer,
                              input='input')

        filters = []
        for size, count in filter_sizes_and_counts:
            # TODO Use sequential containers here?
            # The question is then: Do we need to access them later on and how do we do that?
            convolution = Convolution1D(count, size)
            self.network.add_node(name='convolution-%d' % size,
                                  layer=convolution,
                                  input='embedding')
            pooling = OneMaxPooling(count=count)
            self.network.add_node(name='max-pooling-%d' % size,
                                  layer=pooling,
                                  input='convolution-%d' % size)
            self.convolutions.append(convolution)
            self.pools.append(pooling)
            filters.append('max-pooling-%d' % size)

        # TODO Use sequential containers here, too
        if len(filters) == 1:
            inputs = {'input': filters[0]}
        else:
            inputs = {'inputs': filters}
        # TODO This should be `softmax` instead of `'softmax'` IMO, but I got an error in `save`:
        # AttributeError: 'Softmax' object has no attribute '__name__'
        self.output = Dense(classes, activation='softmax')
        self.network.add_node(name='softmax',
                              layer=self.output,
                              **inputs)

        self.network.add_output(name='output',
                                input='softmax')

        # TODO Are these actually the parameters we want?
        self.network.compile(optimizer=SGD(), loss={'output': categorical_crossentropy})

    def save(self, basedir):
        # TODO Create `basedir` if it does not exist
        with open(os.path.join(basedir, 'model.yml'), 'w') as model_file:
            model_file.write(self.network.to_yaml())
        # NOTE Maybe use `overwrite=True`
        self.network.save_weights(os.path.join(basedir, 'weights.h5'))
        with open(os.path.join(basedir, 'index.json'), 'w') as index_file:
            json.dump(self.index, index_file)

    def load(self, basedir):
        # TODO What if the index does not match the vocabulary in the model files?
        with open(os.path.join(basedir, 'model.yml'), 'r') as model_file:
            self.network = model_from_yaml(model_file.read(), custom_objects={'OneMaxPooling': OneMaxPooling})
        self.network.load_weights(os.path.join(basedir, 'weights.h5'))
        with open(os.path.join(basedir, 'index.json'), 'r') as index_file:
            self.index = json.load(index_file)

