from collections import OrderedDict

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
from keras.preprocessing.sequence import pad_sequences


def create_index(vocabulary):
    return {word: i for (i, word) in enumerate(vocabulary)}


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
    def __init__(self):
        self.index = None
        self.network = None
        self.embedding_layer = None
        self.convolutions = []
        self.pools = []
        self.output = None
        self.padding_index = None

    def tweets_to_indices(self, tweets):
        return pad_sequences(
            [
                [self.index[word] for word in tweet if word in self.index]
                for tweet in tweets
            ],
            maxlen=70,  # 70 is the maximum number of tokens in a 140 character string
            value=self.padding_index
        )

    # TODO Make the argument list better
    def build_network(self,
                      vocabulary=None,
                      initial_embeddings=None,
                      embedding_dimension=None,
                      filter_configuration=None,
                      classes=2):

        if not filter_configuration:
            raise ValueError('There needs to be at least one filter')

        if initial_embeddings:
            # TODO Shouldn't this just be `.dimension`?
            # TODO Should we complain if there was an explicit embedding dimension?
            embedding_dimension = initial_embeddings.vector_size

            # TODO See above; don't rely on the interface of `gensim.models.Word2Vec`
            vocabulary = initial_embeddings.index2word + list(vocabulary)
            # TODO This is not very elegant
            vocabulary = OrderedDict((v, None) for v in vocabulary).keys()
        else:
            if not embedding_dimension:
                raise ValueError('Either an embedding dimension or a set of initial embeddings must be given')

        # There is no need for an explicit padding symbol in the index or vocabulary
        self.index = create_index(vocabulary)
        self.padding_index = len(self.index)

        self.network = Graph()
        self.network.add_input(name='input', input_shape=(None,), dtype='int')  # TODO 'int' should not be a string
        self.embedding_layer = Embedding(input_dim=len(self.index) + 1,  # + 1 for padding
                                         output_dim=embedding_dimension)
        self.network.add_node(name='embedding',
                              layer=self.embedding_layer,
                              input='input')

        # HACK The given initial embeddings might not contain some of the vocabulary items
        #   So we initialize everything uniformly and then override the vectors we know.
        #   This way, the unknown embeddings are still randomized
        if initial_embeddings:
            embedding_weights = self.embedding_layer.get_weights()[0]
            for index, word in enumerate(initial_embeddings.index2word):
                embedding_weights[index] = initial_embeddings[word]
            self.embedding_layer.set_weights([embedding_weights])

        filters = []
        for size in filter_configuration:
            # TODO Use sequential containers here?
            # The question is then: Do we need to access them later on and how do we do that?
            count = filter_configuration[size]
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

    def fit(self, classes, *args, **kwargs):
        def output_for_class(class_number):
            output = [0] * len(classes)
            output[class_number] = 1
            return output

        self.network.fit(
            {
                'input': np.concatenate(tuple(
                    self.tweets_to_indices(class_) for class_ in classes
                )),
                'output': np.array(
                    [output_for_class(class_number) for class_number, class_ in enumerate(classes) for tweet in class_]
                )
            },
            *args, **kwargs
        )

    def predict(self, tweets, *args, **kwargs):
        return self.network.predict(
            {'input': self.tweets_to_indices(tweets)},
            *args, **kwargs
        )

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

