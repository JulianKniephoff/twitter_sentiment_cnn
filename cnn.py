from collections import namedtuple
from os import path, makedirs
import json

import numpy as np

from theano.tensor.nnet import softmax
from keras.models import Graph, model_from_yaml
from keras.layers.core import Dense, Lambda, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.optimizers import Adagrad
from keras.objectives import categorical_crossentropy
from keras.preprocessing.sequence import pad_sequences


def _create_index(vocabulary):
    return {word: i for (i, word) in enumerate(vocabulary)}


# TODO Does this live in the right scope?
def _one_max_pooling(x):
    # TODO This import business is wonky
    from theano.tensor import max
    return max(x, 1)


# Wrap a `Lambda` layer with a specific function
# NOTE This is necessary to deserialize this layer
class _OneMaxPooling(Lambda):
    def __init__(self, count, **kwargs):
        # Count represents the output shape
        # TODO `count` is not really a good name
        # NOTE This has to live in a different attribute, though,
        #   since `output_shape` is not properly deserialized
        self.count = count
        # TODO Why do we have to specify the `output_shape` at all?
        super(_OneMaxPooling, self).__init__(
            function=_one_max_pooling,
            output_shape=(self.count,),
            **kwargs
        )

    def get_config(self):
        config = super(_OneMaxPooling, self).get_config()
        # Add `count` to the config so that it gets serialized
        # alongside the rest of the configuration
        config['count'] = self.count
        return config


class CNN:
    def __tweets_to_indices(self, tweets):
        return pad_sequences(
            [
                [self.__index[word] for word in tweet.tokens if word in self.__index]
                for tweet in tweets
            ],
            # The maximum number of tokens in a 140 character string
            maxlen=70,
            value=self.__padding_index,
            padding='post'
        )

    def __prepare_labeled_tweets(self, tweets):
        def output_for_class(class_number):
            output = [0] * self.__classes
            output[class_number] = 1
            return output

        return {
            'input': self.__tweets_to_indices(
                labeled_tweet.tweet for labeled_tweet in tweets
            ),
            # TODO Is the list needed here
            #   or is a generator sufficient?
            'output': np.array([
                output_for_class(labeled_tweet.label)
                for labeled_tweet in tweets
            ])
        }

    def classes(self):
        return self.__classes

    def build_network(
            self,
            initial_embeddings,
            vocabulary_size,
            filter_configuration,
            dropout_rate,
            activation,
            # TODO Get rid of this default parameter
            classes=2
    ):
        if not filter_configuration:
            raise ValueError('There needs to be at least one filter')
        if not initial_embeddings:
            raise ValueError ("We need pretrained word embeddings")

        vocabulary = sorted(
            initial_embeddings.vocab,
            key=lambda word: initial_embeddings.vocab[word].count,
            reverse=True
        )[:vocabulary_size]

        self.__index = _create_index(vocabulary)
        # There is no need for an explicit padding symbol
        # in the index or vocabulary
        self.__padding_index = len(vocabulary)

        self.__network = Graph()
        # TODO 'int' should not be a string
        self.__network.add_input(name='input', input_shape=(None,), dtype='int')

        initial_weights = [np.array(
            [initial_embeddings[word] for word in vocabulary] +
            [np.zeros(initial_embeddings.vector_size)]
        )]

        embedding_layer = Embedding(
            input_dim=len(self.__index) + 1,  # + 1 for padding
            output_dim=initial_embeddings.vector_size,
            weights=initial_weights
        )
        self.__network.add_node(
            name='embedding',
            layer=embedding_layer,
            input='input'
        )

        filters = []
        for size in filter_configuration:
            # TODO Use sequential containers here?
            #   The question is then: Do we need to access them later on
            #   and how do we do that?
            count = filter_configuration[size]
            convolution = Convolution1D(count, size, activation=activation)
            # TODO Use format
            self.__network.add_node(
                name='convolution-%d' % size,
                layer=convolution,
                input='embedding'
            )
            pooling = _OneMaxPooling(count=count)
            self.__network.add_node(
                name='max-pooling-%d' % size,
                layer=pooling,
                input='convolution-%d' % size
            )
            filters.append('max-pooling-%d' % size)

        # TODO Use sequential containers here, too
        if len(filters) is 1:
            inputs = {'input': filters[0]}
        else:
            inputs = {'inputs': filters}

        if dropout_rate:
            dropout_layer = Dropout(p=dropout_rate)
            self.__network.add_node(
                name='dropout',
                layer=dropout_layer,
                concat_axis=1,  # Work around a Theano bug
                **inputs
            )
            inputs = {'input': 'dropout'}

        # TODO This should be `softmax` instead of `'softmax'` IMO,
        #   but I got an error in `save`:
        #   > `AttributeError: 'Softmax' object has no attribute '__name__'`
        self.__classes = classes
        dense_layer = Dense(self.__classes, activation='softmax')
        self.__network.add_node(
            name='dense',
            layer=dense_layer,
            concat_axis=1,  # Work around a Theano bug
            **inputs
        )

        self.__network.add_output(name='output', input='dense')

        # TODO Are these actually the parameters we want?
        self.__network.compile(
            optimizer=Adagrad(),
            loss={'output': categorical_crossentropy}
        )

    def save(self, basedir):
        makedirs(basedir, exist_ok=True)

        with open(path.join(basedir, 'model.yml'), 'w') as model_file:
            model_file.write(self.__network.to_yaml())
        self.__network.save_weights(
            path.join(basedir, 'weights.h5'),
            overwrite=True
        )
        with open(path.join(basedir, 'index.json'), 'w') as index_file:
            json.dump(self.__index, index_file)

    def load(self, basedir):
        with open(path.join(basedir, 'model.yml'), 'r') as model_file:
            self.__network = model_from_yaml(
                model_file.read(),
                custom_objects={'_OneMaxPooling': _OneMaxPooling}
            )

        self.__network.load_weights(path.join(basedir, 'weights.h5'))

        with open(path.join(basedir, 'index.json'), 'r') as index_file:
            self.__index = json.load(index_file)
            self.__padding_index = len(cnn.__index)
            self.__classes = cnn.__network.outputs['output'].output_dim

    def fit_generator(self, generator_generator, batch_size, *args, **kwargs):
        # TODO This should not be a closure ...
        #   Maybe this should not even be here ...
        def infinite_generator():
            while True:
                epoch_generator = generator_generator()
                yield from epoch_generator

        generator = infinite_generator()

        def tweet_generator():
            # TODO This seems redundant. Can we compose generators somehow?
            while True:
                yield self.__prepare_labeled_tweets(
                    [next(generator) for _ in range(batch_size)]
                )

        self.__network.fit_generator(
            tweet_generator(),
            *args, **kwargs
        )

    def predict(self, tweets, *args, **kwargs):
        return self.__network.predict(
            {'input': self.__tweets_to_indices(tweets)},
            *args, **kwargs
        )['output']
