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

        # In case we are getting an iterator, we collect it here
        # since we iterate over it twice
        tweets = list(tweets)
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

    @classmethod
    def build(
            cls,
            embeddings,
            vocabulary_size,
            filters,
            dropout,
            activation,
            # TODO Get rid of this default parameter
            classes=2
    ):
        cnn = cls()

        vocabulary = sorted(
            embeddings.vocab,
            key=lambda word: embeddings.vocab[word].count,
            reverse=True
        )[:vocabulary_size]

        cnn.__index = _create_index(vocabulary)
        # NOTE This is not actually a valid index into the vocabulary.
        #   We don't actually need an explicit padding symbol anywhere.
        cnn.__padding_index = len(vocabulary)

        cnn.__network = Graph()
        # TODO 'int' should not be a string
        cnn.__network.add_input(name='input', input_shape=(None,), dtype='int')

        initial_weights = [np.array(
            [embeddings[word] for word in vocabulary] +
            [np.zeros(embeddings.vector_size)]
        )]

        embedding_layer = Embedding(
            input_dim=len(cnn.__index) + 1,  # + 1 for padding
            output_dim=embeddings.vector_size,
            weights=initial_weights
        )
        cnn.__network.add_node(
            name='embedding',
            layer=embedding_layer,
            input='input'
        )

        filter_outputs = []
        for size in filters:
            count = filters[size]
            convolution = Convolution1D(count, size, activation=activation)
            # TODO Use format
            cnn.__network.add_node(
                name='convolution-%d' % size,
                layer=convolution,
                input='embedding'
            )
            pooling = _OneMaxPooling(count=count)
            cnn.__network.add_node(
                name='max-pooling-%d' % size,
                layer=pooling,
                input='convolution-%d' % size
            )
            filter_outputs.append('max-pooling-%d' % size)

        if len(filter_outputs) is 1:
            inputs = {'input': filter_outputs[0]}
        else:
            inputs = {'inputs': filter_outputs}

        if dropout:
            dropout_layer = Dropout(p=dropout)
            cnn.__network.add_node(
                name='dropout',
                layer=dropout_layer,
                concat_axis=1,  # Work around a Theano bug
                **inputs
            )
            inputs = {'input': 'dropout'}

        # TODO This should be `softmax` instead of `'softmax'` IMO,
        #   but I got an error in `save`:
        #   > `AttributeError: 'Softmax' object has no attribute '__name__'`
        cnn.__classes = classes
        dense_layer = Dense(cnn.__classes, activation='softmax')
        cnn.__network.add_node(
            name='dense',
            layer=dense_layer,
            concat_axis=1,  # Work around a Theano bug
            **inputs
        )

        cnn.__network.add_output(name='output', input='dense')

        # TODO Are these actually the parameters we want?
        cnn.__network.compile(
            optimizer=Adagrad(),
            loss={'output': categorical_crossentropy}
        )

        return cnn


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

    @classmethod
    def load(cls, basedir):
        cnn = cls()
        with open(path.join(basedir, 'model.yml'), 'r') as model_file:
            cnn.__network = model_from_yaml(
                model_file.read(),
                custom_objects={'_OneMaxPooling': _OneMaxPooling}
            )

        cnn.__network.load_weights(path.join(basedir, 'weights.h5'))

        with open(path.join(basedir, 'index.json'), 'r') as index_file:
            cnn.__index = json.load(index_file)
            cnn.__padding_index = len(cnn.__index)
            cnn.__classes = cnn.__network.outputs['output'].output_dim

        return cnn

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
