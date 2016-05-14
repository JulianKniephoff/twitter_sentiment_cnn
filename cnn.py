from collections import namedtuple, OrderedDict

import os.path
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


LabeledTweet = namedtuple('LabeledTweet', ['tweet', 'label'])


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
        self.dense_layer = None
        self.dropout_layer = None
        self.padding_index = None
        self.classes = None

    def tweets_to_indices(self, tweets):
        return pad_sequences(
            [
                [self.index[word] for word in tweet if word in self.index]
                for tweet in tweets
            ],
            maxlen=70,  # 70 is the maximum number of tokens in a 140 character string
            value=self.padding_index,
            padding='post'
        )

    def build_network(self,
                      initial_embeddings,
                      filter_configuration,
                      vocabulary_size=None,
                      dropout_rate=None,
                      activation='linear',
                      classes=2):

        if not filter_configuration:
            raise ValueError('There needs to be at least one filter')
        if not initial_embeddings:
            raise ValueError ("We need pretrained word embeddings")

        vocabulary = sorted(
            initial_embeddings.vocab,
            key=lambda word: initial_embeddings.vocab[word].count,
            reverse=True
        )[:vocabulary_size]

        self.index = create_index(vocabulary)
        # There is no need for an explicit padding symbol in the index or vocabulary
        self.padding_index = len(vocabulary)

        self.network = Graph()
        self.network.add_input(name='input', input_shape=(None,), dtype='int')  # TODO 'int' should not be a string

        initial_weights = [np.array(
            [initial_embeddings[word] for word in vocabulary] +
            [np.zeros(initial_embeddings.vector_size)]
        )]

        self.embedding_layer = Embedding(input_dim=len(self.index) + 1,  # + 1 for padding
                                         output_dim=initial_embeddings.vector_size,
                                         weights=initial_weights)
        self.network.add_node(name='embedding',
                              layer=self.embedding_layer,
                              input='input')

        filters = []
        for size in filter_configuration:
            # TODO Use sequential containers here?
            # The question is then: Do we need to access them later on and how do we do that?
            count = filter_configuration[size]
            convolution = Convolution1D(count, size, activation=activation)
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
        if len(filters) is 1:
            inputs = {'input': filters[0]}
        else:
            inputs = {'inputs': filters}

        if dropout_rate:
            self.dropout_layer = Dropout(p=dropout_rate)
            self.network.add_node(name='dropout',
                                  layer=self.dropout_layer,
                                  **inputs)
            inputs = {'input': 'dropout'}

        # TODO This should be `softmax` instead of `'softmax'` IMO, but I got an error in `save`:
        # AttributeError: 'Softmax' object has no attribute '__name__'
        self.classes = classes
        self.dense_layer = Dense(self.classes, activation='softmax')
        self.network.add_node(name='dense',
                              layer=self.dense_layer,
                              **inputs)

        self.network.add_output(name='output',
                                input='dense')

        # TODO Are these actually the parameters we want?
        self.network.compile(optimizer=Adagrad(), loss={'output': categorical_crossentropy})

    def fit_generator(self, generator_generator, batch_size, *args, **kwargs):
        # TODO This should not be a closure ...
        #   Maybe this should not even be here ...
        def infinite_generator():
            while True:
                epoch_generator = generator_generator()
                yield from epoch_generator

        generator = infinite_generator()

        def labeled_tweets_to_keras(tweets):
            def output_for_class(class_number):
                output = [0] * self.classes
                output[class_number] = 1
                return output

            return {
                'input': self.tweets_to_indices(
                    labeled_tweet.tweet for labeled_tweet in tweets
                ),
                'output': np.array(
                    [output_for_class(labeled_tweet.label) for labeled_tweet in tweets]
                )
            }

        def tweet_generator():
            while True:  # TODO This seems redundant. Can we compose generators somehow?
                yield labeled_tweets_to_keras(
                    [next(generator) for _ in range(batch_size)]
                )

        self.network.fit_generator(
            tweet_generator(),
            *args, **kwargs
        )

    #def predict(self, tweets, *args, **kwargs):
    #    return self.network.predict(
    #        {'input': self.tweets_to_indices(tweets)},
    #        *args, **kwargs
    #    )

    def save(self, basedir):
        # TODO Create `basedir` if it does not exist
        with open(os.path.join(basedir, 'model.yml'), 'w') as model_file:
            model_file.write(self.network.to_yaml())
        # NOTE Maybe use `overwrite=True`
        self.network.save_weights(os.path.join(basedir, 'weights.h5'), overwrite=True)
        with open(os.path.join(basedir, 'index.json'), 'w') as index_file:
            json.dump(self.index, index_file)

    def load(self, basedir):
        # TODO What if the index does not match the vocabulary in the model files?
        with open(os.path.join(basedir, 'model.yml'), 'r') as model_file:
            self.network = model_from_yaml(model_file.read(), custom_objects={'OneMaxPooling': OneMaxPooling})
        self.network.load_weights(os.path.join(basedir, 'weights.h5'))
        with open(os.path.join(basedir, 'index.json'), 'r') as index_file:
            self.index = json.load(index_file)

