import util

from itertools import chain

import argparse
from argparse import ArgumentParser, ArgumentTypeError
import csv
import yaml

from gensim.models import Word2Vec

from cnn import CNN, LabeledTweet


def positive_integer(string):
    try:
        integer = util.positive_integer(string)
    except ValueError:
        raise ArgumentTypeError('Argument needs to be a positive integer')
    return integer


def filter_configuration(string):
    try:
        configuration = yaml.load(string)
    except yaml.ParserError:
        raise ArgumentTypeError('Not a valid YAML string')

    for size, count in configuration.items():
        for name, value in [('size', size), ('count', count)]:
            try:
                util.positive_integer(value)
            except ValueError:
                raise ArgumentTypeError('%s needs to be a positive integer' % name)
    return configuration


def word2vec_model(string):
    try:
        return Word2Vec.load(string)
    # TODO Catch more specific exceptions
    except:
        raise ArgumentTypeError('Could not read embeddings from %s' % string)


def parse_args():
    parser = ArgumentParser(description='Train a CNN')
    # TODO More validations for these parameters?
    parser.add_argument('-t', '--dataset',
                        type=argparse.FileType('r'),
                        required=True)

    parser.add_argument('-e', '--embeddings',
                        type=word2vec_model,
                        required=True)

    parser.add_argument('-f', '--filters',
                        type=filter_configuration,
                        required=True)

    parser.add_argument('-c', '--epochs',
                        type=positive_integer,
                        default=1,
                        help='default: %(default)s')
    parser.add_argument('-b', '--batch',
                        type=positive_integer,
                        default=50,
                        help='default: %(default)s')

    # TODO We should ensure that this directory exists or create it
    parser.add_argument('-o', '--output',
                        required=True)

    return parser.parse_args()


def parse_tweets(filename):
    with open(filename) as file:
        for i, row in enumerate(csv.reader(file)):
            if i % 100 == 0:
                pass  # print('Read %d tweets' % i)
            yield LabeledTweet(tweet=row[2:], label=int(row[0]))


def train(dataset, embeddings, filters, epochs, batch_size):
    tweet_count = sum(1 for tweet in parse_tweets(dataset))

    print('building network')
    cnn = CNN()
    cnn.build_network(embeddings, filters, classes=3)

    print('training')
    # We have to read the file here, again, possibly multiple times
    # the previous iterator does not work anymore
    cnn.fit_generator(
        lambda: parse_tweets(dataset),  # TODO This is ugly
        nb_epoch=epochs,
        batch_size=batch_size,
        samples_per_epoch=tweet_count
    )

    return cnn


def main():
    args = parse_args()
    cnn = train(
        args.dataset.name,
        args.embeddings,
        args.filters,
        args.epochs,
        args.batch
    )
    cnn.save(args.output)


if __name__ == '__main__':
    main()

