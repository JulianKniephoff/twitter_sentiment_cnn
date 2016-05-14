import util

import argparse
from argparse import ArgumentParser, ArgumentTypeError
import csv
import yaml

from gensim.models import Word2Vec

from cnn import CNN, LabeledTweet


def positive_integer(argument):
    try:
        integer = util.positive_integer(argument)
    except ValueError:
        raise ArgumentTypeError('Argument needs to be a positive integer')
    return integer


def rate(argument):
    try:
        rate = util.rate(argument)
    except ValueError:
        raise ArgumentTypeError('Argument needs to be a number in the interval [0, 1]')
    return rate


def filter_configuration(argument):
    try:
        configuration = yaml.load(argument)
    except yaml.ParserError:
        raise ArgumentTypeError('Not a valid YAML string')

    for size, count in configuration.items():
        for name, value in [('size', size), ('count', count)]:
            try:
                util.positive_integer(value)
            except ValueError:
                raise ArgumentTypeError('%s needs to be a positive integer' % name)
    return configuration


def word2vec_model(argument):
    try:
        return Word2Vec.load(argument)
    # TODO Catch more specific exceptions
    except:
        raise ArgumentTypeError('Could not read embeddings from %s' % argument)


def parse_args():
    parser = ArgumentParser(description='Train a CNN')
    # TODO More validations for these parameters?
    parser.add_argument('-t', '--dataset',
                        type=argparse.FileType('r'),
                        required=True)

    # TODO Making a subcommand mandatory currently requires setting a `dest
    #   and setting the `required` property.
    #   I don't think this is intended behavior
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    load_subparser = subparsers.add_parser('load')
    load_subparser.add_argument('model')  # TODO Validation

    new_subparser = subparsers.add_parser('new')
    new_subparser.add_argument('-e', '--embeddings',
                               type=word2vec_model,
                               required=True)

    new_subparser.add_argument('-f', '--filters',
                               type=filter_configuration,
                               required=True)

    new_subparser.add_argument('-v', '--vocabulary-size',
                               type=positive_integer)

    new_subparser.add_argument('-d', '--dropout-rate',
                               type=rate)

    # TODO Validate this
    new_subparser.add_argument('-a', '--activation',
                               default='linear',
                               help='default: %(default)s')

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


def train(dataset, embeddings, model, vocabulary_size, filters, dropout_rate, activation, epochs, batch_size):
    tweet_count = sum(1 for tweet in parse_tweets(dataset))

    cnn = CNN()
    if model:
        print('loading preexisting model')
        cnn.load(model)
    else:
        print('building network')
        cnn.build_network(
            embeddings,
            filters,
            vocabulary_size=vocabulary_size,
            dropout_rate=dropout_rate,
            classes=3
        )

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
        # TODO Wow, what a hack. O_o
        args.embeddings if args.command == 'new' else None,
        args.model if args.command == 'load' else None,
        args.vocabulary_size if args.command == 'new' else None,
        args.filters if args.command == 'new' else None,
        args.dropout_rate if args.command == 'new' else None,
        args.activation if args.command == 'new' else None,
        args.epochs,
        args.batch
    )
    cnn.save(args.output)


if __name__ == '__main__':
    main()

