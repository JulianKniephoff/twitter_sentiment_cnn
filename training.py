from util import parse_tweets

import argparse
from argparse import ArgumentParser
from argtypes import positive_integer, rate, filter_configuration, word2vec_model

from cnn import CNN


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

