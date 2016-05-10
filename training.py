import util

import argparse
import csv
import yaml

from gensim.models import Word2Vec

from cnn import CNN


def positive_integer(string):
    try:
        integer = util.positive_integer(string)
    except ValueError:
        raise argparse.ArgumentTypeError('Argument needs to be a positive integer')
    return integer


def filter_configuration(string):
    try:
        configuration = yaml.load(string)
    except yaml.ParserError:
        raise argparse.ArgumentTypeError('Not a valid YAML string')

    for size, count in configuration.items():
        for name, value in [('size', size), ('count', count)]:
            try:
                util.positive_integer(value)
            except ValueError:
                raise argparse.ArgumentTypeError('%s needs to be a positive integer' % name)
    return configuration


def word2vec_model(string):
    try:
        return Word2Vec.load(string)
    # TODO Catch more specific exceptions
    except:
        raise argparse.ArgumentTypeError('Could not read embeddings from %s' % string)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CNN')
    # TODO More validations for these parameters?
    parser.add_argument('-p', '--positive',
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument('-u', '--unclear',
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument('-n', '--negative',
                        type=argparse.FileType('r'),
                        required=True)

    embedding_arguments = parser.add_mutually_exclusive_group(required=True)
    embedding_arguments.add_argument('-d', '--dimension',
                                     type=positive_integer)
    embedding_arguments.add_argument('-e', '--embeddings',
                                     type=word2vec_model)

    parser.add_argument('-f', '--filters',
                        type=filter_configuration,
                        required=True)

    parser.add_argument('-c', '--epochs',
                        type=positive_integer,
                        default=1)
    parser.add_argument('-b', '--batch',
                        type=positive_integer,
                        default=50)

    # TODO We should ensure that this directory exists or create it
    parser.add_argument('-o', '--output',
                        required=True)

    return parser.parse_args()


def extract_vocabulary(tweets):
    return set([word for tweet in tweets for word in tweet])


def parse_tweets(file):
    return [row[1:] for row in csv.reader(file)]


def train(positive, unclear, negative, dimension, embeddings, filter_configuration, epochs, batch_size):
    # TODO Perform validation here, too
    positive_tweets = parse_tweets(positive)
    unclear_tweets = parse_tweets(unclear)
    negative_tweets = parse_tweets(negative)

    vocabulary = extract_vocabulary(positive_tweets + unclear_tweets + negative_tweets)
    cnn = CNN()

    cnn.build_network(vocabulary, embeddings, dimension, filter_configuration, 3)

    cnn.fit([positive_tweets, unclear_tweets, negative_tweets], nb_epoch=epochs, batch_size=batch_size)

    return cnn


def main():
    args = parse_args()
    cnn = train(args.positive, args.unclear, args.negative, args.dimension, args.embeddings, args.filters, args.epochs, args.batch)
    cnn.save(args.output)


if __name__ == '__main__':
    main()

