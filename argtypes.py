import yaml
from argparse import ArgumentTypeError

from gensim.models import Word2Vec

from . import util


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
        raise ArgumentTypeError(
            'Argument needs to be a number in the interval [0, 1]'
        )
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
                raise ArgumentTypeError(
                    '{} needs to be a positive integer'.format(name)
                )
    return configuration


def word2vec_model(argument):
    try:
        return Word2Vec.load(argument)
    # TODO Catch more specific exceptions
    except:
        raise ArgumentTypeError(
            'Could not read embeddings from {}'.format(argument)
        )
