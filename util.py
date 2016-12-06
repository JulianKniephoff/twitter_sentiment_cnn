import csv

from .cnn import LabeledTweet


def positive_integer(x):
    integer = int(x)
    if integer <= 0:
        # TODO Is `repr` the right thing here? Do you even need it?
        # TODO Use `format`
        raise ValueError('{} is not a positive integer'.format(x))
    return integer

def rate(x):
    rate = float(x)
    if rate < 0 or rate > 1:
        # TODO Is `repr` the right thing here? Do you even need it?
        # TODO Use `format`
        raise ValueError('{} is not a rate'.format(x))
    return rate


def parse_tweets(filename):
    with open(filename, newline='') as file:
        for i, row in enumerate(csv.reader(file)):
            yield LabeledTweet(tweet=row[2:], label=int(row[0]))
