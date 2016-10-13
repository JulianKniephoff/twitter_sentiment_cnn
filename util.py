import csv

from cnn import LabeledTweet


def positive_integer(x):
    integer = int(x)
    if integer <= 0:
        raise ValueError('%s is not a positive integer' % repr(x))
    return integer

def rate(x):
    rate = float(x)
    if rate < 0 or rate > 1:
        raise ValueError('%s is not a rate' % repr(x))
    return rate


def parse_tweets(filename):
    with open(filename, newline='') as file:
        for i, row in enumerate(csv.reader(file)):
            yield LabeledTweet(tweet=row[2:], label=int(row[0]))
