import csv


def positive_integer(x):
    integer = int(x)
    if integer <= 0:
        raise ValueError('{} is not a positive integer'.format(x))
    return integer

def rate(x):
    rate = float(x)
    if rate < 0 or rate > 1:
        raise ValueError('{} is not a rate'.format(x))
    return rate
