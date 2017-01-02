import csv


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
