def positive_integer(x):
    integer = int(x)
    if integer <= 0:
        raise ValueError('%s is not a positive integer' % repr(x))
    return integer

