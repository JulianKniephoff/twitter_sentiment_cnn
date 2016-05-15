from util import parse_tweets

from collections import namedtuple

import argparse
from argparse import ArgumentParser
from argtypes import positive_integer

import numpy as np

from sklearn import cross_validation
from sklearn.metrics import precision_score, recall_score, f1_score

from cnn import CNN, LabeledTweet


def parse_args():
    parser = ArgumentParser('Evaluate a CNN')

    # TODO Validation
    parser.add_argument('-m', '--model',
                        required=True)

    parser.add_argument('-d', '--dataset',
                        required=True,
                        type=argparse.FileType('r'))

    parser.add_argument('-b', '--batch-size',
                        default=50,
                        type=positive_integer)
    parser.add_argument('-e', '--epochs',
                        default=1,
                        type=positive_integer)

    return parser.parse_args()


EvaluationResult = namedtuple('EvaluationResult', ['p', 'r'])


# TODO Should we really assume that tweets is just a list?
def evaluate(model, train_tweets, test_tweets, train_labels, test_labels, batch_size, epochs):
    cnn = CNN()
    cnn.load(model)

    cnn.fit_generator(
        # TODO Ugh, really? Maybe just use or wrap fit?
        lambda: (LabeledTweet(label=label, tweet=tweet) for label, tweet in zip(train_labels, train_tweets)),
        batch_size=batch_size,
        nb_epoch=epochs,
        samples_per_epoch=len(train_tweets)
    )
    predictions = cnn.predict(test_tweets)
    predicted_labels = [p.argmax() for p in predictions['output']]

    return EvaluationResult(
        p=precision_score(test_labels, predicted_labels, average=None),
        r=recall_score(test_labels, predicted_labels, average=None)
    )


def cross_validate(model, dataset, epochs, batch_size):
    test_tweets = list(parse_tweets(dataset.name))  # TODO It sucks that this reopens the file
    n = len(test_tweets)

    texts = np.array([labeled_tweet.tweet for labeled_tweet in test_tweets])
    labels = np.array([labeled_tweet.label for labeled_tweet in test_tweets])

    cv = cross_validation.KFold(n, 10)
    for train, test in cv:
        scores = evaluate(
            model,
            texts[train], texts[test],
            labels[train], labels[test],
            batch_size, epochs
        )
        print('precision: ', scores.p)
        print('recall: ', scores.r)

    #print(precision_score(true_labels, predicted_labels))
    #print(recall_score(true_labels, predicted_labels))
    #print(f1_score(true_labels, predicted_labels))


def main():
    args = parse_args()
    cross_validate(**vars(args))


if __name__ == '__main__':
    main()