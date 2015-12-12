import csv

tweets_path = '/daten/DCNNPreprocessings/all_fixed.csv'
with open(tweets_path) as tweets_file:
    tweets = list()
    vocabulary = set()
    for row in csv.reader(tweets_file):
        tweet = row[1:]
        tweets.append(tweet)
        for word in tweets:
            vocabulary.add(word)

    print(len(tweets), 'Tweets')
    print(len(vocabulary), 'words')
