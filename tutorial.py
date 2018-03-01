#!/usr/bin/env python3

import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("labeledTrainData.tsv",header = 0, delimiter = "\t", quoting = 3)
#print(train.shape)
#print(train.columns.values)
#example1 = BeautifulSoup(train["review"][0])
#letters_only = re.sub("[^a-zA-Z]"," ", example1.get_text())
#print(letters_only)
#print(train["review"][0])
#print(example1.get_text())
#lower_case = letters_only.lower()
#words = lower_case.split()
#print(stopwords.words("english"))
#words = [w for w in words if not w in stopwords.words("english")]
#print(words)


def review_to_words(raw_review):
    #1.Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #2.Remove non-letters
    letters_only = re.sub("[^a-zA-Z]"," ", review_text)
    #3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #6.Join the words back into one string separated by space, and return the result.
    return(" ".join(meaningful_words))

#clean_review = review_to_words(train["review"][0])
#print(clean_review)
'''
for r in train["review"]:
    clean_review = review_to_words(r)
    print("%s" % clean_review)
'''


'''
#Get the number of reviews based on the dataframe column size num_reviews = train["review"].size
num_reviews = train["review"].size
#Initialize an empty list to hold the clean reviews
clean_train_reviews = []
#Loop over each review; create an index i that goes from 0 to the length of the movie review list
print("Cleaning and parsing the training set movie reviews...\n")
with open('clean_train_reviews','w') as f:
     for i in range(0, num_reviews):
         #Call our function for each one, and add the result to the list of clean reviews
         if((i+1)%1000 == 0):
            print("Review %d of %d\n" % (i+1,num_reviews))
         #clean_train_reviews.append(review_to_words(train["review"][i]))
         print(review_to_words(train["review"][i]),file = f)
'''

clean_train_reviews = []
with open('clean_train_reviews','r') as f:
    for review in f.readlines():
        #review = f.read()
        clean_train_reviews.append(review)
#print(len(clean_train_reviews))

print("Creating the bag of words..\n")
#Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = None,max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
#Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
#print(vocab)
#Sum up the counts of each vocabulary word

dist = np.sum(train_data_features,axis = 0)
#For each, print the vocabulary word and the number of times it
#appears in the training set
'''
for tag, count in zip(vocab,dist):
    print(count,tag)
'''
#print (train_data_features.shape)

print("Training the randow forest...")
#Initialize a Random Forest Classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)
#Fit the forest to the sentiment labels as the response variable
#This may take a few minutes to run
forest = forest.fit(train_data_features, train["sentiment"])

#Read the test data
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)
#Verify that there are 25000 rows and 2 columns
print(test.shape)
#Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

print("CLeaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if((i+1)%1000 == 0):
        print("Review %d of %d\n" % (i+1,num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

#Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

#use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

#Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
output = pd.DataFrame(data = {"id":test["id"],"sentiment":result})

#Use pandas to write the comma-separated output file
output.to_csv("Bag_of_Words_model.csv",index = False, quoting = 3)

