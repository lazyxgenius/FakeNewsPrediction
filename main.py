#importing the dependencies
import joblib
import numpy as np
import pandas as pd
import re  # searching text in document
from nltk.corpus import stopwords  # stopwords are thos words that doesnt add much value to text
from nltk.stem.porter import PorterStemmer  # gives us the root word for a particular word
from sklearn.feature_extraction.text import TfidfVectorizer  # used to convert trext to feature vector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk

nltk.download('stopwords')

print(stopwords.words('english'))

# data collection
fn = pd.read_csv(r"C:\Users\Aditya PC\PycharmProjects\FakeNewsPrediction\train.csv")
fn.head()

print(fn.shape)

# counting the number of missing values
fn.isnull().sum()

# replacing the null values with empty string
fn = fn.fillna('')

# merging title and author and text column
fn['content'] = fn['author'] + ' ' + fn['title'] + ' ' + fn['text']

print(fn['content'])

# Initialize the PorterStemmer
port_stem = PorterStemmer()


# Define the stemming function
def stemming(content):
    # Replace non-alphabet characters with a space
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    # Split into words
    stemmed_content = stemmed_content.split()
    # Stem the words and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    # Join the words back into a single string
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# Apply the stemming function to the content column
fn['content'] = fn['content'].apply(stemming)

print(fn['content'])

# separating the data and label
X = fn['content'].values
Y = fn['label'].values

print(X)

print(Y)

# converting text data to numerical
vectorizer = TfidfVectorizer()
# TFIDF - term frequency inverse document freq - count a number of times a word is being used in a document
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

# splitting train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# STRATIFY MEANS SPLIT y EQUALLY

# training the model
model = LogisticRegression()

model.fit(X_train, Y_train)

# Evaluation
# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of training data : ", training_data_accuracy)

# Evaluation
# accuracy score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score of test data : ", test_data_accuracy)

filename = "saved_FNP_LR.joblib"
joblib.dump(model, filename)


# Making a Predictive System
input1 = X[0]  # first news will be taken
# model = joblib.load('saved_FNP_LR.joblib')
prediction = model.predict(input1)
print(prediction)

if prediction[0] == 0:
    print("News is real")
else:
    print("News is fake")

print(Y[0])  # to check if the prediction is right

