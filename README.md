# SMS Spam Detection using Naive Bayes Classifier

This project demonstrates how to train a machine learning model for detecting spam messages in SMS data using the Naive Bayes classifier.

## Prerequisites

To run this project, you need the following software:

* Python 3.6+
* Jupyter Notebook
* Pandas
* Scikit-learn
* NLTK

## Dataset

The SMS Spam Collection dataset contains a set of SMS tagged messages that have been collected for SMS Spam research. It can be downloaded from https://archive.ics.uci.edu/ml/datasets/sms+spam+collection. The dataset consists of two columns: "label" and "message". The "label" column indicates whether the SMS is spam or not, and the "message" column contains the text of the SMS.

## Data Preprocessing

The first step is to clean and preprocess the data. The following steps are performed for each SMS message:

* Remove all non-alphabetic characters.
* Convert all characters to lowercase.
* Tokenize the message into individual words.
* Remove all stop words.
* Stem the remaining words.

## Creating the Bag of Words Model

The Bag of Words model is used to represent each SMS message as a vector of word frequencies. The CountVectorizer class from the Scikit-learn library is used to create the Bag of Words model. It is initialized with a maximum vocabulary size of 2500 words.

## Training the Model

The Naive Bayes classifier is used to train the model. The MultinomialNB class from Scikit-learn is used for this purpose.

## Testing the Model

The trained model is tested on a test set of SMS messages. The accuracy and confusion matrix of the model are calculated using Scikit-learn's accuracy_score and confusion_matrix functions.

## Conclusion

This project demonstrates how to train a machine learning model for SMS spam detection using the Naive Bayes classifier. The trained model achieved a high accuracy in detecting spam messages.
