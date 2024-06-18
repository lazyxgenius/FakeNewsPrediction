This repository contains a Python script that uses logistic regression to predict whether a news article is real or fake. 
The script performs the following steps:

  Data Preprocessing: The script preprocesses the text data by removing non-alphabet characters, converting to lowercase, splitting into words, stemming the words, and removing stopwords.
  
  Feature Extraction: The script converts the preprocessed text data into numerical features using TF-IDF vectorization.
  
  Model Training: The script trains a logistic regression model on the preprocessed data.
  
  Model Evaluation: The script evaluates the model's performance on both the training and test data.
  
  Model Saving: The script saves the trained model to a file.
  
  Predictive System: The script uses the saved model to make predictions on new, unseen data.
