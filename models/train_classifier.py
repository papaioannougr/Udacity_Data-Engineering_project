import sys
import pandas as pd
import numpy as np
import pickle
import sqlite3
from sqlalchemy import create_engine
import re


import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report


def load_data(database_filepath):
    """
    Function which is used to read the data which is saved in a sqlite database and give back X, Y data for the ML model, as long as category names
    of Y 
    database_filepath:   It indicates the sqlite database where the data are going to feed the ML process are stored.
                         Entries are going be like: 'sqlite:///DatabaseName.db', where DatabaseName.db is the relative sqlite database
    
    This function will return X, Y and category_names The independent and dependent variables for the ML pipeline and the names of Y categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DataFrame_generated", engine)
    X = df['message']
    Y = df.drop(columns=['id','genre','original','message'],axis=1)
    category_names=Y.columns
    return X,Y,category_names


def tokenize(text):
    """
    This function takes the input text and it will transform it into tokens.
    text: The next which will be transformed into tokesns ()
    
    This function will return the transformation of the fed text into a list of tokens with the same name (tokens), 
    """
    # normalize case and remove punctuation   
    text =re.sub(r"[^a-zA-Z0-9]", " ",text.lower())
    
    # Tokenize text
    tokens=word_tokenize(text)
    
    #Initialiaze important functions for stop words and the lemmatizer type of Normalization od text
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens
    


def build_model():
    """
    This function defines and return a pipeline object which transforms text message in usable features and fit a ML model on data.
    
    """
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize, ngram_range=(1, 1), max_features=500)),
        ('tfidf',TfidfTransformer(use_idf=False)),
        ('clf',MultiOutputClassifier(KNeighborsClassifier(n_neighbors=4,)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function uses the declared ML procedure in order to apply and evaluate the results on the test sample
     
    model:  Modelling procedure which has been applied on train data and it is used evaluate results on test data
    X_test, Y_test: They are the corresponding independent and dependent variables from the test sample which are used to evaluate the model
    category_names: They define the set of dependent variables upon which the model is evaluated
    
    Function produces results   
   
    """
    #Create the predicted values after applying the model on test sample
    y_pred = pd.DataFrame(model.predict(X_test),columns=category_names)
    
    #Print results
    for column in category_names:
        print("Classification report for {}:".format(column),"\n")
        print(classification_report(Y_test[column], y_pred[column]))
        print("\n")
        print("Labels for {}:".format(column),np.unique(y_pred[column]))
        print("\n")
        print("Confusion Matrix for {}:".format(column),"\n")
        print(confusion_matrix(Y_test[column], y_pred[column], labels=np.unique(y_pred[column])))
        print("\n")
        accuracy=(Y_test[column].reset_index(drop=True)==y_pred[column].reset_index(drop=True)).mean()
        print("Accuracy of prediction for {}:".format(column),accuracy)
        print("\n")
        print("----------------------------------------------------")
        print("\n")


def save_model(model, model_filepath):
    """
    This function stores locally an existing model as a pickle file in order this model can be uploaded in any time and be used again.
    
    Model: It is the name of the model that has already been deployed andd it is going to be saved from this procedure
    model_filepath: It is the full path of the location where the model is saved, including the the name of the saved model and its extension: *.pkl
    """
    Pkl_Filename = model_filepath
    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()