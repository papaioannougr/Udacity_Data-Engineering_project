# Udacity_Data-Engineering_project
This repository contains all the necessary data and code files related with the project of the 3rd  Udacity - DataScience nanodegree part.

CONTENTS:

1) ETL Pipeline Preparation.ipynb : A python notebook where code ETL Pipeline Prparation was tested and optimized
2) messages.csv & categories.csv: Data files which were used in ETL Pipeline Preparation.ipynb
3) ML Pipeline Preparation.ipynb : A python notebook where ML Pipeline was tested and optimized
4) README.md: Contents of repository & instructions to run the application

Folders:

A) data:

  1) disaster_messages.csv & disaster_categories.csv: data files which will be loaded during ETL pipeline
  2) process_data.py: ETL pipeline which loads, cleans and save the data into a database
  3) DisasterResponse.db: I'ts the databse where the data are stored after going through the ETL pipelines
  
B) models:
  1) train_classifier.py: ML pipeline which loads data, tokenize the messages, setup a pipeline of transformers and predictors,
                          deploys a model, evaluates the results and save the model in a pickle format
  2) classifier.pkl: the saved model from ML pipeline
  
C) app:
  1) Folder: templates:
      i) go.html & master.html: application templates to categorize & visualize the inserted messages 
  2) run.py: It's application's exeution code
  
  
  INSTRUCTIONS:
  
  1) Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
          python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves
          python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        
  2) Run the following command in the app's directory to run your web app.
      app/python run.py
      
  3) On a new terminal execute command:
      env|grep WORK
      
  4) From results of the previous execution, use SPACEID  &  SPACEDOMAIN and go to the following page:
      https://SPACEID-3001.SPACEDOMAIN
  
