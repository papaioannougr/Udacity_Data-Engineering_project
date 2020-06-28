import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    Function which is used to read data about messages and their categorization, which are in 2 separate source files
    messages_filepath:   It indicates sources file which includes messages (use brackets when define this as input in function save_data)
    categories_filepath: It indicates sources file which includes the categorization of messages (use brackets when define this as input in function save_data)
    
    This function will return a dataframe 'data_fr' which is the merge of the 2 input sources files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge these two Dataframes
    data_fr = messages.merge(categories, how='outer',on='id')
    
    return data_fr

    


def clean_data(df):
    """
    Function which is used to clean data about categories
    df: The data where clean actions are going to be applied

    This function will return a dataframe 'data_fr' which is the cleaned version of input dataFrame df
    """
    # Drop column 'categories' and replace that with one column for every individual category existing in the initial column.
    df = pd.concat([df, df['categories'].str.split(";",expand=True)], axis=1).drop(['categories'],  axis=1)
    
    # Create a list with names which will replace df's column names
    category_colnames = df.iloc[0][4:].str.split("-",expand=True)[0].tolist()
    for i in range(0,4):
        category_colnames.insert(0, df.columns[0:4].tolist()[::-1][i])
    
        
    # Rename the columns in df. New columns which were created from splitted categories column, are namned after the corresponding names in category_colnames
    df.columns = category_colnames
    
    # Keep the last digit of the values in the recently renamed columns. These values are binaries {0,1}. We transform them into integers
    for column in df.loc[:,df.columns[4:]]:
        # set each value to be the last character of the string
        df[column] = df[column].str.strip().str[-1] 
        # convert column from string to numeric
        df[column] = df[column].astype(int)
    
    data_fr=df
    
    return data_fr
    
    

def save_data(df, database_filename):
    """
    This Function saves Dataframe: df 
    into an sqlite database
    df:                It is the DataFrame which has been gone through clean_data function
    database_filename: It is the database file name (use brackets when define this as input in function save_data)
    
    ***Function creates a database which name name is predefined as Project_database
    """    
    #conn = sqlite3.connect(database_filename) 
    #df.to_sql("DataFrame_generated", conn, if_exists="replace", index=False)    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DataFrame_generated', engine,  if_exists="replace", index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()