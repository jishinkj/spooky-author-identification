#importing libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk import word_tokenize

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import *   

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import seaborn as sns
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import pickle

from time import time

#%%
from classification_code_class import Classification

#%% Load the datasets

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

#%% Dimensions of both the datasets

train.shape 
# (19579, 3)

test.shape
# (8392, 2)

#%% Store response to a variable and concat train and test

train_df = train.iloc[:,:-1]
y_true = train.iloc[:,-1]


df = pd.concat([train_df, test], axis = 0)

#%% CHECK FOR MISSING VALUES AND DROP THEM

df.isnull().sum()
# nomissing values in either of the datasets

#%% CHECK CLASS BALANCE

train['author'].value_counts()
# =============================================================================
# EAP    7900
# MWS    6044
# HPL    5635
# =============================================================================

#%%
# to check the percentage of each class 
100*train['author'].value_counts()/len(train.author)

# =============================================================================
# EAP    40.349354
# MWS    30.869809
# HPL    28.780837
# =============================================================================

# visualize class imbalance
sns.countplot(train.author)

#%% Shuffling

from sklearn.utils import shuffle
df = shuffle(df, random_state = 7)

#%% SEPARATE DATAFRAMES FOR EACH SENTIMENT

# positivedf = df.loc[df["sentiment"] == 'positive']
# negativedf = df.loc[df["sentiment"] == 'negative']
# neutraldf = df.loc[df["sentiment"] == 'neutral']

#%% 
#buitinstopwords of nltk library
builtinstopwords = stopwords.words('english')


#words needed for sentiment analysis
neededWords = ['mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
                'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
                'won', "won't", 'wouldn', "wouldn't",'aren', "aren't", 'couldn', "couldn't", 
                'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
                "haven't", 'isn', "isn't",'don', "don't", 'should', "should've", 'no', 'nor', 'not'] 

for word in neededWords:
    if word in builtinstopwords: 
        builtinstopwords.remove(word)
#print (builtinstopwords)

#%% Generate Word Cloud for each of the authors, if relevant. Check other kernels

#%%
#lm = WordNetLemmatizer()

def clean(content):
    '''Code to clean text'''
    temptext = re.sub('[^a-zA-Z]', ' ', str(content))
    temptext = temptext.lower()
    tokens = nltk.word_tokenize(temptext)
    #tokens = [word for word in tokens if word not in set(builtinstopwords)] 
    cleanbody= [lm.lemmatize(word) for word in tokens if not word in set(builtinstopwords)]
    return (str(cleanbody)[1:-1])

def exec_time(start, end):
    if (end - start) <= 60:
        print("Total Execution time was {} seconds".format(end - start))
    else:
        print("Total Execution time was {} minutes".format((end - start)/60))

#%% 
# TODO - check the difference in time when lemmatizer is instantiated inside the clean function vs outside
lm = WordNetLemmatizer()
df['cleaned']=df['text'].apply(lambda x : clean(x))
df.reset_index(drop=True,inplace=True)

#%%
# FIT THE TFIDF VECTORIZER AND PICKLE THE VOCAB

#tfidf_obj = TfidfVectorizer(max_df=0.5,min_df=0.01,use_idf=True)
tfidf_obj = TfidfVectorizer(max_features = 5000)
X_train_tfidf = tfidf_obj.fit_transform(df.cleaned)
# X_train_tfidf.shape

# feature_list = tfidf_obj.vocabulary_
# feature_list

#%%
#converting the array to a dataframe
X_train_test = pd.DataFrame(X_train_tfidf.todense(), columns = tfidf_obj.get_feature_names())

#y = df2['sentimentbinary']

X = X_train_test.iloc[:train.shape[0],:]

y = y_true

test_df = X_train_test.iloc[train.shape[0]:,:]
#%%
# TODO - Convert binaryReplacement to dictionary from epiphany code 
def binaryReplacement(variable):
    '''converting output categorical variable to numeric'''
    if variable == 'EAP':
        encoded_value = 0
    elif variable == 'HPL':
        encoded_value = 1
    else:
        encoded_value = 2
    return (encoded_value)

y = train['author'].apply(lambda x : binaryReplacement(x))

#%%
#del df, train, test, final_df, 
#%%

#analysis(predictionwithsampling(RandomOverSampler,RandomForestClassifier()))

#%% Instantiate the class, creae instance called 'class_obj'
from classification_code_class import Classification

start = time()
class_obj = Classification()
class_obj.analysis(X, y, RandomOverSampler, RandomForestClassifier())
class_obj.plot_roc()
end = time()
exec_time(start, end)

#%%
param_grid = { 
   'n_estimators': [2, 5],
   'criterion' : ['gini', 'entropy']
}

start = time()
best_parameters = class_obj.grid_search(param_grid) # returns best parameter grid 
print("Best Parameters are :", best_parameters)
end = time()
exec_time(start, end)
#%% Re-train the classifier with new hyper-parameters

class_obj.analysis(X, y, RandomOverSampler, RandomForestClassifier())
#class_obj.pickle_classifier()
class_obj.plot_roc()
#%%
y_pred_submission = class_obj._predict(test_df)

#%%

# param_grid = { 
#    'n_estimators': [2, 5, 10, 15,20,50,100],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'max_depth' : [4,5,6,7,8],
#    'criterion' :['gini', 'entropy']
# }
    
#%%
    


#%% convert to csv
sub_df = pd.DataFrame()
sub_df['id'] = test['id']
sub_df['EAP'] = y_pred_submission[:,0]
sub_df['HPL'] = y_pred_submission[:,1]
sub_df['MWS'] = y_pred_submission[:,2]

sub_df.to_csv("submissions/v7.csv", index = False, header=True)

