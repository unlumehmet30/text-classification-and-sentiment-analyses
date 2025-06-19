# -*- coding: utf-8 -*-
"""
Created on Sun May 25 12:36:32 2025

@author: mhmtn
"""

#import libs
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#data loading
df=pd.read_csv(r"C:\Users\mhmtn\Downloads\spam.csv",encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
df.columns=["label","text"]
#EDA
print(df.isna().sum()) # there is no null
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
text=list(df.text)
lemmatizer=WordNetLemmatizer()
corpus=[]
for i in range(len(text)):
    r=re.sub("[^a-zA-Z]"," ",text[i]) #remove all things except letter
    r=r.lower() #lower case
    r=r.split()
    r=[word for word in r if word not in stopwords.words("english")]
    r=[lemmatizer.lemmatize(word) for word in r]
    r=" ".join(r)
    corpus.append(r)
#model 
df["text2"]=corpus
x=df["text2"]
y=df["label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)

cv=CountVectorizer()
x_train_cv=cv.fit_transform(x_train)
x_test_cv=cv.transform(x_test)
#training and evaulate 
dt=DecisionTreeClassifier()
dt.fit(x_train_cv,y_train)
prediction=dt.predict(x_test_cv)
c_matrix=confusion_matrix(prediction,y_test)
score=dt.score(x_test_cv,y_test)






