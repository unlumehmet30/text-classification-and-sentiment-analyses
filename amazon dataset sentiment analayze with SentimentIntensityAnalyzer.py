# -*- coding: utf-8 -*-
"""
Created on Sun May 25 15:26:36 2025

@author: mhmtn
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


df=pd.read_csv(r"C:\Users\mhmtn\Downloads\amazon.csv")
lemmatizer=WordNetLemmatizer()
def clean_text(text):
    token_list=word_tokenize(text.lower())
    filtered=[token for token in token_list if token not in stopwords.words("english")]
    lemmatized=[lemmatizer.lemmatize(token)for token in filtered]
    text=" ".join(lemmatized)
    return text
df["review2"]=df["reviewText"].apply(clean_text)

analyzer=SentimentIntensityAnalyzer()
def get_sentiment(text):
    score=analyzer.polarity_scores(text)
    sentiment=1 if score["compound"]>0.1 else 0
    return sentiment
df["sentiment"]=df["review2"].apply(get_sentiment)

c_matrix=confusion_matrix(df["Positive"], df["sentiment"])
plt.figure(figsize=(5,4))
sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()
cr=classification_report(df["Positive"], df["sentiment"])
print(f"classification report:{cr}")

































        