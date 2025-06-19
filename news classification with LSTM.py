import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")




news=fetch_20newsgroups(subset="all")
x,y=news.data,news.target
tokenizer=Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x)
x_seq=tokenizer.texts_to_sequences(x)
x_pad=pad_sequences(x_seq,maxlen=100)
label_enc=LabelEncoder()
#y_encoded=label_enc.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x_pad,y,test_size=0.3,random_state=42)


def build_model():
    model=Sequential()
    
    model.add(Embedding(input_dim=100000,output_dim=64,input_length=100))
    model.add(LSTM(units=64,return_sequences=False))
    model.add(Dropout(.5))
    model.add(Dense(20,activation="softmax"))
    
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
model=build_model()
model.summary()

early=EarlyStopping(monitor="val_accuracy",patience=3,restore_best_weights=True)
history=model.fit(x_train,y_train,
                  epochs=10,
                  batch_size=64,
                  validation_split=.1,
                  callbacks=[early])
loss,accuracy=model.evaluate(x_test,y_test)


plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"],marker="*",label="t_loss")
plt.plot(history.history["val_loss"],marker="*",label="val_loss")
plt.title("training and validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.grid("true")
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],marker="o",label="accuarcy")
plt.plot(history.history["val_accuracy"],marker="o",label="val_accuracy")
plt.title("training and validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid("true")

plt.show()




































