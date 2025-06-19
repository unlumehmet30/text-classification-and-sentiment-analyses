# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:42:19 2025

@author: mhmtn
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,roc_curve,auc
import kerastuner as kt
from kerastuner.tuners import RandomSearch
import warnings
warnings.filterwarnings("ignore")


#veriseti hazırlama
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)

max_len=100
x_train=pad_sequences(x_train,maxlen=max_len)
x_test=pad_sequences(x_test,maxlen=max_len)


#model  
def build(hp):
    model=Sequential()
    model.add(Embedding(input_dim=10000,
                        output_dim=hp.Int("embedding_output",min_value=32,
                                          max_value=128,step=32),
                        input_length=max_len))
    model.add(SimpleRNN(units=hp.Int("rnn_units",min_value=32,
                                     max_value=128,step=32)))
    model.add(Dropout(rate=hp.Float("dropout_rate",min_value=0.2,max_value=.5,step=0.1)))
    model.add(Dense(1,activation="sigmoid"))
    
    
    
    
    model.compile(optimizer=hp.Choice("optimizer",["adam","rmsprop"]),
                  loss="binary_crossentropy", 
                  metrics=["accuracy","auc"])
    
    return model
 
    
#best parameters
tuner=RandomSearch(
    build,
    objective="val_loss",
    max_trials=5,
    executions_per_trial=1,
    directory="RNN_tuner",
    project_name="imdb_rnn"
    )
early=EarlyStopping(monitor="val_loss",patience=5,
                    restore_best_weights=True)
tuner.search(x_train,y_train,
             epochs=5,
             validation_split=.2,
             callbacks=[early]
             )
#evaluate
best_model=tuner.get_best_models(num_models=1)[0]
loss,accuracy,auc_=best_model.evaluate(x_test,y_test)


print(f"test loss:{loss}\ntest accuracy:{accuracy:.4f}\ntest auc:{auc_:.4f}")

y_pred_prob=best_model.predict(x_test)
y_pred=(y_pred_prob>0.5).astype("int32")

print(classification_report(y_test, y_pred))


fpr,tpr,_=roc_curve(y_test, y_pred_prob)
roc_auc=auc(fpr,tpr)


plt.figure()
plt.plot(fpr,tpr,color="r",lw=2,label="roc_curve (area=%0.2f"% roc_auc)
plt.plot([0,1],[0,1],color="b",lw=2,linestyle="--")
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("receiver operating characteristic")
plt.legend()
plt.show()










































            