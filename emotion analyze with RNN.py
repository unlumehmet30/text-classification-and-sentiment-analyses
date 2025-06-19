# -*- coding: utf-8 -*-
"""
Created on Sat May 24 13:21:21 2025

@author: mhmtn
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data={
    "texts":[
    "Yemekler çok lezzetliydi.",
    "Garsonlar çok ilgisizdi.",
    "Ambiyans harikaydı, çok beğendim.",
    "Servis berbattı, tekrar gelmem.",
    "Tatlılar taptazeydi.",
    "Masalar çok pisti.",
    "Fiyatlar uygun ve kaliteliydi.",
    "Kahvaltı beklentimin altındaydı.",
    "Mekan çok şık dekore edilmiş.",
    "Siparişler çok geç geldi.",
    "Garson çok güleryüzlüydü.",
    "Yemeklerin tuzu yoktu.",
    "Tatlı gerçekten mükemmeldi.",
    "Et çiğ kalmıştı.",
    "Çalışanlar çok nazikti.",
    "Sandalye çok rahatsızdı.",
    "Servis hızlı ve sorunsuzdu.",
    "Yemekten sonra mide bulantısı yaşadım.",
    "Manzara büyüleyiciydi.",
    "Çorba soğuktu.",
    "Her şey mükemmeldi, teşekkürler.",
    "Tatlıdan böcek çıktı.",
    "Rezervasyonumuz sorunsuzdu.",
    "Garsonlar kavga ediyordu.",
    "Çocuk menüsü çok düşünceliydi.",
    "Yemeklerin hepsi yanmıştı.",
    "Fiyat-performans açısından çok iyi.",
    "Lavabolar berbattı.",
    "Çalışanlar çok yardımseverdi.",
    "Müzik sesi çok yüksekti.",
    "Çay sıcak ve tazeydi.",
    "Garson yüzümüze bile bakmadı.",
    "Ailecek güzel bir akşam geçirdik.",
    "Tabaklar kirliydi.",
    "Sunum harikaydı.",
    "Yemek kokuyordu.",
    "Atmosfer çok huzurluydu.",
    "Porsiyon çok küçüktü.",
    "Hizmet kalitesi üst düzeydi.",
    "Menü çok basitti.",
    "İkram için teşekkürler.",
    "Hesap yanlış getirildi.",
    "Açık büfe çok çeşitliydi.",
    "Yemekte saç çıktı.",
    "İç mekan çok aydınlıktı.",
    "Yemeklerin tadı kötüydü.",
    "Güler yüzlü karşılama çok hoşuma gitti.",
    "Tatlılar bayattı.",
    "Mekan çok temizdi.",
    "Sipariş karıştı.",
    "Lezzetli bir akşam yemeği yedik.",
    "Masa örtüsü lekeli ve pisti.",
    "Tatlılar tam kıvamındaydı.",
    "Siparişimiz unutulmuştu.",
    "Yemek servisi mükemmeldi.",
    "Bekleme süresi çok uzundu.",
    "Çalışanlar oldukça profesyoneldi.",
    "Yemek yağ içindeydi.",
    "Sunum çok zarifti.",
    "Yemek servisi dağınıktı.",
    "Garsonlar çok ilgiliydi.",
    "Menü çok karışıktı, anlaşılmıyordu.",
    "Çocuklar için özel oyun alanı vardı.",
    "Salata çok bayattı.",
    "Et lokum gibiydi, çok lezzetliydi.",
    "Peçeteler eksikti.",
    "Her şey çok özenliydi.",
    "İçecekler ılık geldi.",
    "Tatlıdan sonra kahve ikram edildi.",
    "Yemek çok tuzluydu.",
    "Kahvaltı tabağı çok doyurucuydu.",
    "Garson siparişi yanlış aldı.",
    "Ortam çok samimiydi.",
    "İçecekler gazsızdı.",
    "Mekanın dekorasyonu çok zevkliydi.",
    "Sandalyeler gıcırdıyordu.",
    "Fiyatlar çok makuldü.",
    "Havalandırma yetersizdi.",
    "Müşteriye değer verildiğini hissettik.",
    "Yemek beklediğim gibi değildi.",
    "Tatlıyı çok beğendik.",
    "Yemekten sonra mide ağrısı çektim.",
    "İkramlar bizi çok memnun etti.",
    "Servis personeli saygısızdı.",
    "Sunum çok yaratıcıydı.",
    "Garsonlar asıktı.",
    "Müzik çok keyifliydi.",
    "Masamız temizlenmemişti.",
    "Her şey çok tazeydi.",
    "Yemekler kötü kokuyordu.",
    "Köfte tam kıvamında pişmişti.",
    "Servis yavaştı.",
    "Yemekleri çok beğendik.",
    "Tuvaletler çok temizdi.",
    "Yemekler çok baharatlıydı.",
],

"labels": [
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",

    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
]
}





data=pd.DataFrame(data)
le=LabelEncoder()
data["labels"]=le.fit_transform(data["labels"])
tokenizer=Tokenizer()
tokenizer.fit_on_texts(data["texts"])
sequences=tokenizer.texts_to_sequences(data["texts"])
word_index=tokenizer.word_index
#padding
max_len=max(len(seq) for seq in sequences)
x=pad_sequences(sequences,maxlen=max_len)
print(x.shape)
y=data["labels"]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.5,random_state=42)
sentences=[text.split() for text in data["texts"]]
word2_model=Word2Vec(sentences,vector_size=50,window=5,min_count=1)
embeding_dim=50
embeding_matrix=np.zeros((len(word_index)+1,embeding_dim))
for word,i in word_index.items():
    if word in word2_model.wv:
        embeding_matrix[i]=word2_model.wv[word]
        
#model building        
model=Sequential()

model.add(Embedding(input_dim=len(word_index)+1,output_dim=embeding_dim,
                    weights=[embeding_matrix],input_length=max_len,trainable=True))
model.add(SimpleRNN(50,return_sequences=False,))
model.add(Dense(1,activation="sigmoid"))

#model compiling
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#train
model.fit(x_train,y_train,epochs=10,batch_size=2,validation_data=(x_test,y_test))
#evaulate
loss,accuracy=model.evaluate(x_test,y_test)
print(f"loss:{loss},accuracy:{accuracy}")

#predict
def classify_sentences(sentence):
    seq=tokenizer.texts_to_sequences([sentence])
    padded=pad_sequences(seq,maxlen=max_len)
    prediction=model.predict(padded)
    prediction=(prediction>0.5).astype(int)
    if prediction==1:
        prediction="positive"
    else:
        prediction="negative"
    return prediction
sentence="yemekler idare ederdi servis yavaştı sakin bir ortamdı ve gayet temizdi"
prediction=classify_sentences(sentence)
print(f"prediction:{prediction}")







      
































