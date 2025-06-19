# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:04:53 2025

@author: mhmtn
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Daha fazla kelime al
num_words = 20000
maxlen = 100

# Veri hazırlığı
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
word_index = imdb.get_word_index()
reverse_word_index = {index + 3: word for word, index in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

def decode_review(encoded_review):
    return " ".join(reverse_word_index.get(i, "?") for i in encoded_review)

# Positional Encoding Katmanı
class PositionalEncoding(layers.Layer):
    def __init__(self, maxlen, embed_size):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, embed_size)

    def get_angles(self, pos, i, d_model):
        angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return angles

    def positional_encoding(self, maxlen, embed_size):
        angle_rads = self.get_angles(
            np.arange(maxlen)[:, np.newaxis],
            np.arange(embed_size)[np.newaxis, :],
            embed_size
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Transformer Block
class Transformer_block(layers.Layer):
    def __init__(self, embed_size, heads, dropout_rate=0.3):
        super(Transformer_block, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = models.Sequential([
            layers.Dense(embed_size * 4, activation="relu"),
            layers.Dense(embed_size)
        ])
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        attention = self.attention(x, x)
        x = self.norm1(x + self.dropout1(attention, training=training))
        feed_forward = self.feed_forward(x)
        return self.norm2(x + self.dropout2(feed_forward, training=training))

# Transformer Model
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, embed_size, heads, input_dim, output_dim, dropout_rate, maxlen):
        super(TransformerModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=embed_size)
        self.pos_encoding = PositionalEncoding(maxlen, embed_size)
        self.transformer_blocks = [Transformer_block(embed_size, heads, dropout_rate) for _ in range(num_layers)]
        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(output_dim, activation="sigmoid")

    def call(self, x, training=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        x = self.global_avg_pooling(x)
        x = self.dropout(x, training=training)
        return self.fc(x)

# Model parametreleri
num_layers = 2  # derinlik azaltıldı
embed_size = 64
num_heads = 4
input_dim = num_words
output_dim = 1
dropout_rate = 0.5

# Modeli oluştur
model = TransformerModel(num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate, maxlen)
dummy_input = tf.random.uniform((1, maxlen), minval=0, maxval=num_words, dtype=tf.int32)
model(dummy_input)

# Derle
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Eğit
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Sonuçları çiz
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], marker="*", label="train_loss")
plt.plot(history.history["val_loss"], marker="*", label="val_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], marker="*", label="train_acc")
plt.plot(history.history["val_accuracy"], marker="*", label="val_acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# İnteraktif tahmin fonksiyonu
def classify_review(model, review, word_index, maxlen):
    encoded_text = [word_index.get(word, 2) for word in review.lower().split()]  # bilinmeyenler için <UNK> = 2
    padded_text = pad_sequences([encoded_text], maxlen=maxlen)
    pred = model.predict(padded_text)
    if pred[0][0] >= 0.6:
        prediction = "positive"
    else:
        prediction = "negative"
    return prediction, pred[0][0]

review = input("write a review: ")
label, score = classify_review(model, review, word_index, maxlen)
print(f"label: {label}\nscore: {score:.3f}")
