import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_portion = .8

sentences = bbc['text']
labels = bbc['category']

train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, shuffle=False, train_size=training_portion)

#because the label is in word, therefore we need
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

train_labels = label_tokenizer.texts_to_sequences(train_labels)
test_labels = label_tokenizer.texts_to_sequences(test_labels)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# print(np.unique(train_labels))
# print(np.unique(test_labels))

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_pad = pad_sequences(train_sequences, maxlen= max_length,truncating=trunc_type, padding=padding_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_pad = pad_sequences(test_sequences, maxlen=max_length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer = "adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs.get('accuracy') > 0.85 and logs.get('val_accuracy')>0.85:
            self.model.stop_training=True

callbacks = myCallback()

model.fit(train_pad,
          train_labels,
          epochs = 30,
          validation_data = (test_pad, test_labels),
          callbacks = [callbacks])


