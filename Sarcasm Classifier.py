import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

train_size=20000
oov_tok = "<OOV>"
trunc_type = "post"
embedding_size = 16
vocab_size = 10000
max_length = 32
padding_type = "post"

train_sentences = sentences[0:train_size]
train_labels = labels[0:train_size]
test_sentences = sentences[train_size:]
test_labels = labels[train_size:]

train_labels_final = np.array(train_labels)
test_labels_final = np.array(test_labels)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_pad = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_pad = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs.get('accuracy')>0.83 and logs.get('val_accuracy')>0.83:
            self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_size, input_length = max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(train_pad,
          train_labels_final,
          epochs=10,
          validation_data = (test_pad, test_labels_final),
          callbacks = [callbacks]
          )















