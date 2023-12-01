import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']
print("there are {} train data and {} test data\n".format(len(train_data), len(test_data)))

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for sentence, label in train_data:
    train_sentences.append(str(sentence.numpy().decode('utf8')))
    train_labels.append(label.numpy())
# print(train_sentences[1])

for sentence, label in test_data:
    test_sentences.append(str(sentence.numpy().decode('utf8')))
    test_labels.append(label.numpy())

train_labels_final = np.array(train_labels)
test_labels_final = np.array(test_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences= tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating = trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer = "adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.83 and logs.get('val_accuracy') > 0.83:
            self.model.stop_training = True

callbacks = myCallback()

model.fit(train_padded,
          train_labels_final,
          epochs= 10,
          validation_data = (test_padded, test_labels_final),
          callbacks = [callbacks]
)

