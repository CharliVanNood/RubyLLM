import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, LSTM, Embedding, Dropout, Masking, LayerNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tokenizers import Tokenizer

tf.config.optimizer.set_jit(True)

import keras
import numpy as np
import random
import os
import json
from os import walk

import time

Epochs = 5

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

def generate_seq(model, tokenizer, max_length, seed_text, n_words, out_text, breakPoints):
    in_text = seed_text
    return_text = out_text
    writing = True
    replication = False
    wordsWritten = 0
    outsequence = []

    encodedInput = tokenizer.encode(in_text).ids

    word_ = 0
    while word_ < n_words and writing:
        word_ += 1
        wordConverted = False
        word = ""

        yhat = 0

        try:
            encoded = pad_sequences([encodedInput], maxlen=max_length, padding='pre', value=3)
            print(encoded)
            prediction = model.predict(encoded)

            percentage_ = 100
            percentages_ = []

            while percentage_ > 10:
                result_ = np.float32(np.max(prediction))
                percentage_ = round(result_ * 100)
                yhat = prediction[0].tolist().index(result_)
                for x in range(percentage_):
                    percentages_.append(yhat)
                prediction[0][yhat] = 0

            if len(percentages_) == 0:
                print("NO VALID WORDS FOUND")
                percentages_.append(random.randint(0, VocabSize))

            yhat = percentages_[random.randint(0, len(percentages_) - 1)]

            out_word = ''

            wordConverted = True
            word = tokenizer.decode([yhat])
            print("word created: " + str(word))
        except Exception as e:
            print(e)
            print("prediction doesn't exist")
            out_word = "eh"
            in_text += out_word
            return_text += out_word
            if random.randint(0, 5) == 0:
                writing = False

        if wordConverted:
            try:
                outsequence.append(yhat)
                encodedInput.append(yhat)
                out_word = word
                in_text += out_word
                return_text += out_word
                wordsWritten += 1
            except Exception as e:
                print("word doesn't exist")

        if not writing:
            word_ = n_words

    return_text = tokenizer.decode(outsequence)

    return return_text


def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)
    
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        #mask = tf.linalg.band_part(tf.ones((tf.shape(score)[1], tf.shape(score)[2])), -1, 0)
        #scaled_score += (mask * -1e9)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, embed_dim)
    
    def get_angles(self, pos, i, d_model):
        angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = PositionalEncoding(maxlen, embed_dim)
    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.token_emb(x)
        return self.pos_emb(x)


def build_transformer_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers = 6):
    inputs = keras.layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, 0.1)(x)
    x = Dropout(0.1)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(vocab_size, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


class quickSave(keras.callbacks.Callback):
    def __init__(self, model, tokenizer, sequenceLength):
        self.model = model
        self.tokenizer = tokenizer
        self.sequenceLength = sequenceLength

    def on_epoch_end(self, epoch, logs={}):
        print(generate_seq(self.model, self.tokenizer, self.sequenceLength, "hello how are you doing today?[SEP]", 64, "hello how are you doing today?[SEP]", False))
        print('saving model')
        amountOfFiles = len(next(walk("./trainingOutput"), (None, None, []))[2]) - 3
        self.model.save(f"./trainingOutput/epoch{str(amountOfFiles + 1)}.h5")
        self.tokenizer.save(f"./trainingOutput/epoch{str(amountOfFiles + 1)}.json")
        print('saved model')

        self.history_file = "training_history.json"

        metrics = {key: float(value) for key, value in logs.items()}
        metrics["epoch"] = epoch + 1
        
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                history_data = json.load(f)
            history_data.append(metrics)
        else:
            history_data = [metrics]
        
        with open(self.history_file, "w") as f:
            json.dump(history_data, f, indent=4)
        print('Metrics saved.')

def TrainModelNew():
    print("*** Loading Tokenizer Data ***")
    tokenizer = Tokenizer.from_file("checkpoints/tokenizer.json")
    output = tokenizer.encode("Hello, this is a test.[SEP]")
    print("IDs:", output.ids)
    print("Tokens:", output.tokens)
    print("*** Loaded Tokenizer Data ***\n")

    print("*** Loading Tokenized Data ***")
    with open('tokenized/data.json', 'r') as file:
        data = json.load(file)
    train_x = np.array(data[0])
    train_y = np.array(data[1])
    vocab_size = data[2]
    print("Sequences loaded: " + str(len(train_y)))
    print("Sequence  length: " + str(len(train_x[0])))
    print("vocab     length: " + str(vocab_size))
    print("*** Loaded Tokenized Data ***\n")
    sequenceLength = len(train_x[0])

    print("*** compiling model ***")
    model = build_transformer_model(vocab_size, sequenceLength, 128, 2, 128, 3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("*** compiled model ***\n")

    epochs_ = input(f"epochs (default: {Epochs}): ")
    if epochs_ == "" or not epochs_.isnumeric():
        epochs_ = Epochs
    else:
        epochs_ = int(epochs_)

    print("*** Model Info ***")
    print(model.summary())

    if os.path.exists("training_history.json"):
        os.remove("training_history.json")
        print("training_history.json has been deleted.")
    else:
        print("training_history.json does not exist.")

    input("start training: ")
    print("*** Training ***")

    model.fit(train_x, train_y, batch_size=512, epochs=epochs_, validation_split=0.05, callbacks=[quickSave(model, tokenizer, sequenceLength), LearningRateScheduler(lr_schedule)])

    amountOfFiles = len(next(walk("./trainingOutput"), (None, None, []))[2]) - 3
    model.save(f"./trainingOutput/epoch{str(amountOfFiles + 1)}.h5")

    print("*** Training done ***")
    print("")
