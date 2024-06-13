import sys
import os

import tensorflow as tf
import numpy as np
import random
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle

class STTModels:
    def __init__(self, loader):
        """Initialize STT modeling."""
        self.loader = loader
        file_path = self.loader.get_file_path()
        self.train_audio_paths, self.train_audio_text, self.train_audio_duration = self.loader.load_transcription(
            file_path + 'train/', type='train')
        self.valid_audio_paths, self.valid_audio_text, self.valid_audio_duration = self.loader.load_transcription(
            file_path + 'test/', type='validation')
        self.cur_train_index = 0
        self.cur_valid_index = 0
        self.window = 20
        self.rng = random.Random(123)
        self.max_freq = 8000
        self.feat_dim = int(0.001 * self.window * self.max_freq) + 1
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.mfcc_dim = 13

    def model(self, input_dim, filters, kernel_size, conv_stride,
              conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=2,
              cell=tf.keras.layers.GRU, activation='tanh'):
        """Build a deep network for speech."""
        input_data = tf.keras.Input(name='the_input', shape=(None, input_dim))
        conv_1d = tf.keras.layers.Conv1D(filters, kernel_size,
                                         strides=conv_stride,
                                         padding=conv_border_mode,
                                         activation='relu',
                                         name='layer_1_conv',
                                         dilation_rate=1)(input_data)
        conv_bn = tf.keras.layers.BatchNormalization(name='conv_batch_norm')(conv_1d)

        if number_of_layers == 1:
            layer = cell(units, activation=activation,
                         return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
            layer = tf.keras.layers.BatchNormalization(name='bt_rnn_1')(layer)
        else:
            layer = cell(units, activation=activation,
                         return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
            layer = tf.keras.layers.BatchNormalization(name='bt_rnn_1')(layer)

            for i in range(number_of_layers - 2):
                layer = cell(units, activation=activation,
                             return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate)(layer)
                layer = tf.keras.layers.BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

            layer = cell(units, activation=activation,
                         return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
            layer = tf.keras.layers.BatchNormalization(name='bt_rnn_final')(layer)

        time_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(output_dim))(layer)
        y_pred = tf.keras.layers.Activation('softmax', name='softmax')(time_dense)
        model = Model(inputs=input_data, outputs=y_pred)
        model.output_length = lambda x: self.cnn_output_length(
            x, kernel_size, conv_border_mode, conv_stride)
        print(model.summary())
        return model

    def ctc_lambda_func(self, args):
        """CTC lambda function."""
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def add_ctc_loss(self, input_to_softmax):
        """Add a CTC loss to the model."""
        the_labels = tf.keras.Input(
            name='the_labels', shape=(None,), dtype='float32')
        input_lengths = tf.keras.Input(
            name='input_length', shape=(1,), dtype='int64')
        label_lengths = tf.keras.Input(
            name='label_length', shape=(1,), dtype='int64')
        output_lengths = tf.keras.layers.Lambda(
            input_to_softmax.output_length)(input_lengths)
        loss_out = tf.keras.layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [input_to_softmax.output, the_labels, output_lengths, label_lengths])
        model = Model(
            inputs=[input_to_softmax.input, the_labels,
                    input_lengths, label_lengths],
            outputs=loss_out)
        return model

    def next_train(self, batch_size):
        """Obtain a batch of training data."""
        while True:
            ret = self.get_batch('train')
            self.cur_train_index += batch_size
            if self.cur_train_index >= len(self.train_audio_text) - batch_size:
                self.cur_train_index = 0
                self.shuffle_data_by_partition('train')
            yield ret

    def next_valid(self, batch_size):
        """Obtain a batch of validation data."""
        while True:
            ret = self.get_batch('valid')
            self.cur_valid_index += batch_size
            if self.cur_valid_index >= len(self.valid_audio_text) - batch_size:
                self.cur_valid_index = 0
                self.shuffle_data_by_partition('valid')
            yield ret

    def text_to_int_sequence(self, text):
        """Convert text to an integer sequence."""
        char_map, _ = self.loader.map_index()
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = char_map['<SPACE>']
            else:
                ch = char_map.get(c, char_map['<UNK>'])
            int_sequence.append(ch)
        return int_sequence

    def featurize(self, audio_clip):
        """Calculate the corresponding feature for a given audio clip."""
        (rate, sig) = wav.read(audio_clip)
        return mfcc(sig, rate, numcep=13)

    def normalize(self, feature, eps=1e-14):
        """Center a feature using the mean and std."""
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def get_batch(self, partition):
        """Obtain a batch of train, validation, or test data."""
        if partition == 'train':
            audio_paths = self.train_audio_paths
            texts = self.train_audio_text
            cur_index = self.cur_train_index
        elif partition == 'valid':
            audio_paths = self.valid_audio_paths
            texts = self.valid_audio_text
            cur_index = self.cur_valid_index
        else:
            raise Exception("Invalid partition. Must be train/validation")

        features = [self.normalize(self.featurize(a)) for a in
                    audio_paths[cur_index:cur_index+self.minibatch_size]]

        max_length = max([features[i].shape[0]
                         for i in range(0, self.minibatch_size)])
        max_string_length = max([len(texts[cur_index+i])
                                for i in range(0, self.minibatch_size)])

        X_data = np.zeros([self.minibatch_size, max_length,
                           self.feat_dim])
        labels = np.ones([self.minibatch_size, max_string_length]) * 28
        input_length = np.zeros([self.minibatch_size, 1])
        label_length = np.zeros([self.minibatch_size, 1])

        for i in range(0, self.minibatch_size):
            feat = features[i]
            input_length[i] = feat.shape[0]
            X_data[i, :feat.shape[0], :] = feat

            label = np.array(self.text_to_int_sequence(texts[cur_index+i]))
            labels[i, :len(label)] = label
            label_length[i] = len(label)

        outputs = {'ctc': np.zeros([self.minibatch_size])}
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length
                  }
        return (inputs, outputs)

    def fit_train(self, k_samples=100):
        """Estimate the mean and std of the features from the training set."""
        k_samples = min(k_samples, len(self.train_audio_paths))
        samples = self.rng.sample(self.train_audio_paths, k_samples)
        feats = [self.featurize(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)

    def train(self, audio_gen, input_to_softmax, model_name, minibatch_size=20, optimizer=tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5), epochs=20, verbose=1):
        """Train the model."""
        self.minibatch_size = minibatch_size
        self.spectrogram = False
        self.fit_train()
        num_train_examples = len(self.train_audio_paths)
        steps_per_epoch = num_train_examples // minibatch_size
        num_valid_samples = len(self.valid_audio_paths)
        validation_steps = num_valid_samples // minibatch_size

        model = self.add_ctc_loss(input_to_softmax)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

