import tensorflow as tf
import numpy as np
import SplitClassifier


# Class to generate batches
# The datagenerator inherits from the sequence class which is used to generate
# data for each batch of training. Using a sequence generator is much more
# in terms of training time because it allows for variable size batches. (depending
# on the maximum length of sequences in the batch)
class SplitDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_set1, x_set2, y_set, batch_size, classifier, shuffle=True):
        self._x1 = x_set1
        self._x2 = x_set2
        self._y = y_set
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._tokenizer = classifier.tokenizer
        self._max_length = classifier._max_length

    def __len__(self):
        return int(np.ceil(len(self._x1) / self._batch_size))

    def __getitem__(self, idx):
        batch_x1 = self._x1[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_x2 = self._x2[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]

        # Tokenize the input
        tokenized1 = self._tokenizer(batch_x1, padding=True, truncation=True, max_length=self._max_length,
                                     return_tensors='tf')
        tokenized2 = self._tokenizer(batch_x2, padding=True, truncation=True, max_length=self._max_length,
                                     return_tensors='tf')

        return [tokenized1['input_ids'], tokenized1['attention_mask'], tokenized2['input_ids'],
                tokenized2['attention_mask']], batch_y

    def on_epoch_end(self):
        """
        Method is called each time an epoch ends. This will shuffle the data at
        the end of an epoch, which ensures the batches are not identical each
        epoch (therefore improving performance)
        :return:
        """
        if self._shuffle:
            idxs = np.arange(len(self._x1))
            np.random.shuffle(idxs)
            self._x2 = [self._x2[idx] for idx in idxs]
            self._x1 = [self._x1[idx] for idx in idxs]
            self._y = self._y[idxs]


