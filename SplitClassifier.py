from transformers import AutoTokenizer, TFAutoModel, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow_addons as tfa
import os

from SplitDataGenerator import *
from keras_examples.CustomCallbacks import *
#from keras_examples.Metrics import *
from abc import ABC, abstractmethod


class SplitClassifier(ABC):
    '''
    Classifier class, which holds a language model and a classifier
    This class can be modified to create whatever architecture you want,
    however it requres the following instance variables:
    self.language_mode_name - this is a passed in variable and it specifies
       which HuggingFace language model to use
    self.tokenizer - this is created, but is just an instance of the tokenizer
       corresponding to the HuggingFace language model you use
    self.model - this is the Keras/Tensor flow classification model. This is
       what you can modify the architecture of, but it must be set

    Upon instantiation, the model is constructed. It should then be trained, and
    the model will then be saved and can be used for prediction.

    Training uses a DataGenerator object, which inherits from a sequence object
    The DataGenerator ensures that data is correctly divided into batches.
    This could be done manually, but the DataGenerator ensures it is done
    correctly, and also allows for variable length batches, which massively
    increases the speed of training.
    '''
    # These are some of the HuggingFace Models which you can use
    # general models
    BASEBERT = 'bert-base-uncased'
    DISTILBERT = 'distilbert-base-uncased'
    ROBERTA = 'roberta-base'
    GPT2 = 'gpt2'
    ALBERT = 'albert-base-v2'

    # social media models
    ROBERTA_TWITTER = 'cardiffnlp/twitter-roberta-base'
    BIOREDDIT_BERT = 'cambridgeltl/BioRedditBERT-uncased'

    # biomedical and clinical models
    # these all are written in pytorch so had to be converted
    # see models directory for the models, and converting_pytorch_to_keras.txt
    # for an explanation of where they came from and how they were converted
    BIO_BERT = './models/biobert_v1.1_pubmed'
    BLUE_BERT_PUBMED = './models/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12'
    BLUE_BERT_PUBMED_MIMIC = './models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12'
    CLINICAL_BERT = './models/bert_pretrain_output_all_notes_150000'
    DISCHARGE_SUMMARY_BERT = './models/bert_pretrain_output_disch_100000'
    BIOCLINICAL_BERT = './models/biobert_pretrain_output_all_notes_150000'
    BIODISCHARGE_SUMMARY_BERT = './models/biobert_pretrain_output_disch_100000'

    # some default parameter values
    EPOCHS = 10
    BATCH_SIZE = 200
    MAX_LENGTH = 512
    # Note: MAX_LENGTH varies depending on the model. For Roberta, max_length = 768.
    #      For BERT its 512
    LEARNING_RATE = 0.01
    DROPOUT_RATE = 0.8
    MODEL_OUT_FILE_NAME = ''

    @abstractmethod
    def __init__(self, language_model_name, language_model_trainable=False, max_length=MAX_LENGTH,
                 learning_rate=LEARNING_RATE, dropout_rate=DROPOUT_RATE):
        '''
        Initializer for a language model. This class should be extended, and
        the model should be built in the constructor. This constructor does
        nothing, since it is an abstract class. In the constructor however
        you must define:
        self.tokenizer
        self.model
        '''
        self.tokenizer = None
        self.model = None
        self._language_model_name = language_model_name
        self._language_model_trainable = language_model_trainable
        self._max_length = max_length
        self._learning_rate = learning_rate
        self._dropout_rate = dropout_rate
        self.tokenizer = AutoTokenizer.from_pretrained(self._language_model_name)

    def load_language_model(self):
        # either load the language model locally or grab it from huggingface
        if os.path.isdir(self._language_model_name):
            language_model = TFBertModel.from_pretrained(self._language_model_name, from_pt=True)
        # else the language model can be grabbed directly from huggingface
        else:
            language_model = TFAutoModel.from_pretrained(self._language_model_name)

        # set properties
        language_model.trainable = self._language_model_trainable
        language_model.output_hidden_states = False

        # return the loaded model
        return language_model

    def train(self, x1, x2, y, batch_size=BATCH_SIZE, validation_data=None, epochs=EPOCHS,
              model_out_file_name=MODEL_OUT_FILE_NAME, early_stopping_monitor='loss', early_stopping_patience=0,
              class_weights=None):
        '''
        Trains the classifier
        :param x: the training data
        :param y: the training labels

        :param batch_size: the batch size
        :param: validation_data: a tuple containing x and y for a validation dataset
                so, validation_data[0] = val_x and validation_data[1] = val_y
                If validation data is passed in, then all metrics (including loss) will
                report performance on the validation data
        :param: epochs: the number of epochs to train for
        '''

        # create a DataGenerator from the training data
        training_data = SplitDataGenerator(x1, x2, y, batch_size, self)

        # generate the validation data (if it exists)
        if validation_data is not None:
            validation_data = SplitDataGenerator(validation_data[0], validation_data[1], validation_data[2], batch_size,
                                            self)

            # set up callbacks
        callbacks = []
        if not model_out_file_name == '':
            callbacks.append(SaveModelWeightsCallback(self, model_out_file_name))
        if early_stopping_patience > 0:
            callbacks.append(EarlyStopping(monitor=early_stopping_monitor, patience=5))

        # print ("*****class weights = " + str(class_weights))
        # exit()

        # fit the model to the training data
        self.model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data,
            class_weight=class_weights,
            verbose=2,
            callbacks=callbacks
        )

    # function to predict using the NN
    def predict(self, x1, x2, batch_size=BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """
        if not isinstance(x1, tf.keras.utils.Sequence):
            tokenized = self.tokenizer(x1, padding=True, truncation=True, max_length=self._max_length,
                                       return_tensors='tf')
            x1 = (tokenized['input_ids'], tokenized['attention_mask'])

        if not isinstance(x2, tf.keras.utils.Sequence):
            tokenized2 = self.tokenizer(x2, padding=True, truncation=True, max_length=self._max_length,
                                        return_tensors='tf')
            x2 = (tokenized2['input_ids'], tokenized2['attention_mask'])

        return self.model.predict((x1, x2), batch_size=batch_size)

    # function to save the model weights
    def save_weights(self, filepath):
        """
        Saves the model weights
        :return: None
        """
        self.model.save_weights(filepath)

    # function to load the model weights
    def load_weights(self, filepath):
        """
        Loads weights for the model
        :param filepath: the filepath (without extension) of the model #TODO - is that the file_path?
        :return:
        """
        self.model.load_weights(filepath)


class Split_CLS_softmax(SplitClassifier):

    def __init__(self, language_model_name, language_model_trainable=False, max_length=SplitClassifier.MAX_LENGTH,
                 learning_rate=SplitClassifier.LEARNING_RATE, dropout_rate=SplitClassifier.DROPOUT_RATE):
        SplitClassifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        # create the language model
        language_model = self.load_language_model()

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids_Q = Input(shape=(None,), dtype=tf.int32, name="input_ids_Q")
        input_padding_mask_Q = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask_Q")

        input_ids_S = Input(shape=(None,), dtype=tf.int32, name="input_ids_S")
        input_padding_mask_S = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask_S")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings_Q = language_model(input_ids=input_ids_Q, attention_mask=input_padding_mask_Q)[0]
        embeddings_S = language_model(input_ids=input_ids_S, attention_mask=input_padding_mask_S)[0]

        sentence_representation_language_model = tf.keras.layers.Concatenate()(
            [embeddings_Q[:, 0, :], embeddings_S[:, 0, :]])

        # softmax
        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(sentence_representation_language_model)  # output3

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids_Q, input_padding_mask_Q, input_ids_S, input_padding_mask_S],
                           outputs=[final_output])

        # my_metrics = MyTextClassificationMetrics(2)
        # metrics = my_metrics.get_all_metrics()

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)]
            # metrics = metrics
        )


class Split_CLS_layers(SplitClassifier):

    def __init__(self, language_model_name, language_model_trainable=False, max_length=SplitClassifier.MAX_LENGTH,
                 learning_rate=SplitClassifier.LEARNING_RATE, dropout_rate=SplitClassifier.DROPOUT_RATE):
        SplitClassifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                                 max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        # create the language model
        language_model = self.load_language_model()

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids_Q = Input(shape=(None,), dtype=tf.int32, name="input_ids_Q")
        input_padding_mask_Q = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask_Q")

        input_ids_S = Input(shape=(None,), dtype=tf.int32, name="input_ids_S")
        input_padding_mask_S = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask_S")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings_Q = language_model(input_ids=input_ids_Q, attention_mask=input_padding_mask_Q)[0]
        embeddings_S = language_model(input_ids=input_ids_S, attention_mask=input_padding_mask_S)[0]

        sentence_representation_language_model = tf.keras.layers.Concatenate()(
            [embeddings_Q[:, 0, :], embeddings_S[:, 0, :]])


        # now, create some dense layers
        # dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation_language_model))

        # dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(self._dropout_rate)
        output2 = dropout2(dense2(output1))

        # dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(self._dropout_rate)
        output3 = dropout3(dense3(output2))

        # softmax
        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(output3)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids_Q, input_padding_mask_Q, input_ids_S, input_padding_mask_S],
                           outputs=[final_output])

        # my_metrics = MyTextClassificationMetrics(2)
        # metrics = my_metrics.get_all_metrics()

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)]
            # metrics = metrics
        )

class Split_BiLSTM_softmax(SplitClassifier):

    def __init__(self, language_model_name, language_model_trainable=False, max_length=SplitClassifier.MAX_LENGTH,
                 learning_rate=SplitClassifier.LEARNING_RATE, dropout_rate=SplitClassifier.DROPOUT_RATE):
        SplitClassifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                                 max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        # create the language model
        language_model = self.load_language_model()

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids_Q = Input(shape=(None,), dtype=tf.int32, name="input_ids_Q")
        input_padding_mask_Q = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask_Q")

        input_ids_S = Input(shape=(None,), dtype=tf.int32, name="input_ids_S")
        input_padding_mask_S = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask_S")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings_Q = language_model(input_ids=input_ids_Q, attention_mask=input_padding_mask_Q)[0]
        embeddings_S = language_model(input_ids=input_ids_S, attention_mask=input_padding_mask_S)[0]

        embeddings = tf.keras.layers.Concatenate()(
            [embeddings_Q, embeddings_S])

        lstm_size = 128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)

        # softmax
        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(sentence_representation_biLSTM)  # output3

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids_Q, input_padding_mask_Q, input_ids_S, input_padding_mask_S],
                           outputs=[final_output])

        # my_metrics = MyTextClassificationMetrics(2)
        # metrics = my_metrics.get_all_metrics()

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)]
            # metrics = metrics
        )

class Split_BiLSTM_layers(SplitClassifier):

    def __init__(self, language_model_name, language_model_trainable=False, max_length=SplitClassifier.MAX_LENGTH,
                 learning_rate=SplitClassifier.LEARNING_RATE, dropout_rate=SplitClassifier.DROPOUT_RATE):
        SplitClassifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        # create the language model
        language_model = self.load_language_model()

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids_Q = Input(shape=(None,), dtype=tf.int32, name="input_ids_Q")
        input_padding_mask_Q = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask_Q")

        input_ids_S = Input(shape=(None,), dtype=tf.int32, name="input_ids_S")
        input_padding_mask_S = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask_S")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings_Q = language_model(input_ids=input_ids_Q, attention_mask=input_padding_mask_Q)[0]
        embeddings_S = language_model(input_ids=input_ids_S, attention_mask=input_padding_mask_S)[0]

        embeddings = tf.keras.layers.Concatenate()(
            [embeddings_Q, embeddings_S])

        # We can create a sentence embedding using the one directly from BERT, or using a biLSTM
        # OR, we can return the sequence from BERT (just don't slice) or the BiLSTM (use retrun_sequences=True)
        # create the sentence embedding layer - using the BERT sentence representation (cls token)

        # Note: we are slicing because this is a sentence classification task. We only need the cls predictions
        # not the individual words, so just the 0th index in the 3D tensor. Other indices are embeddings for
        # subsequent words in the sequence (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

        # Alternatively, we can use a biLSTM to create a sentence representation -- This seems to generally work better
        # create the sentence embedding layer using a biLSTM and BERT token representations
        lstm_size=128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)

        # now, create some dense layers
        # dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation_biLSTM))


        # dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(self._dropout_rate)
        output2 = dropout2(dense2(output1))

        # dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(self._dropout_rate)
        output3 = dropout3(dense3(output2))

        # softmax
        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(output3)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids_Q, input_padding_mask_Q, input_ids_S, input_padding_mask_S],
                           outputs=[final_output])

        # my_metrics = MyTextClassificationMetrics(2)
        # metrics = my_metrics.get_all_metrics()

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)]
            # metrics = metrics
        )