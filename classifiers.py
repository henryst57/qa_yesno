from keras_examples.Classifier import *

class CLS_to_softmax(Classifier):
    def __init__(self, language_model_name, language_model_trainable=False, max_length=Classifier.MAX_LENGTH,
                 learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        language_model = self.load_language_model()

        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]
        sentence_representation_language_model = embeddings[:, 0, :]  # CLS token

        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(sentence_representation_language_model)

        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)])

class CLS_to_layers(Classifier):

    def __init__(self, language_model_name, language_model_trainable=False, max_length=Classifier.MAX_LENGTH,
                 learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        language_model = self.load_language_model()

        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]
        sentence_representation_language_model = embeddings[:, 0, :]  # CLS token

        # now, create some dense layers
        # dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation_language_model))

        # dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(self._dropout_rate)
        output2 = dropout2(dense2(output1))  # // vector output

        # dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(self._dropout_rate)
        output3 = dropout3(dense3(output2))

        # softmax
        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(output3)

        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)])

class BiLSTM_to_softmax(Classifier):
    def __init__(self, language_model_name, language_model_trainable=False, max_length=Classifier.MAX_LENGTH,
                 learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        language_model = self.load_language_model()

        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        lstm_size=128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)

        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(sentence_representation_biLSTM)

        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                     tfa.metrics.F1Score(1)])

class BiLSTM_to_layers(Classifier):

    def __init__(self, language_model_name, language_model_trainable=False, max_length=Classifier.MAX_LENGTH,
                 learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        language_model = self.load_language_model()

        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        lstm_size = 128
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
        output2 = dropout2(dense2(output1))  # // vector output

        # dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(self._dropout_rate)
        output3 = dropout3(dense3(output2))

        # softmax
        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(output3)

        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)])
