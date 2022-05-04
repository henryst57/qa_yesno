from classifiers import *
from keras_examples.Dataset import *
from keras_examples.Classifier import *
from Split_experiment import run_models


def control_experiment(classifier_func,arch, folder="data"):
    # load the dataset
    data_filepath = os.path.join("..", folder, "bioasq_data.tsv")
    data = Binary_Text_Classification_Dataset(data_filepath, validation_set_size=0.2)

    # create classifier and load data for a binary text classifier
    language_model_name = Classifier.BASEBERT
    num_classes = 8
    max_length = 512
    max_epoch = 100
    batch_size = 20
    learning_rate = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    counter = 0
    for rate in learning_rate:
        counter += 1
        print("\n\nTraining on a learning rate of {}\n".format(rate))

        classifier = classifier_func(language_model_name, max_length=max_length, language_model_trainable=True,
                                     learning_rate=rate)

        # get the training data
        train_x, train_y = data.get_train_data()
        val_x, val_y = data.get_validation_data()

        # train the model
        # If you want to save model weights, use below.
        classifier.train(train_x, train_y,
                         validation_data=(val_x, val_y),
                         class_weights=data.get_train_class_weights(),
                         early_stopping_patience=5,
                         early_stopping_monitor='loss',
                         epochs=max_epoch,
                         batch_size=batch_size
                         )

        # weights_filepath = os.path.join("..", 'weights', "weights_control_{}_{}".format(arch, counter))
        # classifier.save_weights(weights_filepath)

        y_pred = classifier.predict(train_x)
        predictions = np.round(y_pred)
        print("predictions = " + str(predictions) + "\n\n")


def BoolQ_experiment(classifier_func,arch, folder="data"):
    # load the dataset
    data_filepath = os.path.join("..", folder, "balanced_boolq.tsv")
    data = Binary_Text_Classification_Dataset(data_filepath, validation_set_size=0.2)
    data_filepath = os.path.join("..", folder, "bioasq_data.tsv")
    bio_data = Binary_Text_Classification_Dataset(data_filepath, validation_set_size=0.2)

    # create classifier and load data for a binary text classifier
    language_model_name = Classifier.BASEBERT
    num_classes = 8
    max_length = 512
    max_epoch = 100
    batch_size = 20
    learning_rate = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    counter = 0
    for rate in learning_rate:
        counter += 1
        print("\n\nTraining on a learning rate of {}\n".format(rate))

        classifier = classifier_func(language_model_name, max_length=max_length, language_model_trainable=True,
                                     learning_rate=rate)

        # get the training data
        train_x, train_y = data.get_train_data()
        val_x, val_y = data.get_validation_data()

        train_x2, train_y2 = bio_data.get_train_data()
        val_x2, val_y2 = bio_data.get_validation_data()

        # train the model
        # If you want to save model weights, use below.
        classifier.train(train_x, train_y,
                         validation_data=(val_x, val_y),
                         class_weights=data.get_train_class_weights(),
                         early_stopping_patience=5,
                         early_stopping_monitor='loss',
                         epochs=max_epoch,
                         batch_size=batch_size
                         )
        classifier.train(train_x2, train_y2,
                         validation_data=(val_x2, val_y2),
                         class_weights=bio_data.get_train_class_weights(),
                         early_stopping_patience=5,
                         early_stopping_monitor='loss',
                         epochs=max_epoch,
                         batch_size=batch_size
                         )

        # weights_filepath = os.path.join("..", 'weights', "weights_tune_{}_{}".format(arch, rate))
        # classifier.save_weights(weights_filepath)

        y_pred = classifier.predict(train_x2)
        predictions = np.round(y_pred)
        print("predictions = " + str(predictions) + "\n\n")


def run_split_experiment():
    run_models()


if __name__ == '__main__':

    run_split_experiment()

    models = [BiLSTM_to_layers, BiLSTM_to_softmax, CLS_to_layers, CLS_to_softmax]#[::-1]
    titles = ['BiLSTM_to_layers', 'BiLSTM_to_softmax', 'CLS_to_layers', 'CLS_to_softmax']#[::-1]

    # folder = input("Please enter the folder the data is located: ")

    print("Testing control experiment...\n")
    for i in range(len(models)):
        print("Testing {} model...".format(titles[i]))
        control_experiment(models[i], titles[i])

    print("Testing BoolQ experiment...\n")
    for i in range(len(models)):
        print("Testing {} model...".format(titles[i]))
        BoolQ_experiment(models[i], titles[i])



