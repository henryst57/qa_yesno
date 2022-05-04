from SplitClassifier import *
from SplitDataset import *

def Split_BioASQ_experiment(classifier_func, arch, r, folder='data'):
    # load the dataset
    data_filepath = os.path.join("..", folder, 'bioasq_split.tsv')
    data = Binary_Text_Classification_Dataset(data_filepath, validation_set_size=0.2)


    # create classifier and load data for a binary text classifier
    # language_model_name = Classifier.ROBERTA_TWITTER
    language_model_name = SplitClassifier.BASEBERT
    num_classes = 8
    max_length = 512
    max_epoch = 100
    batch_size = 20
    learning_rate = [1*(10**-r)]#[0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    rate_index = 0
    for rate in learning_rate:
        rate_index += 1
        print("Training on a leaning rate of {}\n".format(rate))
        classifier = classifier_func(language_model_name, max_length=max_length, language_model_trainable=True,
                                            learning_rate=rate)

        # get the training data
        train_x1, train_y, train_x2 = data.get_train_data()
        val_x1, val_y, val_x2 = data.get_validation_data()

        # train the model
        # If you want to save model weights, use below.
        classifier.train(train_x1, train_x2, train_y,
                         validation_data=[val_x1, val_x2, val_y],
                         class_weights=data.get_train_class_weights(),
                         early_stopping_patience=5,
                         early_stopping_monitor='loss',
                         epochs=max_epoch,
                         batch_size=batch_size
                         )

        # weights_filepath = input("Please enter a name for the file you want to save")
        weights_filepath = "../weights/weights_{}{}".format(arch, r)
        classifier.save_weights(weights_filepath)


        y_pred = classifier.predict(train_x1, train_x2)
        predictions = np.round(y_pred)
        print("predictions = " + str(predictions) + "\n\n")
# f1 = sklearn.metrics.f1_score(test_y, predictions)
# print("f1 = " + str(f1))

def Split_BoolQ_experiment(classifier_func, arch, exp, r, folder='data'):
    # load the dataset
    data_filepath = os.path.join("..", folder, 'balanced_boolq_split.tsv')
    # test_filepath = '8B1_golden.json'
    data = Binary_Text_Classification_Dataset(data_filepath, validation_set_size=0.2)
    # test = Binary_Text_Classification_Dataset(test_filepath)
    data_filepath = os.path.join("..", folder, 'bioasq_split.tsv')
    bio_data = Binary_Text_Classification_Dataset(data_filepath, validation_set_size=0.2)

    # create classifier and load data for a binary text classifier
    # language_model_name = Classifier.ROBERTA_TWITTER
    language_model_name = SplitClassifier.BASEBERT
    num_classes = 8
    max_length = 512
    max_epoch = 100
    batch_size = 20
    learning_rate = [1*(10**-r)]#[0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    rate_index = 0
    for rate in learning_rate:
        rate_index += 1
        print("Training on a leaning rate of {}\n".format(rate))
        classifier = classifier_func(language_model_name, max_length=max_length, language_model_trainable=True,
                                            learning_rate=rate)

        # get the training data
        train_x1, train_y, train_x2 = data.get_train_data()
        val_x1, val_y, val_x2 = data.get_validation_data()

        train2_x1, train2_y, train2_x2 = bio_data.get_train_data()
        val2_x1, val2_y, val2_x2 = bio_data.get_validation_data()
        # test_x, test_y = test.get_train_data()

        # train the model
        # If you want to save model weights, use below.
        classifier.train(train_x1, train_x2, train_y,
                         validation_data=[val_x1, val_x2, val_y],
                         class_weights=data.get_train_class_weights(),
                         early_stopping_patience=5,
                         early_stopping_monitor='loss',
                         epochs=max_epoch,
                         batch_size=batch_size
                         )

        classifier.train(train2_x1, train2_x2, train2_y,
                         validation_data=[val2_x1, val2_x2, val2_y],
                         class_weights=bio_data.get_train_class_weights(),
                         early_stopping_patience=5,
                         early_stopping_monitor='loss',
                         epochs=max_epoch,
                         batch_size=batch_size
                         )



        # weights_filepath = input("Please enter a name for the file you want to save")
        weights_filepath = "../weights/weights_{}_{}{}".format(exp, arch, r)
        classifier.save_weights(weights_filepath)


        y_pred = classifier.predict(train_x1, train_x2)
        predictions = np.round(y_pred)
        print("predictions = " + str(predictions) + "\n\n")

# f1 = sklearn.metrics.f1_score(test_y, predictions)
# print("f1 = " + str(f1))

def run_models():
    models = [Split_BiLSTM_layers, Split_BiLSTM_softmax, Split_CLS_layers, Split_CLS_softmax][::-1]
    titles = ['Split_BiLSTM_layers', 'Split_BiLSTM_softmax', 'Split_CLS_layers', 'Split_CLS_softmax'][::-1]

    # print("Testing control experiment...\n")
    # for i in range(len(models)):
    #     if i == 2:
    #         break
    #     print("Testing {} model...".format(titles[i]))
    #     Split_BioASQ_experiment(models[i], titles[i])
    #
    # print("Testing BoolQ experiment...\n")
    # for i in range(len(models)):
    #     if i == 2:
    #         break
    #     print("Testing {} model...".format(titles[i]))
    #     Split_BoolQ_experiment(models[i], titles[i])

    print("\n\n\nTest 1/8 BioASQ experiment for Split_CLS_softmax model at a learning rate of 1e-6")
    Split_BioASQ_experiment(models[0], titles[0], 6)
    print("\n\n\nTest 2/8 BioASQ experiment for Split_CLS_softmax model at a learning rate of 1e-8")
    Split_BioASQ_experiment(models[0], titles[0], 8)
    print("\n\n\nTest 3/8 BioASQ experiment for Split_CLS_layers model at a learning rate of 1e-6")
    Split_BioASQ_experiment(models[1], titles[1], 6)
    print("\n\n\nTest 4/8 BioASQ experiment for Split_CLS_layers model at a learning rate of 1e-7")
    Split_BioASQ_experiment(models[1], titles[1], 7)
    print("\n\n\nTest 5/8 BioASQ experiment for Split_CLS_layers model at a learning rate of 1e-8")
    Split_BioASQ_experiment(models[1], titles[1], 8)
    print("\n\n\nTest 6/8 BoolQ experiment for Split_CLS_softmax model at a learning rate of 1e-6")
    Split_BoolQ_experiment(models[0], titles[0], "BoolQ", 6)
    print("\n\n\nTest 7/8 BoolQ experiment for Split_CLS_layers model at a learning rate of 1e-5")
    Split_BoolQ_experiment(models[1], titles[1], "BoolQ", 5)
    print("\n\n\nTest 8/8 BoolQ experiment for Split_CLS_layers model at a learning rate of 1e-7")
    Split_BoolQ_experiment(models[1], titles[1], "BoolQ", 7)
    print("\n\n\nDONE!")

if __name__ == '__main__':
    run_models()