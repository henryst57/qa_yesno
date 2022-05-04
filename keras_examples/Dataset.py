import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection
from ast import literal_eval
from abc import ABC, abstractmethod

#import preprocessor as p
import sklearn.utils

#TODO - Should I have a default seed value here?
SEED = 3

#Abstract dataset class
class Dataset(ABC):
    
    @abstractmethod
    def __init__(self, seed=SEED, validation_set_size=0):
        """
        Constructor for a Dataset
        validation_set_size is the percentage to use for validation set (e.g. 0.2 = 20%
        """
        self.seed = seed
        self._val_set_size = validation_set_size
        self._train_X = None
        self._train_Y = None
        self._val_X = None
        self._val_Y = None


    def _training_validation_split(self, data, labels):
        """
        Performs a stratified training-validation split
        """   
        # Error Checking
        if (self._val_set_size >= 1):
            raise Exception("Error: test set size must be greater than 0 and less than 1")
        
        if (self._val_set_size > 0):
            #Split the data - this automatically does a stratified split
            # meaning that the class ratios are maintained
            self._train_X, self._val_X, self._train_Y, self._val_Y = sklearn.model_selection.train_test_split(data, labels, test_size=self._val_set_size, random_state=self.seed)
        else:
            # set the data to unsplit
            self._train_X = data
            self._train_Y = labels
            self._val_X = None
            self._val_Y = None

            
    def get_train_data(self):
        if self._train_X is None or self._train_Y is None:
            raise Exception("Error: train data does not exist, you must call _training_validation_split after loading data")
        return self._train_X, self._train_Y

    
    def get_validation_data(self):
        if self._val_X is None or self._val_Y is None:
            raise Exception("Error: val data does not exist, you must specify a validation split percent")
        return self._val_X, self._val_Y

    
    #You can tweek this however you want
    def preprocess_data(self, data):
        # preprocess tweets to remove mentions, URL's
        p.set_options(p.OPT.MENTION, p.OPT.URL)  # P.OPT.HASHTAG, p.OPT.EMOJI
        data = data.apply(p.clean)

        # Tokenize special Tweet characters
        # p.set_options(p.OPT.NUMBER)  # p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.RESERVED,
        # data = data.apply(p.tokenize)

        return data.tolist()


    def get_train_class_weights(self):
        return self.class_weights


    def _determine_class_weights(self):
        raise NotImplemented("ERROR: Class weights is not implemented for this dataset type")
            

#Load a data and labels for a text classification dataset
class Binary_Text_Classification_Dataset(Dataset):
    '''
    Class to load and store a text classification dataset. Text classification datasets
    contain text and a label for the text, and possibly other information. Columns
    are assumed to be tab seperated and each row corresponds to a different sample

    Inherits from the Dataset class, only difference is in how the data is loaded
    upon initialization
    '''
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, label_column_names=None, seed=SEED, validation_set_size=0):
        '''
        Method to instantiate a text classification dataset
        :param data_file_path: the path to the file containing the data
        :param text_column_name: the column name containing the text
        :param label_column_name: the column name containing the label (class) of the text (for binary labeled data)
        :param label_column_names: a list of column names containing the label (class) of the text data (for multi-label data)
           text_column_name must be specified for multi-label data
        :param seed: the seed for random split between test and training sets
        :param make_test_train_split: a number between 0 and 1 determining the percentage size of the test set
           if no number is passed in (or if a 0 is passed in), then no test train split is made
        '''
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)

        #load the labels
        if(label_column_names is None): #check if binary or multi-label #TODO --- this isn't necessarily true. I think I should just load the data differently for multilabel or multi-class problems (create a different method)
            #no label column names passed in, so it must be binary
            #load the labels with or without column header info
            if (text_column_name is None or label_column_name is None):
                text_column_name = 'text'
                label_column_name = 'label'
                df = pd.read_csv(data_file_path, header=None, names=[text_column_name, label_column_name], delimiter='\t').dropna()
            else:
                df = pd.read_csv(data_file_path, delimiter='\t').dropna()

            labels = df[label_column_name].values.reshape(-1, 1)
            
        #load multilabel classification data. Column names are required
        else:
            #TODO - implement this if/when it is needed. Actual implementation depends on the data format
            if (text_column_name is None):
                raise Exception("Error: text_column_name must be specified to load multilabel data")

            #NOTE: in the case of multiclass data, where class is an integer, it is easy to encode
            # as one-hot data with the following command:
            #label_encoder = OneHotEncoder(sparse=False)
            #labels = label_encoder.fit_transform(df[label_column_name].values.reshape(-1, 1))
            #----however, with a single column this else statement won't get called since its just a single column
            #    ...so, would need to modify this method
            
            #sklearn.multilabel binarizer is another option.
            # in the end though, what you should get is a list of lists e.g. [0,1,1],[1,0,0],[0,1,0] where each triplet
            # are the labels for a single sample. If confised, check the label encoder to check
            raise Exception("Error: not yet implemented, depends on dataset")

            
        #load the data
        raw_data = df[text_column_name]
        data = df[text_column_name].values.tolist()

        #data = self.preprocess_data(raw_data)

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights()


    def _determine_class_weights(self):
        """
        Creates a dictionary of class weights such as 
        class_weight = {0: 1., 1: 50., 2:2.}
        """

        #calculate the weight of each class
        num_positive = np.sum(self._train_Y, axis=0)
        total_samples = len(self._train_Y)

        #create the class weights (0 = neg, 1 = pos)
        self.class_weights = {}
        self.class_weights[1] = num_positive/total_samples
        self.class_weights[0] = 1.-self.class_weights[1]
