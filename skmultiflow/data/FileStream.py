__author__ = 'Guilherme Matsumoto'

from skmultiflow.data import BaseInstanceStream
from skmultiflow.core.instances.Instance import Instance
import pandas as pd
import numpy as np

class FileStream(BaseInstanceStream.BaseInstanceStream):
    '''
        CSV File Stream
        -------------------------------------------
        Generates a stream based on the data from a file
        
        Parser parameters
        ---------------------------------------------
        -f: CSV file to load
    '''
    def __init__(self, file_opt, num_classes = 2):
        super().__init__()
        # default values
        if file_opt.file_type is "CSV":
            self.read_function = pd.read_csv
        else:
            raise ValueError('Unsupported format: ', file_opt.file_type)
        self.file_name = None
        self.instance_index = 0
        self.instance_length = 0
        self.X = None
        self.y = None
        self.current_instance_x = None
        self.current_instance_y = None
        self.num_classes = num_classes
        self.num_numerical_attributes = 0
        self.num_nominal_attributes = 0
        self.num_values_per_nominal_att = 0
        self.attributes_header = None
        self.classes_header = None
        self.instances = None
        self.num_attributes = 0

        self.configure(file_opt, num_classes, 0)

    def configure(self, file_opt, num_classes, index = -1):
        '''
        __init__(self, file_name, index)
        
        Parameters
        ----------------------------------------
        file_name : string
                   Name of the file
        index : int
                Class index parameter
        '''
        self.file_name = file_opt.get_file_name()
        self.instance_index = index
        self.instances = None
        self.instance_length = 0
        self.current_instance_x = None
        self.current_instance_y = None
        self.num_classes = num_classes
        self.X = None
        self.y = None




    def prepare_for_use(self):
        self.restart()


    def restart(self):
        '''
        restart(self)
        ----------------------------------------
        Read the file and set object attributes
        '''

        try:
            instance_aux = self.read_function(self.file_name)
            self.instance_length = len(instance_aux.index)
            self.num_attributes = len(instance_aux.columns) - 1
            labels = instance_aux.columns.values.tolist()
            self.X = instance_aux.iloc[:, 0:(len(labels)-1)]
            self.y = instance_aux.iloc[:, (len(labels)-1):]
            self.attributes_header = labels[0:(len(labels) - 1)]
            self.classes_header = labels[(len(labels) - 1):]
            self.instance_index = 0
        except IOError:
            print("CSV file reading failed. Please verify the file format.")
        pass

    def is_restartable(self):
        return True

    def next_instance(self, batch_size = 1):

        self.instance_index += 1
        #self.current_instance_x = self.X[self.instance_index-1:self.instance_index+batchSize-1].values[0]
        #self.current_instance_y = self.y[self.instance_index-1:self.instance_index+batchSize-1].values[0]
        self.current_instance_x = self.X.iloc[self.instance_index - 1:self.instance_index + batch_size - 1, :].values
        self.current_instance_y = self.y.iloc[self.instance_index - 1:self.instance_index + batch_size - 1, :].values.flatten()
        return (self.current_instance_x, self.current_instance_y)

    def has_more_instances(self):
        return ((self.instance_length - self.instance_index) > 0)

    def estimated_remaining_instances(self):
        return (self.instance_length - self.instance_index)

    def print_df(self):
        print(self.X)
        print(self.y)

    def get_instances_length(self):
        return self.instance_length

    def get_num_attributes(self):
        return self.num_attributes

    def has_more_mini_batch(self):
        pass

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_classes(self):
        return self.num_classes

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.classes_header

    def get_last_instance(self):
        return (self.current_instance_x, self.current_instance_y)

    def get_plot_name(self):
        return "File Stream - " + str(self.num_classes) + " class labels"

    def get_classes(self):
        c = np.unique(self.y)
        return c