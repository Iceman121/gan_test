# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:28:48 2018

@author: Shashwat Pathak
"""

# =============================================================================
# Chapter 0: Import Modules
# =============================================================================
# import matplotlib.pyplot as plt
import logging
from keras.datasets import mnist

# =============================================================================
# Chapter 1: Setting Logging Parameters
# =============================================================================
#logging.basicConfig(filename='reports/image_read.log',
#                    format='%(asctime)s-%(levelname)s:%(message)s',
#                    datefmt='%d/%m/%Y %H:%M:%S',
#                    level=logging.DEBUG)


# =============================================================================
# Chapter 2: Define Class for image manipulation
# =============================================================================
class zatanna():
    '''
    Class to read in images from a folder,
    resize/refit them to a window,
    pad them and wait
    '''
    def __init__(self, location=False, img_rows=28, img_cols=28):
        '''
        Initialize class
        '''
        logging.info('Getting Location Info')
        self.location = location

        logging.info('Passing Image Dimensions')
        self.img_rows = img_rows
        self.img_cols = img_cols

    def read_images(self):
        '''
        Reading Images from file
        '''
        pass

    def resize_images(self):
        '''
        Resize Images to a uniform size
        '''
        pass

    def cut_images(self):
        '''
        Cut images to fit a frame
        '''
        pass

    def greyscale(self):
        '''
        Convert all images to greyscale
        '''

    def use_mnist(self):
        '''
        Utility Function to pass through MNIST dataset
        '''
        logging.info('Getting mnist data')
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        logging.info('Reshaping to Keras usable dimensions')
        X_train = X_train.reshape(X_train.shape[0],
                                  self.img_rows, self.img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0],
                                self.img_rows, self.img_cols, 1)

        logging.info('Converting RGB value to float')
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        logging.info('Scaling to 0-1 range')
        X_train /= 255
        X_test /= 255

        logging.info('Passing train and test as attributes of the class')
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
