# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:42:51 2018

@author: Shashwat Pathak
"""

# =============================================================================
# Chapter 0: Import Modules
# =============================================================================
import logging
import numpy as np

from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.core import Dense, Activation, Reshape, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from src.visualization.visualize import plot_loss, plot_gen

# =============================================================================
# Chapter 1: Setting Logging Parameters
# =============================================================================
#logging.basicConfig(filename='reports/model_train.log',
#                    format='%(asctime)s-%(levelname)s:%(message)s',
#                    datefmt='%d/%m/%Y %H:%M:%S',
#                    level=logging.DEBUG)


# =============================================================================
# Chapter 2: Class to contain Generator and Discriminator
# =============================================================================
class starfire():
    '''
    Class to contain the generator and discriminator models
    and define methods to train and predict them
    '''
    def __init__(self, data_shape):
        '''
        Initialize class
        '''
        logging.info('Setting parameters of the models')
        self.dropout = 0.25
        self.gen_lr = 1e-3
        self.dis_lr = 1e-4
        self.gen_nodes = 200
        self.gen_inputs = 100
        self.losses = {'D': [], 'G': []}
        self.data_shape = data_shape
        self.generator()
        self.discriminator()
        self.gan()

    def generator(self):
        '''
        Architecture of the generator model
        '''
        g_input = Input(shape=[self.gen_inputs])
        Kory = Dense(self.gen_nodes*14*14,
                     kernel_initializer='glorot_normal')(g_input)
        Kory = BatchNormalization()(Kory)
        Kory = Activation('relu')(Kory)
        Kory = Reshape([14, 14, self.gen_nodes])(Kory)
        Kory = UpSampling2D(size=(2, 2))(Kory)
        Kory = Convolution2D(int(self.gen_nodes/2), 3, 3,
                             border_mode='same',
                             kernel_initializer='glorot_uniform')(Kory)
        Kory = BatchNormalization()(Kory)
        Kory = Activation('relu')(Kory)
        Kory = Convolution2D(int(self.gen_nodes/4), 3, 3,
                             border_mode='same',
                             kernel_initializer='glorot_uniform')(Kory)
        Kory = BatchNormalization()(Kory)
        Kory = Activation('relu')(Kory)
        Kory = Convolution2D(1, 1, 1,
                             border_mode='same',
                             kernel_initializer='glorot_uniform')(Kory)
        g_V = Activation('sigmoid')(Kory)
        generator = Model(g_input, g_V)
        generator.compile(loss='binary_crossentropy',
                          optimizer=Adam(self.gen_lr))
        self.kory = generator
#        logging.debug(generator.summary())

    def discriminator(self):
        '''
        Architecture of the discriminator model
        '''
        d_input = Input(shape=self.data_shape)
        Anders = Convolution2D(256, (5, 5),
                               subsample=(2, 2),
                               border_mode='same',
                               activation='relu')(d_input)
        Anders = LeakyReLU(0.2)(Anders)
        Anders = Dropout(self.dropout)(Anders)
        Anders = Convolution2D(512, (5, 5),
                               subsample=(2, 2),
                               border_mode='same',
                               activation='relu')(Anders)
        Anders = LeakyReLU(0.2)(Anders)
        Anders = Dropout(self.dropout)(Anders)
        Anders = Flatten()(Anders)
        Anders = Dense(256)(Anders)
        Anders = LeakyReLU(0.2)(Anders)
        Anders = Dropout(self.dropout)(Anders)
        d_V = Dense(2, activation='softmax')(Anders)
        discriminator = Model(d_input, d_V)
        discriminator.compile(loss='categorical_crossentropy',
                              optimizer=Adam(self.dis_lr))
        self.anders = discriminator

    def gan(self):
        '''
        Compiling generator and discriminator together
        '''
        logging.info('Setting discriminator to non-trainable by default')
        self.make_trainable(False)
        logging.info('Compiling GAN')
        gan_input = Input(shape=[self.gen_inputs])
        Kory_Anders = self.kory(gan_input)
        gan_V = self.anders(Kory_Anders)
        GAN = Model(gan_input, gan_V)
        GAN.compile(loss='categorical_crossentropy',
                    optimizer=Adam(self.gen_lr))
        self.koriandr = GAN

    def pretrain_discriminator(self, train):
        '''
        Pre-train the discriminator model
        '''
        logging.info('Generating random noise to feed to generator')
        noise_gen = np.random.uniform(0, 1, size=[train.shape[0], 100])
        logging.info('Generating Images')
        generated_images = self.kory.predict(noise_gen)
        logging.debug('Shape of gen images {}'.format(generated_images.shape))
        logging.debug('Shape of train images {}'.format(train.shape))
        logging.info('Concatenating actual data with generated images')
        X = np.concatenate((train, generated_images))
        logging.info('Setting up 1-hot encoded annotation for ' +
                     'real and generated images')
        n = train.shape[0]
        y = np.zeros([2*n, 2])
        y[:n, 1] = 1
        y[n:, 0] = 1

        logging.info('Unfreeze weights for pre-training')
        self.make_trainable(True)
        logging.info('Train on generated data')
        self.anders.fit(X, y, nb_epoch=1, batch_size=32)

        logging.info('Getting pre-train accuracy results')
        y_hat = self.anders.predict(X)
        y_hat_idx = np.argmax(y_hat, axis=1)
        y_idx = np.argmax(y, axis=1)
        diff = y_idx - y_hat_idx
        n_tot = y.shape[0]
        n_rig = (diff == 0).sum()
        acc = n_rig*100.0/n_tot
        logging.debug("Accuracy: %0.02f pct (%d of %d) classified correctly"
                      % (acc, n_rig, n_tot))
        logging.info('Saving pre-trained model')
        self.anders.save('models/pretrained_anders.h5')

    def train_gan(self, X_train, nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
        '''
        Train GAN
        '''
        for e in range(nb_epoch):

            logging.info('Generating images')
            image_batch = X_train[np.random.randint(0,
                                                    X_train.shape[0],
                                                    size=BATCH_SIZE), :, :, :]
            noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
            generated_images = self.kory.predict(noise_gen)

            logging.info('Training discriminator on generated images')
            X = np.concatenate((image_batch, generated_images))
            y = np.zeros([2*BATCH_SIZE, 2])
            y[0:BATCH_SIZE, 1] = 1
            y[BATCH_SIZE:, 0] = 1

            self.make_trainable(True)
            d_loss = self.anders.train_on_batch(X, y)
            self.losses["D"].append(d_loss)

            logging.info('Training Generator-Discriminator ' +
                         'stack on input noise to non-generated output class')
            noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
            y2 = np.zeros([BATCH_SIZE, 2])
            y2[:, 1] = 1

            self.make_trainable(False)
            g_loss = self.koriandr.train_on_batch(noise_tr, y2)
            self.losses["G"].append(g_loss)

            if e % plt_frq == plt_frq-1:
                logging.info('Updating plots')
                plot_loss(self.losses)
                plot_gen(self.kory)
                logging.info('Saving Models')
                self.kory.save_weights('models/kory.h5')
                self.anders.save_weights('models/anders.h5')
                self.koriandr.save_weights('models/koriandr.h5')

    def make_trainable(self, val):
        '''
        Changing the trainable property of models
        '''
        if val:
            logging.info('Changing Discriminator to True')
        else:
            logging.info('Changing Discriminator to False')
        self.anders.trainable = val
        for l in self.anders.layers:
            self.anders.trainable = val

    def load(self):
        '''
        Load weights from local
        '''
        self.kory.load_weights('models/kory.h5')
        self.anders.load_weights('models/anders.h5')
        self.koriandr.load_weights('models/koriandr.h5')
