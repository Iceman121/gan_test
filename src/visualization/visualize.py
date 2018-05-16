# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:46:00 2018

@author: Shashwat Pathak
"""

# =============================================================================
# Chapter 0: Import Modules
# =============================================================================
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# =============================================================================
# Chapter 1: Setting Logging Parameters
# =============================================================================
#logging.basicConfig(filename='reports/plot.log',
#                    format='%(asctime)s-%(levelname)s:%(message)s',
#                    datefmt='%d/%m/%Y %H:%M:%S',
#                    level=logging.DEBUG)


# =============================================================================
# Chapter 2: Plot Losses of the Network
# =============================================================================
def plot_loss(losses):
    '''
    Plot the losses of the network
    '''
    logging.info('Plotting the Losses of the' +
                 'Discriminative and Generative Models')
    plt.figure(figsize=(10, 8))
    plt.plot(losses["D"], label='Discriminitive loss')
    plt.plot(losses["G"], label='Generative loss')
    plt.legend()
    plt.savefig('reports/figures/losses.png')
    plt.close()


# =============================================================================
# Chapter 3: Plot generated Images
# =============================================================================
def plot_gen(generator, n_ex=16, dim=(4, 4), figsize=(10, 10)):
    '''
    Plot the generated images
    '''
    logging.info('Generaring noise to seed the image')
    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    logging.info('Passing through image generator')
    generated_images = generator.predict(noise)

    logging.info('Plotting the image generated')
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = generated_images[i, :, :, 0].reshape([28, 28])
        plt.imshow(img)
        plt.axis('off')
    plt.savefig('reports/figures/generated.png')
    plt.close()
