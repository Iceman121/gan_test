# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:30:27 2018

@author: Shashwat Pathak
"""

# =============================================================================
# Chapter 0: Import Modules
# =============================================================================
import logging

from src.data.read_images import zatanna
from src.models.train_model import starfire
from src.visualization.visualize import plot_gen, plot_loss

# =============================================================================
# Chapter 1: Setting Logging Parameters
# =============================================================================
logging.basicConfig(filename='reports/main.log',
                    format='%(asctime)s-%(levelname)s:%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.DEBUG)


# =============================================================================
# Chapter 2: Main
# =============================================================================
def main():
    try:
        logging.info('Setting up data')
        raw_data = zatanna()
        raw_data.use_mnist()

        logging.info('Initializing GAN')
        model = starfire(raw_data.X_train.shape[1:])
        model.pretrain_discriminator(raw_data.X_train)
        model.train_gan(raw_data.X_train)
        plot_loss(model.losses)
        plot_gen(model.kory)
    except Exception as e:
        logging.exception(e)


# =============================================================================
# Chapter 3: Call Main
# =============================================================================
if __name__ == '__main__':
    main()
