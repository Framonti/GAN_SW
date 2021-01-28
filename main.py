import os
import json
import sys

import tensorflow as tf
from keras.datasets.cifar10 import load_data

from models import *
from GAN_training import train
from config import PROJECT_ABSOLUTE_PATH

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # change or not depending on your machine


def gan_train():
    (train_images, _), (_, _) = load_data()  # todo change example dataset into real dataset
    train_images = train_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize from [0,255] to [-1,1]

    with open(os.path.join(PROJECT_ABSOLUTE_PATH, 'params.json'), 'r') as param_file:
        params = json.load(param_file)

    BUFFER_SIZE = params['BUFFER_SIZE']
    BATCH_SIZE = params['BATCH_SIZE']
    EPOCHS = params['EPOCHS']
    noise_dimension = params['noise_dimension']
    num_examples_to_generate = params['num_examples_to_generate_train']
    seed = tf.random.normal(
        [num_examples_to_generate, noise_dimension])  # generates random normal distribution for seed

    # Batch and shuffle the data todo change
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = build_generator(32, 32) # todo change here
    discriminator = build_discriminator()

    train(train_dataset, seed, EPOCHS, BATCH_SIZE, noise_dimension, generator, discriminator, generator_loss,
          discriminator_loss, generator_optimizer, discriminator_optimizer)


# create_gif('gif/test.gif')

def generate(n):
    pass


if __name__ == '__main__':
    train_or_generate = sys.argv[1]
    if train_or_generate == 'train':
        gan_train()
    elif train_or_generate == 'generate':
        number_to_generate = sys.argv[2]
        generate(number_to_generate)
    else:
        raise ValueError('Parameters not recognized')
