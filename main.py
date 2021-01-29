import os
import json
import sys

import tensorflow as tf
from keras.models import load_model

from models import *
from GAN_training import train, train_refactor
from config import CONFIG_ABSOLUTE_PATH, PROJECT_ABSOLUTE_PATH
from image_generation import generate_and_save_images
from load_data import load_data, preprocess_data

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # change or not depending on your machine


def gan_train():
    train_images = load_data()
    train_images = preprocess_data(train_images)

    with open(os.path.join(CONFIG_ABSOLUTE_PATH, 'params.json'), 'r') as param_file:
        params = json.load(param_file)

    BUFFER_SIZE = params['BUFFER_SIZE']
    BATCH_SIZE = params['BATCH_SIZE']
    EPOCHS = params['EPOCHS']
    noise_dimension = params['noise_dimension']
    num_examples_to_generate = params['num_examples_to_generate_train']
    image_width = params['image_width']
    image_height = params['image_height']
    seed = tf.random.normal([num_examples_to_generate, noise_dimension])  # generates random normal distribution for seed

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = build_generator(image_width, image_height, BATCH_SIZE)
    discriminator = build_discriminator((image_height, image_width, 3))

    train(train_dataset, seed, EPOCHS, BATCH_SIZE, noise_dimension, generator, discriminator, generator_loss,
          discriminator_loss, generator_optimizer, discriminator_optimizer)


def gan_train_refactor():
    train_images = load_data()
    train_images = preprocess_data(train_images)

    with open(os.path.join(CONFIG_ABSOLUTE_PATH, 'params.json'), 'r') as param_file:
        params = json.load(param_file)

    n_batch = params['n_batch']
    n_epochs = params['n_epochs']
    latent_dimension = params['latent_dimension']
    # num_examples_to_generate = params['num_examples_to_generate_train']
    image_width = params['image_width']
    image_height = params['image_height']

    generator = build_generator_refactor(image_width, image_height, latent_dimension)
    discriminator = build_discriminator_refactor((image_height, image_width, 3))
    gan_model = define_gan(generator, discriminator)

    train_refactor(generator, discriminator, gan_model, train_images, latent_dimension, n_epochs, n_batch)


def generate(n):
    load_path = os.path.join(PROJECT_ABSOLUTE_PATH, 'saved_generator')
    generator = load_model(load_path, compile=False)

    with open(os.path.join(CONFIG_ABSOLUTE_PATH, 'params.json'), 'r') as param_file:
        params = json.load(param_file)

    noise_dimension = params['noise_dimension']
    num_examples_to_generate = params['num_examples_to_generate_train']

    for i in range(n):
        seed = tf.random.normal([num_examples_to_generate, noise_dimension])
        generate_and_save_images(generator, seed, i, training=False)


if __name__ == '__main__':
    train_or_generate = sys.argv[1]
    if train_or_generate == 'train':
        gan_train_refactor()
    elif train_or_generate == 'generate':
        number_to_generate = int(sys.argv[2])
        generate(number_to_generate)
    else:
        raise ValueError('Parameters not recognized')
