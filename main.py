from models import *
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from GAN_training import train, configure_checkpoints

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# todo change example dataset into real dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

with open('params.json', 'r') as param_file:
    params = json.load(param_file)

BUFFER_SIZE = params['BUFFER_SIZE']
BATCH_SIZE = params['BATCH_SIZE']
EPOCHS = params['EPOCHS']
noise_dimension = params['noise_dimension']
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dimension]) # generates random normal distribution for seed

# Batch and shuffle the data todo change
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = build_generator()
discriminator = build_discriminator()

"""
noise = tf.random.normal([1, noise_dimension])
generated_image = generator(noise, training=False)
# Visualize the generated sample
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

decision = discriminator(generated_image)
print(decision)
"""

checkpoint, checkpoint_manager = configure_checkpoints(generator_optimizer, discriminator_optimizer,
                                                       generator, discriminator)
train(train_dataset, seed, EPOCHS, BATCH_SIZE, noise_dimension, generator, discriminator, generator_loss,
      discriminator_loss, generator_optimizer, discriminator_optimizer)