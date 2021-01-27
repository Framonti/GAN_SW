from models import *
import tensorflow as tf
import os
import json
from GAN_training import train
from keras.datasets.cifar10 import load_data
from config import PROJECT_ABSOLUTE_PATH

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # change or not depending on your machine

(train_images, _), (_, _) = load_data()


'''
# todo change example dataset into real dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
'''


# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize from [0,255] to [-1,1]

with open(os.path.join(PROJECT_ABSOLUTE_PATH, 'params.json'), 'r') as param_file:
    params = json.load(param_file)

BUFFER_SIZE = params['BUFFER_SIZE']
BATCH_SIZE = params['BATCH_SIZE']
EPOCHS = params['EPOCHS']
noise_dimension = params['noise_dimension']
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dimension]) # generates random normal distribution for seed


'''# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y'''


# Batch and shuffle the data todo change
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = build_generator(32, 32)
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

train(train_dataset, seed, EPOCHS, BATCH_SIZE, noise_dimension, generator, discriminator, generator_loss,
      discriminator_loss, generator_optimizer, discriminator_optimizer)

# create_gif('gif/test.gif')
