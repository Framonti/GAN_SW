import numpy as np
from numpy.random import randint, randn


def select_real_samples(dataset, n_samples):        # todo n_samples = batch_size?
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):  # todo n_samples = batch_size?
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = np.zeros((n_samples, 1))
    return X, y
