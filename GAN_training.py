import os
import time

import tensorflow as tf
import numpy as np

from image_generation import generate_and_save_images, plot_history
from config import PROJECT_ABSOLUTE_PATH
from generate_data import generate_latent_points, select_real_samples, generate_fake_samples


def configure_checkpoints(generator, discriminator):
    checkpoint_dir = os.path.join(PROJECT_ABSOLUTE_PATH, 'checkpoints')
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    return checkpoint, manager


def train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs, n_batch):

    checkpoint, manager = configure_checkpoints(generator, discriminator)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    batches_per_epoch = int(dataset.shape[0] / n_batch)
    n_steps = batches_per_epoch * n_epochs
    half_batch = int(n_batch / 2)

    # prepare lists for storing stats each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()

    start = time.time()
    current_epoch = 0
    for i in range(n_steps):
        # get randomly selected 'real' samples
        X_real, y_real = select_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_loss1, d_acc1 = discriminator.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, d_acc2 = discriminator.train_on_batch(X_fake, y_fake)
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)

        # record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        a1_hist.append(d_acc1)
        a2_hist.append(d_acc2)
        # evaluate the model performance every 'epoch'
        if (i + 1) % batches_per_epoch == 0:
            generate_and_save_images(generator, latent_dim, current_epoch+1, training=True)
            print(f'Time for epoch {current_epoch + 1} is {time.time() - start} sec')
            current_epoch += 1
            checkpoint.step.assign_add(1)
            if int(checkpoint.step) % 5 == 0:
                save_path = manager.save()
                print(f"Saved checkpoint for step {int(checkpoint.step)}: {save_path}")
            start = time.time()

    plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
    generate_and_save_images(generator, latent_dim, n_epochs, training=True)  # name
    save_generator_path = os.path.join(PROJECT_ABSOLUTE_PATH, 'saved_generator')
    generator.save(save_generator_path)