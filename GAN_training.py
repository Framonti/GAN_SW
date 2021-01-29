import os
import time

import tensorflow as tf
import numpy as np

from image_generation import generate_and_save_images, generate_and_save_images_refactor
from config import PROJECT_ABSOLUTE_PATH
from generate_data import generate_latent_points, select_real_samples, generate_fake_samples


def configure_checkpoints(generator_optimizer, discriminator_optimizer, generator, discriminator):
    checkpoint_dir = os.path.join(PROJECT_ABSOLUTE_PATH, 'checkpoints')
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    return checkpoint, manager


def configure_checkpoints_refactor(generator, discriminator):
    checkpoint_dir = os.path.join(PROJECT_ABSOLUTE_PATH, 'checkpoints')
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    return checkpoint, manager


# tf.function annotation causes the function to be "compiled" as part of the training
@tf.function
def train_step(images, batch_size, noise_dimension, generator, discriminator, generator_loss, discriminator_loss,
               generator_optimizer, discriminator_optimizer):
    # Create a random noise to feed it into the model for the image generation
    noise = tf.random.normal([batch_size, noise_dimension])

    # Generate images and calculate loss values
    # GradientTape method records operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calculate gradients using loss values and model variables
    # "gradient" method computes the gradient using
    # operations recorded in context of this tape (gen_tape and disc_tape).

    # It accepts a target (e.g., gen_loss) variable and
    # a source variable (e.g.,generator.trainable_variables)
    # target --> a list or nested structure of Tensors or Variables to be differentiated.
    # source --> a list or nested structure of Tensors or Variables.
    # target will be differentiated against elements in sources.

    # "gradient" method returns a list or nested structure of Tensors
    # (or IndexedSlices, or None), one for each element in sources.
    # Returned structure is the same as the structure of sources.
    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

    # Process gradients and run the optimizers
    # "apply_gradients" method processes aggregated gradients.
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# from IPython import display # A command shell for interactive computing in Python.
def train(dataset, seed, epochs, batch_size, noise_dimension, generator, discriminator, generator_loss,
          discriminator_loss,
          generator_optimizer, discriminator_optimizer):
    checkpoint, manager = configure_checkpoints(generator_optimizer, discriminator_optimizer, generator, discriminator)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, batch_size, noise_dimension, generator, discriminator, generator_loss,
                       discriminator_loss, generator_optimizer, discriminator_optimizer)

        # 2 - Produce intermediate images
        generate_and_save_images(generator, seed, epoch+1, training=True)

        # Save the model every 5 epochs as a checkpoint
        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 5 == 0:
            save_path = manager.save()
            print(f"Saved checkpoint for step {int(checkpoint.step)}: {save_path}")

        # Print out the completed epoch no. and the time spent
        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

    # Generate a final image after the training is completed
    generate_and_save_images(generator, seed, epochs, training=True) # name
    save_generator_path = os.path.join(PROJECT_ABSOLUTE_PATH, 'saved_generator')
    generator.save(save_generator_path)


def train_refactor(generator, discriminator, gan_model, dataset, latent_dim, n_epochs, n_batch):
    checkpoint, manager = configure_checkpoints_refactor(generator, discriminator)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    batches_per_epoch = int(dataset.shape[0] / n_batch)
    # calculate the total iterations based on batch and epoch
    n_steps = batches_per_epoch * n_epochs
    # calculate the number of samples in half a batch
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
            generate_and_save_images_refactor(generator, latent_dim, current_epoch+1, training=True)
            print(f'Time for epoch {current_epoch + 1} is {time.time() - start} sec')
            current_epoch += 1
            checkpoint.step.assign_add(1)
            if int(checkpoint.step) % 5 == 0:
                save_path = manager.save()
                print(f"Saved checkpoint for step {int(checkpoint.step)}: {save_path}")
            start = time.time()

    generate_and_save_images_refactor(generator, latent_dim, n_epochs, training=True)  # name
    save_generator_path = os.path.join(PROJECT_ABSOLUTE_PATH, 'saved_generator')
    generator.save(save_generator_path)