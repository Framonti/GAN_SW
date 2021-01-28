import os
import time

import tensorflow as tf

from image_generation import generate_and_save_images
from config import PROJECT_ABSOLUTE_PATH


def configure_checkpoints(generator_optimizer, discriminator_optimizer, generator, discriminator):
    checkpoint_dir = os.path.join(PROJECT_ABSOLUTE_PATH, 'checkpoints')
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
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
        generate_and_save_images(generator, seed, epochs+1, training=True)

        # Save the model every 5 epochs as a checkpoint
        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 5 == 0:
            save_path = manager.save()
            print(f"Saved checkpoint for step {int(checkpoint.step)}: {save_path}")

        # Print out the completed epoch no. and the time spent
        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

    # Generate a final image after the training is completed
    # display.clear_output(wait=True)
    generate_and_save_images(generator, seed, epochs, training=True) # name
    save_generator_path = os.path.join(PROJECT_ABSOLUTE_PATH, 'saved_generator')
    generator.save(save_generator_path)
