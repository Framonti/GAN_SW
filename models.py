import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, \
    Flatten


def build_generator(output_width, output_height, batch_size):
    """
    Accepts 1D arrays and outputs 1260x800 pixels images. # todo not yet
    # acutal input: 128x80
    :return:
    """
    starting_width = output_width // 16  # right now: 8
    starting_height = output_height // 16  # right now 5

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer((100,), batch_size=batch_size))
    model.add(Dense(starting_height * starting_width * 256, input_shape=(100,)))  # is 256 the batch size??
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # reshape 1D into 2D;
    model.add(Reshape((starting_height, starting_width, 256)))
    # assert model.output_shape == (None, 5, 8, 256)

    # Conv2DTranspose:  increase the size of a smaller array
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    # assert model.output_shape == (None, 10, 16, 128)
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    # assert model.output_shape == (None, 20, 32, 128)
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    # assert model.output_shape == (None, 40, 64, 128)
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    # assert model.output_shape == (None, 80, 128, 128)
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (3, 3), padding='same', activation='tanh'))
    #  assert model.output_shape == (None, 80, 128, 3)

    return model


def build_discriminator(input_shape):
    model = tf.keras.Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model


generator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
