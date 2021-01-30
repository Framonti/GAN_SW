import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, \
    Flatten

from keras.initializers import RandomNormal


def build_generator(output_width, output_height, latent_dim):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    model = tf.keras.Sequential()

    starting_width = output_width // 16  # right now: 8
    starting_height = output_height // 16  # right now 5
    n_nodes = starting_width * starting_height * 256

    # 128x80 image
    model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))

    # starting size: (5, 8, 256)
    model.add(Reshape((starting_height, starting_width, 256)))

    # upsample to (10, 16, 128)
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to (20, 32, 128)
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to (40, 64, 128)
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to (80, 128, 64)
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))

    # output 128x80x3
    model.add(Conv2D(3, (3, 3), padding='same', activation='tanh', kernel_initializer=init))  # todo (3,3)?
    return model


def build_discriminator(input_shape):
    # right now input_shape=(80, 128, 3)
    init = RandomNormal(stddev=0.02)
    model = tf.keras.Sequential()
    # downsample to (80, 128, 64)
    model.add(Conv2D(64, (4, 4), padding='same', input_shape=input_shape, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))

    # downsample to (40, 64, 128)
    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))

    # downsample to (20, 32, 128)
    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))

    # downsample to (10, 16, 256)
    model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))

    # downsample to (5, 8, 256)
    model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))

    # classifier
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
