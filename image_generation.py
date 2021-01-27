import matplotlib.pyplot as plt
import PIL
import glob
import imageio
import os
from config import PROJECT_ABSOLUTE_PATH


# todo change a lot this one
'''def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)

    #  Plot the generated images
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')
    # 3 - Save the generated images
    plt.savefig(f'images_generated/image_epoch_{epoch}.png')
    plt.show()'''


def generate_and_save_images(g_model, epoch, test_input):

    predictions = g_model(test_input, training=False)
    predictions = (predictions+1)/2.0   # scale from [-1,1] to [0,1]

    # plot images
    for i in range(predictions.shape[0]):
        # define subplot
        plt.subplot(4, 4, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(predictions[i])
    # save plot to file
    plt.savefig(os.path.join(PROJECT_ABSOLUTE_PATH, f'images_generated/image_epoch_{epoch}.png'))
    plt.show()
    plt.close()


def display_image(image_name):
    return PIL.Image.open(f'{image_name}.png')


def create_gif(gif_name):
    with imageio.get_writer(gif_name, mode='I') as writer:
        filenames = glob.glob('image*.png')  # todo change here
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

        # imageio.

