import os
import json

import matplotlib.pyplot as plt

from config import PROJECT_ABSOLUTE_PATH


def generate_and_save_images(g_model, epoch, test_input):

    predictions = g_model(test_input, training=False)
    predictions = (predictions+1)/2.0   # scale from [-1,1] to [0,1]

    with open(os.path.join(PROJECT_ABSOLUTE_PATH, 'params.json'), 'r') as params_json:
        params = json.load(params_json)
    width = params['image_width_test']/100.0 # todo change
    height = params['image_height_test']/100.0 # todo change
    fig = plt.figure(figsize=(width, height))
    # plot images
    '''for i in range(predictions.shape[0]):
        # define subplot
        plt.subplot(4, 4, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(predictions[i])'''
    plt.imshow(predictions[0])
    # save plot to file
    plt.savefig(os.path.join(PROJECT_ABSOLUTE_PATH, f'images_generated/image_epoch_{epoch}.png'))
    plt.show()
    plt.close()
