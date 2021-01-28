import os
import json

import matplotlib.pyplot as plt

from config import PROJECT_ABSOLUTE_PATH, CONFIG_ABSOLUTE_PATH


def generate_and_save_images(g_model, test_input, name, training):

    predictions = g_model(test_input, training=False)
    predictions = (predictions+1)/2.0   # scale from [-1,1] to [0,1]

    with open(os.path.join(CONFIG_ABSOLUTE_PATH, 'params.json'), 'r') as params_json:
        params = json.load(params_json)
    width = params['image_width']/100.0
    height = params['image_height']/100.0
    fig = plt.figure(figsize=(width, height), frameon=False)
    # plot image
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(predictions[0], aspect='auto')
    # save plot to file
    if training:
        save_path = os.path.join(PROJECT_ABSOLUTE_PATH, f'images_training/image_epoch_{name}.png')
    else:
        save_path = os.path.join(PROJECT_ABSOLUTE_PATH, f'images_generation/image_{name}.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()
