import os
import json

import matplotlib.pyplot as plt

from config import PROJECT_ABSOLUTE_PATH, CONFIG_ABSOLUTE_PATH
from generate_data import generate_fake_samples


def generate_and_save_images(g_model, latent_dim, name, training, n_samples=1):
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    X = (X + 1) / 2.0  # scale from [-1,1] to [0,1]

    with open(os.path.join(CONFIG_ABSOLUTE_PATH, 'params.json'), 'r') as params_json:
        params = json.load(params_json)
    width = params['image_width'] / 100.0
    height = params['image_height'] / 100.0
    fig = plt.figure(figsize=(width, height), frameon=False)
    # plot image
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(X[0], aspect='auto')
    # save plot to file
    if training:
        save_path = os.path.join(PROJECT_ABSOLUTE_PATH, f'images_training/image_epoch_{name}.png')
    else:
        save_path = os.path.join(PROJECT_ABSOLUTE_PATH, f'images_generation/image_{name}.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-real')
    plt.plot(d2_hist, label='d-fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
    # plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(a1_hist, label='acc-real')
    plt.plot(a2_hist, label='acc-fake')
    plt.legend()
    # save plot to file
    plt.savefig(os.path.join(PROJECT_ABSOLUTE_PATH, f'performances/plot_line_plot_loss.png'))
    plt.close()
