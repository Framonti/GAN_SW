import os
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from config import PROJECT_ABSOLUTE_PATH


def load_data_test():
    dir_name = os.path.join(PROJECT_ABSOLUTE_PATH, 'yt_thumbnails')
    num_train_samples = len(os.listdir(dir_name)) - 1  # -1 due to hidden file ".directory"

    x_train = []

    for i in range(num_train_samples):
        img_path = os.path.join(dir_name, f'{i}.jpg')
        img = load_img(img_path)
        img_array = img_to_array(img)
        x_train.append(img_array)

    x_train = np.array(x_train)
    return x_train

