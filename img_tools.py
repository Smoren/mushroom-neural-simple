import numpy as np
from PIL import Image


def open_img(file_name: str) -> np.array:
    img = Image.open(file_name)
    return np.array(img)


def save_img(file_name: str, channels):
    img_data = np.zeros((channels[0].shape[0], channels[0].shape[1], 3)).astype(np.uint8)
    for i in range(len(channels)):
        img_data[:, :, i] = channels[i]

    new_img = Image.fromarray(img_data)
    new_img.save(file_name)
    new_img.show()
