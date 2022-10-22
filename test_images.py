from random import shuffle

import numpy as np

import activation
import tools
from img_tools import open_img, save_img
from structs import NeuralNetwork

DATA_FILE_NAME = 'output/images.json'

nn = NeuralNetwork()  # создаем нейронную сеть
nn.add_input_layer(3)  # добавляем входной слой
nn.add_layer(64, activation_class=activation.ActivationRelu, random_radius=0.5, use_bias=True)
nn.add_layer(64, activation_class=activation.ActivationSigmoid, random_radius=0.5, use_bias=True)
nn.add_layer(3, activation_class=activation.ActivationSigmoid, random_radius=0.5, use_bias=True)

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали
input_img = open_img('input/images/1.jpg')
train_data = []

if False and import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    # готовим обучающую выборку
    for y in range(input_img.shape[0]):
        for x in range(input_img.shape[1]):
            pixel = input_img[y][x]
            train_data.append([
                [y/input_img.shape[0], x/input_img.shape[1], 1],
                [pixel[0]/256, pixel[1]/256, pixel[2]/256]
            ])

    # выполняем заданное количество эпох обучения
    for i in range(0, 100):
        shuffle(train_data)
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(train_data, 0.7, False)
        print('LOSS: {:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

# for (y, x), pixel in train_data:
#     y = int(y*input_img.shape[0])
#     x = int(x*input_img.shape[1])
#     input_img[y][x][0] = pixel[0]*256
#     input_img[y][x][1] = pixel[1]*256
#     input_img[y][x][2] = pixel[2]*256

for y in range(input_img.shape[0]):
    for x in range(input_img.shape[1]):
        pixel = nn.run([y/input_img.shape[0], x/input_img.shape[1], 1]).get_output()
        input_img[y][x][0] = pixel[0]*256
        input_img[y][x][1] = pixel[1]*256
        input_img[y][x][2] = pixel[2]*256

my_channels = []
for i in range(3):
    channel = input_img[:, :, i]
    my_channels.append(channel)

save_img('output/test.jpg', my_channels)
