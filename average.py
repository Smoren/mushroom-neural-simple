import math
import random
import activation
from structs import NeuralNetwork
import tools

DATA_FILE_NAME = 'output/average.json'

nn = NeuralNetwork()  # создаем нейронную сеть
nn.add_input_layer(2)  # добавляем входной слой
nn.add_layer(4, activation_class=activation.ActivationSigmoid, random_radius=2, use_bias=True)  # добавляем скрытый слой
nn.add_layer(4, activation_class=activation.ActivationSigmoid, random_radius=2, use_bias=True)  # добавляем скрытый слой
nn.add_layer(4, activation_class=activation.ActivationSigmoid, random_radius=2, use_bias=True)  # добавляем скрытый слой
nn.add_layer(4, activation_class=activation.ActivationSigmoid, random_radius=2, use_bias=True)  # добавляем скрытый слой
nn.add_layer(1, activation_class=activation.ActivationGauss, random_radius=2)  # добавляем выходной слой

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    def train_data_generator(count):
        for i in range(count):
            a = random.random()
            b = random.random()
            yield [[a, b], [math.sqrt((a + b) / 2)]]

    # выполняем заданное количество эпох обучения
    for i in range(0, 1500):
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(train_data_generator(100), 0.1, False)
        print('LOSS: {:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

nn.run([0.10, 0.20])  # выставляем на входы сигналы и выполняем прямой проход
print(round(math.sqrt(0.15), 4), nn.get_output())

nn.run([0.50, 0.20])  # выставляем на входы сигналы и выполняем прямой проход
print(round(math.sqrt(0.35), 4), nn.get_output())

nn.run([0.1, 0.3])  # выставляем на входы сигналы и выполняем прямой проход
print(round(math.sqrt(0.2), 4), nn.get_output())

nn.run([0.10, 0.10])  # выставляем на входы сигналы и выполняем прямой проход
print(round(math.sqrt(0.10), 4), nn.get_output())

nn.run([0.100, 0.002])  # выставляем на входы сигналы и выполняем прямой проход
print(round(math.sqrt(0.051), 4), nn.get_output())
