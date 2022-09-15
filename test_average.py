import math
import random
import activation
from structs import NeuralNetwork
import tools


def function_to_approx(a, b):
    return math.sqrt((a + b) / 2)


def train_data_generator(count):
    for i in range(count):
        a = random.random()
        b = random.random()
        yield [[a, b], [function_to_approx(a, b)]]


def print_result(expected, actual):
    print('{} == {}'.format(round(expected, 4), round(actual, 4)))


DATA_FILE_NAME = 'output/average.json'

nn = NeuralNetwork()  # создаем нейронную сеть
nn.add_input_layer(2)  # добавляем входной слой
nn.add_layer(8, activation_class=activation.ActivationRelu, random_radius=2, use_bias=True)  # добавляем скрытый слой
nn.add_layer(8, activation_class=activation.ActivationSigmoid, random_radius=2, use_bias=True)  # добавляем скрытый слой
nn.add_layer(8, activation_class=activation.ActivationSigmoid, random_radius=2, use_bias=True)  # добавляем скрытый слой
nn.add_layer(8, activation_class=activation.ActivationGauss, random_radius=2, use_bias=True)  # добавляем скрытый слой
nn.add_layer(1, activation_class=activation.ActivationRelu, random_radius=2)  # добавляем выходной слой

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    gen = train_data_generator(1000)
    train_data = [x for x in gen]

    # выполняем заданное количество эпох обучения
    for i in range(0, 100):
        print('')
        print('EPOCH #{}'.format(i))

        if i % 10 == 0:
            loss_total = nn.train(train_data_generator(1000), 0.2, False)
        else:
            loss_total = nn.train(train_data, 0.2, False)

        print('LOSS: {:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

for [[a, b], [value]] in train_data_generator(10):
    nn.run([a, b])  # выставляем на входы сигналы и выполняем прямой проход
    print_result(value, nn.get_output()[0])

test_data = [
    [[0.10, 0.20], function_to_approx(0.10, 0.20)],
    [[0.50, 0.20], function_to_approx(0.50, 0.20)],
    [[0.1, 0.3], function_to_approx(0.1, 0.3)],
    [[0.10, 0.10], function_to_approx(0.10, 0.10)],
    [[0.100, 0.002], function_to_approx(0.100, 0.002)],
]

for [[a, b], value] in test_data:
    nn.run([a, b])  # выставляем на входы сигналы и выполняем прямой проход
    print_result(value, nn.get_output()[0])

