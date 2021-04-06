import activation
from structs import NeuralNetwork
import tools

DATA_FILE_NAME = 'output/simple.json'

nn = NeuralNetwork()  # создаем нейронную сеть
nn.add_input_layer(4)  # добавляем входной слой
nn.add_layer(15, activation_class=activation.ActivationRelu, random_radius=0.15)  # добавляем скрытый слой
nn.add_layer(15, activation_class=activation.ActivationRelu, random_radius=0.15)  # добавляем скрытый слой
nn.add_layer(3, activation_class=activation.ActivationGauss, random_radius=2)  # добавляем скрытый слой
nn.add_layer(10, activation_class=activation.ActivationSigmoid, random_radius=2)  # добавляем скрытый слой
nn.add_layer(3, activation_class=activation.ActivationSigmoid, random_radius=2)  # добавляем выходной слой

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    # готовим обучающую выборку
    train_data = [
        [[0, 0, 0, 0], [0, 0, 0]],
        [[0, 0, 0, 1], [0, 0, 1]],
        [[0, 0, 1, 0], [0, 0, 1]],
        [[0, 0, 1, 1], [0, 1, 0]],
        [[0, 1, 0, 0], [0, 1, 0]],
        [[0, 1, 0, 1], [0, 1, 0]],
        [[0, 1, 1, 0], [0, 1, 0]],
        [[0, 1, 1, 1], [0, 1, 1]],
        [[1, 0, 0, 0], [0, 0, 1]],
        [[1, 0, 0, 1], [0, 1, 0]],
        [[1, 0, 1, 0], [0, 1, 0]],
        [[1, 0, 1, 1], [0, 1, 1]],
        [[1, 1, 0, 0], [0, 1, 0]],
        [[1, 1, 0, 1], [0, 1, 1]],
        [[1, 1, 1, 0], [0, 1, 1]],
        [[1, 1, 1, 1], [1, 0, 0]],
    ]

    # выполняем заданное количество эпох обучения
    for i in range(0, 2000):
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(train_data, 0.1, False)
        print('{:.4f}'.format(loss_total))

    # выполняем заданное количество эпох обучения
    for i in range(0, 2000):
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(train_data, 0.05, False)
        print('{:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

nn.run([0, 0, 0, 0])  # выставляем на входы сигналы и выполняем прямой проход
print(0)
print(nn.get_output())
print()

nn.run([0, 0, 1, 0])  # выставляем на входы сигналы и выполняем прямой проход
print(1)
print(nn.get_output())
print()

nn.run([1, 0, 1, 0])  # выставляем на входы сигналы и выполняем прямой проход
print(2)
print(nn.get_output())
print()

nn.run([1, 1, 0, 1])  # выставляем на входы сигналы и выполняем прямой проход
print(3)
print(nn.get_output())
print()

nn.run([1, 1, 1, 1])  # выставляем на входы сигналы и выполняем прямой проход
print(4)
print(nn.get_output())
print()
