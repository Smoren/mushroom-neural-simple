import random

import activation
from structs import NeuralNetwork
import tools

DATA_FILE_NAME = 'output/draw.json'
INPUT_COUNT = 45
OUTPUT_COUNT = 4
TRAIN_EPOCH_COUNT = 150
ANSWERS = {
    'circle': 0,
    'square': 1,
    'rhombus': 2,
    'triangle': 3
}

nn = NeuralNetwork()  # создаем нейронную сеть
nn.add_input_layer(INPUT_COUNT)  # добавляем входной слой
nn.add_layer(30, random_radius=2)  # добавляем скрытый слой
nn.add_layer(20, random_radius=2)  # добавляем скрытый слой
nn.add_layer(10, random_radius=2)  # добавляем скрытый слой
nn.add_layer(OUTPUT_COUNT)  # добавляем выходной слой

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    dirty_data = []

    input_list = tools.import_json_file('input/draw/input.json')

    train_data = []

    for key in input_list:
        answer = ANSWERS[key]
        output_data = [0 for i in range(OUTPUT_COUNT)]
        output_data[answer] = 1

        i = 0
        for input_data in input_list[key]:
            if len(input_data) != INPUT_COUNT:
                print(i, len(input_data))
            train_data.append([input_data, output_data])
            i += 1

    epoch_losses = []

    # выполняем заданное количество эпох обучения
    for i in range(0, TRAIN_EPOCH_COUNT):
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(train_data, 0.1, False)
        epoch_losses.append(loss_total)
        print('LOSS: {:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

    epoch_losses_formatted = []
    for i in range(len(epoch_losses)):
        loss = epoch_losses[i]
        epoch_losses_formatted.append('({};{}) '.format(i, loss))

    tools.export_file('output/draw_losses.txt', ''.join(epoch_losses_formatted))
