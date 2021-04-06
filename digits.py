from pprint import pprint
from glob import glob
from structs import NeuralNetwork
import tools

DATA_FILE_NAME = 'output/digits.json'
INPUT_LAYER_SIZE = 135
OUTPUT_LAYER_SIZE = 10
SX = 9
SY = 15

nn = NeuralNetwork()  # создаем нейронную сеть
nn.add_input_layer(INPUT_LAYER_SIZE)  # добавляем входной слой
nn.add_layer(36)  # добавляем скрытый слой
nn.add_layer(9)  # добавляем скрытый слой
nn.add_layer(OUTPUT_LAYER_SIZE)  # добавляем выходной слой

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали


def print_signals(signals):
    print('------------')
    for y in range(SY):
        res = ''
        for x in range(SX):
            pos = y*SX + x
            res += ' ' if signals[pos] < 0.3 else '*'
        print(res)
    print('------------')


def invert_signals(signals):
    newsignals = []

    for x in signals:
        newsignals.append(1-x)

    return newsignals


def move_signals(signals, dx, dy):
    newsignals = []
    for i in range(INPUT_LAYER_SIZE):
        newsignals.append(0)

    for y in range(SY):
        for x in range(SX):
            newy = y+dy
            newx = x+dx
            if newy < 0 or newy >= SY or newx < 0 or newx >= SX:
                continue

            if newy < 0 or newy >= SY or newx < 0 or newx >= SX:
                continue

            pos = y*SX + x
            newpos = newy*SX + newx

            newsignals[newpos] = signals[pos]

    return newsignals


if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    dirty_data = []

    # начинаем составлять обучающую выборку
    to_learn = []

    # для каждой директории с обучающими выборками (соответствующими распознаваемой цифре)
    for digit in range(10):
        # получаем список путей к файлам изображений в ней
        files = glob("input/digits/{}/*.json".format(str(digit)))

        # для каждого пути к файлу
        for img in files:
            # читаем все пиксели файла в массив
            arr = tools.import_json_file(img)

            # добавляем набор входов (пикселей) и эталон в обучающую выборку
            to_learn.append([arr, tools.get_output(digit)])
            to_learn.append([invert_signals(arr), tools.get_output(digit)])

            for dx in range(1):
                for dy in range(0, 2):
                    to_learn.append([move_signals(arr, dx, dy), tools.get_output(digit)])
                    to_learn.append([invert_signals(move_signals(arr, dx, dy)), tools.get_output(digit)])

    to_learn.append([[0 for i in range(135)], [0.5 for i in range(10)]])
    to_learn.append([[1 for i in range(135)], [0.5 for i in range(10)]])
    to_learn.append([[0.5 for i in range(135)], [0.5 for i in range(10)]])

    epoch_losses = []  # сюда накопим историю изменения ошибки с каждой эпохой

    # в цикле совершаем заданное количество эпох обучения
    for i in range(0, 1600):
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(to_learn, 0.5, False)
        epoch_losses.append(loss_total)
        print('TOTAL LOSS: {:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

    epoch_losses_formatted = []
    for i in range(len(epoch_losses)):
        loss = epoch_losses[i]
        epoch_losses_formatted.append('({};{}) '.format(i, loss))

    tools.export_file('output/digits_losses.txt', ''.join(epoch_losses_formatted))


# затем получаем пути к файлам из тестовой выборки
files = glob("input/digits/test/*.json")

# для каждого пути к файлу
for img in files:
    # читаем все пиксели файла в массив
    arr = tools.import_json_file(img)

    # подаем данные на сеть и считаем результат
    nn.run(arr)

    # выводим результат
    print()
    print(img)
    print('answer: {} | noise: {}'.format(nn.get_best_index(), round(nn.get_noise(), 4)))
    print(nn.get_output())

nn.run([0 for i in range(135)])
print()
print('ALL 0')
print('answer: {} | noise: {}'.format(nn.get_best_index(), round(nn.get_noise(), 4)))
print(nn.get_output())

nn.run([1 for i in range(135)])
print()
print('ALL 1')
print('answer: {} | noise: {}'.format(nn.get_best_index(), round(nn.get_noise(), 4)))
print(nn.get_output())

nn.run([0.5 for i in range(135)])
print()
print('ALL 0.5')
print('answer: {} | noise: {}'.format(nn.get_best_index(), round(nn.get_noise(), 4)))
print(nn.get_output())
