from glob import glob
import tools
import sys

from structs import NeuralNetwork
from tools import read_img_file, get_output

# увеличим лимит рекурсии, чтобы избежать возможных segfault
sys.setrecursionlimit(500000)

# имя файла для хранения данных обученной нейронной сети
DATA_FILE_NAME = 'output/numbers.json'

# начинаем составлять обучающую выборку
to_learn = []

# для каждой директории с обучающими выборками (соответствующими распознаваемой цифре)
for digit in range(10):
    # получаем список путей к файлам изображений в ней
    files = glob("input/numbers/{}/*.xpm".format(str(digit)))

    # для каждого пути к файлу
    for img in files:
        # читаем все пиксели файла в массив
        arr = read_img_file(img)

        # добавляем набор входов (пикселей) и эталон в обучающую выборку
        to_learn.append([arr, get_output(digit)])

nn = NeuralNetwork()  # создаем нейронную сеть

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выстроим заранее обученную сеть из данных импорта
else:
    nn.add_input_layer(256)  # добавляем входной слой
    nn.add_hidden_layer(64)  # добавляем скрытый слой
    nn.add_hidden_layer(64)  # добавляем скрытый слой
    nn.add_output_layer(10)  # добавляем выходной слой

    epoch_losses = []  # сюда накопим историю изменения ошибки с каждой эпохой

    # в цикле совершаем заданное количество эпох обучения
    for i in range(0, 30):
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(to_learn, 1)
        epoch_losses.append(loss_total)
        print('TOTAL LOSS: {:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

    epoch_losses_formatted = []
    for i in range(len(epoch_losses)):
        loss = epoch_losses[i]
        epoch_losses_formatted.append('({};{}) '.format(i, loss))

    tools.export_file('output/numbers_losses.txt', ''.join(epoch_losses_formatted))

# затем получаем пути к файлам из тестовой выборки
files = glob("input/numbers/test/*.xpm")

# для каждого пути к файлу
for img in files:
    # читаем все пиксели файла в массив
    arr = read_img_file(img)

    # подаем данные на сеть и считаем результат
    nn.run(arr)

    # выводим результат
    print()
    print(img)
    print('answer: {} | noise: {}'.format(nn.get_best_index(), round(nn.get_noise(), 4)))
    print(nn.get_output())

