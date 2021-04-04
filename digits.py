from pprint import pprint
from glob import glob
from structs import NeuralNetwork
import tools

DATA_FILE_NAME = 'output/digits.json'
INPUT_LAYER_SIZE = 135
OUTPUT_LAYER_SIZE = 10

nn = NeuralNetwork()  # создаем нейронную сеть

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    dirty_data = []

    nn.add_input_layer(INPUT_LAYER_SIZE)  # добавляем входной слой
    nn.add_hidden_layer(16)  # добавляем скрытый слой
    nn.add_output_layer(OUTPUT_LAYER_SIZE)  # добавляем выходной слой

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

    to_learn.append([[0 for i in range(135)], [0.5 for i in range(10)]])
    to_learn.append([[1 for i in range(135)], [0.5 for i in range(10)]])
    to_learn.append([[0.5 for i in range(135)], [0.5 for i in range(10)]])

    epoch_losses = []  # сюда накопим историю изменения ошибки с каждой эпохой

    # в цикле совершаем заданное количество эпох обучения
    for i in range(0, 300):
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
    print(nn.get_output())

nn.run([0 for i in range(135)])
print()
print('ALL 0')
print(nn.get_output())

nn.run([1 for i in range(135)])
print()
print('ALL 1')
print(nn.get_output())

nn.run([0.5 for i in range(135)])
print()
print('ALL 0.5')
print(nn.get_output())
