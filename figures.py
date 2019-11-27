from glob import glob
from pprint import pprint

from structs import NeuralNetwork
from tools import read_img_file, get_output


# начинаем составлять обучающую выборку
to_learn = []

# для каждой директории с обучающими выборками (соответствующими распознаваемой цифре)
for digit in range(2):
    # получаем список путей к файлам изображений в ней
    files = glob("figures/{}/*.xpm".format(str(digit)))

    # для каждого пути к файлу
    for img in files:
        # читаем все пиксели файла в массив
        arr = read_img_file(img)

        # добавляем набор входов (пикселей) и эталон в обучающую выборку
        to_learn.append([arr, get_output(digit)])

nn = NeuralNetwork()  # создаем нейронную сеть
nn.add_input_layer(256)  # добавляем входной слой
nn.add_hidden_layer(64)  # добавляем скрытый слой
nn.add_hidden_layer(64)  # добавляем скрытый слой
nn.add_hidden_layer(64)  # добавляем скрытый слой
nn.add_hidden_layer(16)  # добавляем скрытый слой
nn.add_output_layer(2)  # добавляем выходной слой

# в цикле совершаем заданное количество эпох обучения
for i in range(0, 100):
    print('')
    print('EPOCH #{}'.format(i))
    loss_total = nn.train(to_learn, 1)
    print('{:.4f}'.format(loss_total))

# затем получаем пути к файлам из тестовой выборки
files = glob("figures/test/*.xpm")

# для каждого пути к файлу
for img in files:
    # читаем все пиксели файла в массив
    arr = read_img_file(img)

    # подаем данные на сеть и считаем результат
    nn.run(arr)

    # выводим результат
    print()
    print(img)
    pprint(repr(nn.get_output_layer()))
