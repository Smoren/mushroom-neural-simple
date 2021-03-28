from structs import NeuralNetwork
import tools

DATA_FILE_NAME = 'output/simple.json'

nn = NeuralNetwork()  # создаем нейронную сеть

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    nn.add_input_layer(3)  # добавляем входной слой
    nn.add_hidden_layer(4)  # добавляем скрытый слой
    nn.add_hidden_layer(5)  # добавляем скрытый слой
    nn.add_output_layer(2)  # добавляем выходной слой

    nn.run([1, 1, 1])  # выставляем на входы сигналы и выполняем прямой проход
    print(repr(nn))

    # готовим обучающую выборку
    train_data = [
        [[1, 0, 0], [1, 0]],
        [[0, 1, 0], [1, 1]],
        [[0, 0, 1], [0, 1]],
        [[1, 0, 1], [1, 1]],
        [[0, 0, 0], [0, 0]],
    ]

    # выполняем заданное количество эпох обучения
    for i in range(0, 10000):
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(train_data, 0.1)
        print('{:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

nn.run([1, 0, 0])  # выставляем на входы сигналы и выполняем прямой проход

print()
print(nn.get_output())
