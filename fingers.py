from structs import NeuralNetwork
import tools

DATA_FILE_NAME = 'output/fingers.json'
MOTION_COUNT = 3
EXAMPLES_COUNT = 9
FRAMES_COUNT = 30
FRAME_SIZE = 5
TRAIN_EPOCH_COUNT = 400

# MOTION_COUNT = 3
# EXAMPLES_COUNT = 5
# FRAMES_COUNT = 20
# FRAME_SIZE = 5
# TRAIN_EPOCH_COUNT = 400

nn = NeuralNetwork()  # создаем нейронную сеть

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    dirty_data = []

    for motion in range(MOTION_COUNT):
        for example in range(EXAMPLES_COUNT):
            dirty_data.append(tools.import_json_file('input/fingers/{}/{}.json'.format(motion, example)))

    input_layer_size = FRAMES_COUNT * FRAME_SIZE
    output_layer_size = MOTION_COUNT

    nn.add_input_layer(input_layer_size)  # добавляем входной слой
    nn.add_hidden_layer(16)  # добавляем скрытый слой
    nn.add_hidden_layer(16)  # добавляем скрытый слой
    nn.add_output_layer(output_layer_size)  # добавляем выходной слой

    # готовим обучающую выборку
    train_data = []

    for example in dirty_data:
        input_data = []
        output_data = [0 for i in range(MOTION_COUNT)]
        output_data[example['output']] = 1
        for frame in example['input']:
            for finger_value in frame:
                input_data.append(finger_value)

        train_data.append([input_data, output_data])

    epoch_losses = []

    # выполняем заданное количество эпох обучения
    for i in range(0, TRAIN_EPOCH_COUNT):
        print('')
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(train_data, 0.1)
        epoch_losses.append(loss_total)
        print('{:.4f}'.format(loss_total))

    tools.export_json_file(DATA_FILE_NAME, nn.export())

    epoch_losses_formatted = []
    for i in range(len(epoch_losses)):
        loss = epoch_losses[i]
        epoch_losses_formatted.append('({};{}) '.format(i, loss))

    tools.export_file('output/fingers_losses.txt', ''.join(epoch_losses_formatted))

for motion in range(MOTION_COUNT):
    data = tools.import_json_file('input/fingers/test/{}.json'.format(motion))
    input = []

    for frame in data['input']:
        for finger_value in frame:
            input.append(finger_value)

    nn.run(input)  # выставляем на входы сигналы и выполняем прямой проход

    print('MOTION {}'.format(motion))
    print(nn.get_output())
    print()
