import random

from structs import NeuralNetwork
import tools

DATA_FILE_NAME = 'output/fingers.json'
MOTION_COUNT = 3
EXAMPLES_COUNT = 9
FRAMES_COUNT = 30
FRAME_SIZE = 5
TRAIN_EPOCH_COUNT = 800

# MOTION_COUNT = 3
# EXAMPLES_COUNT = 5
# FRAMES_COUNT = 20
# FRAME_SIZE = 5
# TRAIN_EPOCH_COUNT = 400


def extend_with_amplitudes(input):
    amps = []
    for i in range(FRAME_SIZE):
        amps.append(tools.get_amplitude(input[i::FRAME_SIZE]))

    return input+amps


def random_input_generator():
    while True:
        yield extend_with_amplitudes([random.random() for x in range(FRAME_SIZE*FRAMES_COUNT)])


def random_resting_state_generator():
    while True:
        buf = [random.random() for x in range(FRAME_SIZE)]
        res = []

        for i in range(FRAMES_COUNT):
            for x in buf:
                res.append(x)

        yield extend_with_amplitudes(res)


def random_output_generator():
    while True:
        yield [0.5 for x in range(MOTION_COUNT)]


nn = NeuralNetwork()  # создаем нейронную сеть

import_data = tools.import_json_file(DATA_FILE_NAME)  # получим сохраненные данные связей, если сеть ранее обучали

if import_data:
    nn.import_(import_data)  # выполним импорт связей в сеть
else:
    dirty_data = []

    for motion in range(MOTION_COUNT):
        for example in range(EXAMPLES_COUNT):
            dirty_data.append(tools.import_json_file('input/fingers/{}/{}.json'.format(motion, example)))

    input_layer_size = FRAMES_COUNT * FRAME_SIZE + FRAME_SIZE  # еще добавляются амплитуды
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

        train_data.append([extend_with_amplitudes(input_data), output_data])

    for i in range(10):
        train_data.append([random_input_generator(), random_output_generator()])
        train_data.append([random_resting_state_generator(), random_output_generator()])

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

    nn.run(extend_with_amplitudes(input))  # выставляем на входы сигналы и выполняем прямой проход

    print('MOTION {}'.format(motion))
    print('answer: {} | noise: {}'.format(nn.get_best_index(), round(nn.get_noise(), 4)))
    print(nn.get_output())
    print()

nn.run(random_input_generator())
print('MOTION RANDOM')
print('answer: {} | noise: {}'.format(nn.get_best_index(), round(nn.get_noise(), 4)))
print(nn.get_output())
print()

nn.run(random_resting_state_generator())
print('MOTION RANDOM RESTING STATE')
print('answer: {} | noise: {}'.format(nn.get_best_index(), round(nn.get_noise(), 4)))
print(nn.get_output())
print()
