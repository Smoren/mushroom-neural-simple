import glob
import io
import os
from pprint import pprint

from structs import NeuralNetwork


def makeInputArray(imgContent):
    result = []
    values = {}

    j = 0
    findMarks = 2

    while findMarks:
        if not (str(imgContent[j]).find("#000000") == -1):
            values[str(imgContent[j])[3]] = 1
            findMarks -= 1
        if not (str(imgContent[j]).find("#FFFFFF") == -1):
            values[str(imgContent[j])[3]] = 0
            findMarks -= 1
        j += 1

    for i in range(j, len(imgContent)):
        for j in range(3, len(imgContent[i])):
            if not (str(imgContent[i])[j] == '"'):
                result.append(values[str(imgContent[i])[j]])
            else:
                break

    return result


def readImgFile(img):
    imgContent = io.BytesIO()
    fileLink = io.open(img, 'r+b', 0)
    imgContent = fileLink.readlines()
    arr = makeInputArray(imgContent)

    return arr


def getOutput(i):
    result = []

    for j in range(2):
        result.append(0)

    result[i] = 1

    return result


to_learn = []

for digit in range(2):
    files = glob.glob("figures/{}/*.xpm".format(str(digit)))

    for img in files:
        arr = readImgFile(img)
        to_learn.append([arr, getOutput(digit)])

# print(to_learn)

nn = NeuralNetwork()
nn.add_input_layer(256)
nn.add_hidden_layer(256)
nn.add_hidden_layer(128)
nn.add_hidden_layer(64)
nn.add_hidden_layer(32)
nn.add_hidden_layer(16)
nn.add_output_layer(2)

for i in range(0, 30):
    print('')
    print('EPOCH #{}'.format(i))
    loss_total = nn.train(to_learn, 1)
    print('{:.4f}'.format(loss_total))

files = glob.glob("figures/test/*.xpm")
for img in files:
    arr = readImgFile(img)
    nn.run(arr)
    print()
    print(img)
    pprint(repr(nn.get_output_layer()))
