import io
import json
import os.path
import time


def make_input_array(img_content):
    """Из данных изображения составляет список значений для входного слоя нейронной сети

    :param img_content: массив содержимого файла изображения
    :type img_content: list
    :return: список значений для входного слоя нейронной сети
    :rtype: list[float]

    """
    result = []
    values = {}

    j = 0
    find_marks = 2

    while find_marks:
        if not (str(img_content[j]).find("#000000") == -1):
            values[str(img_content[j])[3]] = 1
            find_marks -= 1
        if not (str(img_content[j]).find("#FFFFFF") == -1):
            values[str(img_content[j])[3]] = 0
            find_marks -= 1
        j += 1

    for i in range(j, len(img_content)):
        for j in range(3, len(img_content[i])):
            if not (str(img_content[i])[j] == '"'):
                result.append(values[str(img_content[i])[j]])
            else:
                break

    return result


def read_img_file(img_path):
    """Читает файл и получает из него список значений для входного слоя нейронной сети

    :param img_path: путь к файлу изображения
    :type img_path: str
    :return: список значений для входного слоя нейронной сети
    :rtype: list[float]

    """
    file_link = io.open(img_path, 'r+b', 0)
    img_content = file_link.readlines()
    arr = make_input_array(img_content)

    return arr


def get_output(i):
    """Составляет эталонный набор значений, в котором все нули, а i-й элемент — 1

    :param i: индекс выхода, который будет установлен в 1
    :type i: int
    :return: эталонный набор значений
    :rtype: list[float]

    """
    result = []

    for j in range(10):
        result.append(0)

    result[i] = 1

    return result


def import_json_file(filename):
    if not os.path.exists(filename):
        return False

    r = ''
    f = open(filename)

    for line in f:
        r += line

    f.close()

    return json.loads(r)


def export_json_file(filename, data):
    r = json.dumps(data, sort_keys=True, indent=4)
    f = open(filename, 'w')

    f.write(r)
    f.close()


def export_file(filename, data):
    f = open(filename, 'w')
    f.write(data)
    f.close()


class Profiler:
    def __init__(self, name):
        self.name = name
        self.start = time.time()

    def spent(self):
        return time.time()-self.start

    def log(self):
        print("{}: {} s".format(self.name, self.spent()))


def get_amplitude(input):
    min_value = float('inf')
    max_value = float('-inf')

    for x in input:
        max_value = x if x > max_value else max_value
        min_value = x if x < min_value else min_value

    return max_value-min_value
