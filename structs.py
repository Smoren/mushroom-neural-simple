# coding=utf-8
import math
import random


class Neuron:
    """Класс нейрона
    Нейроны данного класса будем использовать в выходном и скрытых слоях

    Attributes
    ----------
    input (float): накопитель входного сигнала
    output (float): значение выхода нейрона
    link_input (list[Link]): массив входящих связей
    link_output (list[Link]): массив исходящих связей

    """

    def __init__(self):
        self.input = 0
        self.output = 0
        self.link_input = []
        self.link_output = []

    def add_link_input(self, link):
        """Добавляет связь в список входящих связей

        :param link: объект связи
        :type link: Link
        :rtype: Neuron

        """
        self.link_input.append(link)
        return self

    def add_link_output(self, link):
        """Добавляет связь в список исходящих связей

        :param link: объект связи
        :type link: Link
        :rtype: Neuron

        """
        self.link_output.append(link)
        return self

    def add(self, signal):
        """Прибавляет значение сигнала к накопителю входного сигнала

        :param signal: значение поступающего сигнала
        :type signal: float
        :rtype: Neuron

        """
        self.input += signal
        return self

    def reset(self, input=None):
        """Сбрасывает значение накопителя входного сигнала в 0, либо устанавливает его значение в input

        :param input: значение, которое будет записано в накопитель входного сигнала
        :type input: float|None
        :rtype: Neuron
        """
        self.input = 0 if input is None else input
        return self

    def activation(self):
        """Вычисляет значение функции активации от значения накопителя входного сигнала и записывает его в output

        :rtype: Neuron

        """
        self.output = self.get_activation(self.input)
        return self

    def send(self):
        """Отправляет значение выходного сигнала на все исходящие связи

        :rtype: Neuron

        """
        for link in self.link_output:
            link.send(self.output)
        return self

    def back_propagation_if_output_layer(self, ref, speed):
        """Запускает расчет поправок весов по методу обратного распространения ошибки для каждой их входящих связей.
        Используется только для нейронов выходного слоя.

        :param ref: значение эталона из обучающей выборки
        :param speed: коэффициент скорости обучения
        :type ref: float
        :type speed: float
        :rtype: Neuron

        """
        for link in self.link_input:
            link.back_propagation_if_output_layer(ref, speed)
        return self

    def back_propagation_if_hidden_layer(self, speed):
        """Запускает расчет поправок весов по методу обратного распространения ошибки для каждой их входящих связей.
        Используется только для нейронов скрытых слоев.

        :param speed: коэффициент скорости обучения
        :type speed: float
        :rtype: Neuron

        """
        for link in self.link_input:
            link.back_propagation_if_hidden_layer(speed)
        return self

    def back_propagation_apply(self):
        """Применяет ранее вычисленные поправки к весам для каждой из входящих связей

        :rtype Neuron

        """
        for link in self.link_output:
            link.apply_weight_delta()
        return self

    def export(self):
        """Экспорт связей нейрона

        :rtype list
        """
        result = []
        for link in self.link_output:
            result.append(link.export())

        return result

    def import_(self, links_data):
        """Импорт связей нейрона

        :rtype Neuron
        """
        for i in range(len(links_data)):
            link_data = links_data[i]
            link = self.link_output[i]

            link.import_(link_data)

        return self

    @staticmethod
    def get_activation(x):
        """Функция активации — экспоненциальная сигмоида

        :param x: аргумент функции
        :type x: float
        :rtype float

        """
        return 1 / (math.exp(-x) + 1)

    @classmethod
    def get_activation_derivative(cls, x):
        """Производная функции активации

        :param x: аргумент функции
        :type x: float
        :rtype float

        """
        return cls.get_activation(x) * (1 - cls.get_activation(x))

    @staticmethod
    def get_loss(output, ref):
        """Функция вычисления потери потери

        :param output: значение на выходе нейрона
        :param ref: эталонное значение
        :type output: float
        :rtype float

        """
        return 1 / 2 * (output - ref) ** 2

    def __repr__(self):
        """[сигнал на входе => сигнал на выходе]"""
        return "[{:.4f} => {:.4f}]".format(self.input, self.output)


class TransparentNeuron(Neuron):
    """Класс нейрона c "прозрачной" функцией активации (вход = выход)
    Нейроны данного класса будем использовать во входном слое

    Attributes
    ----------
    input (float): накопитель входного сигнала
    output (float): значение выхода нейрона
    link_output (list): массив исходящих связей

    """
    def activation(self):
        """Прозрачная функция активации
        Записывает значение на входе в выход нейрона
        
        :rtype TransparentNeuron

        """
        self.output = self.input
        return self


class Layer:
    """Класс слоя нейронной сети
    Представляет из себя обертку над массивом входящих в слой нейронов
    
    Attributes
    ----------
    neurons (list[Neuron]): список нейронов

    :param neuron_class: класс, нейроны которого будут использоваться в слое
    :param size: количество нейронов в слое
    :type neuron_class: type
    :type size: int
    
    """
    def __init__(self, neuron_class, size):
        self.neurons = []

        for i in range(0, size):
            self.neurons.append(neuron_class())

    def reset(self, input=None):
        """Для всех нейронов с слое сбрасывает значение накопителя входного сигнала в 0, 
        либо устанавливает его значение в input
        
        :param input: набор значений, которые будут записаны в накопители входного сигнала для каждого нейрона
        :type input: list[float]|None
        :rtype: Layer

        """
        if input is None:
            for neuron in self.neurons:
                neuron.reset()
        else:
            for i in range(0, len(self.neurons)):
                self.neurons[i].reset(input[i])
        return self

    def calc(self):
        """Обсчет слоя при прямом проходе
        Для каждого нейрона в слое запускает расчет выходного сигнала от входного с использованием функции активации
        
        :rtype Layer

        """
        for neuron in self.neurons:
            neuron.activation()
        return self

    def get_loss(self, refs):
        """Вычисляет потерю для слоя на основе набора эталонов для каждого нейрона в слое, если он является выходным
        
        :param refs: набор эталонных значений для каждого нейрона в слое
        :type refs: list[float]
        :return: значение потери
        :rtype: float

        """
        loss = 0

        for i in range(0, len(self.neurons)):
            neuron = self.neurons[i]
            ref = refs[i]
            loss += neuron.get_loss(neuron.output, ref)

        return loss

    def send(self):
        """Отправляет все выходные сигналы нейронов слоя по исходящим связям на входы следующего слоя
        
        :rtype: Layer

        """
        for neuron in self.neurons:
            neuron.send()
        return self

    def back_propagation_if_output_layer(self, refs, speed):
        """Запускает расчет поправок весов всех нейронов слоя по методу обратного распространения ошибки 
        для каждой их входящих связей. Используется только для нейронов выходного слоя.

        :param refs: набор значений эталонов для каждого нейрона слоя из обучающей выборки
        :param speed: коэффициент скорости обучения
        :type refs: list[float]
        :type speed: float
        :rtype: Layer

        """
        i = 0
        for neuron in self.neurons:
            neuron.back_propagation_if_output_layer(refs[i], speed)
            i += 1
        return self

    def back_propagation_if_hidden_layer(self, speed):
        """Запускает расчет поправок весов всех нейронов слоя по методу обратного распространения ошибки 
        для каждой их входящих связей. Используется только для нейронов скрытых слоев.

        :param speed: коэффициент скорости обучения
        :type speed: float
        :rtype: Layer

        """
        i = 0
        for neuron in self.neurons:
            neuron.back_propagation_if_hidden_layer(speed)
            i += 1
        return self

    def back_propagation_apply(self):
        """Применяет поправки весов всех входящих связей всех нейронов слоя, вычисленные с помощью 
        метода обратного распространения ошибки
        
        :rtype: Layer

        """
        for neuron in self.neurons:
            neuron.back_propagation_apply()
        return self

    def export(self):
        """Экспорт данных нейронов слоя

        :rtype: list
        """
        result = []
        for neuron in self.neurons:
            result.append(neuron.export())

        return result

    def import_(self, neurons_data):
        """Импорт данных нейронов слоя

        :rtype Neuron
        """
        for i in range(len(neurons_data)):
            neuron_data = neurons_data[i]
            neuron = self.neurons[i]

            neuron.import_(neuron_data)

        return self

    def get_output(self):
        """Возвращает выходные сигналы слоя

        :rtype: dict
        """
        r = []
        i = 0
        for neuron in self.neurons:
            r.append({i: neuron.output})
            i += 1

        return r

    def __len__(self):
        return len(self.neurons)

    def __repr__(self):
        return ', '.join([repr(x) for x in self.neurons])


class Link:
    """Класс связи между нейронами

    Attributes
    ----------
    n_from (Neuron): нейрон, из которого выходит связь
    n_to (Neuron): нейрон, в который входит связь
    weight (float): вес связи
    weight_delta (float): поправка к связи нейрона, вычисляемая с помощью метода обратного распространения ошибки
    weight_delta_param (float): параметр, равный dl/dy * dy/dz, который требуется хранить для расчета поправок весов
        на предыдущем слое

    :param n_from: нейрон, из которого выходит связь
    :param n_to: нейрон, в который входит связь
    :param weight: вес связи
    :type n_from: Neuron
    :type n_to: Neuron
    :type weight: float

    """
    def __init__(self, n_from, n_to, weight):
        self.n_from = n_from
        self.n_to = n_to
        self.weight = weight
        self.weight_delta = 0
        self.weight_delta_param = 0

    def send(self, signal):
        """Отправляет сигнал выхода нейрона, из которого выходит связь, на накопитель входного сигнала нейрона,
        в который входит связь

        :rtype: Link

        """
        self.n_to.add(signal * self.weight)
        return self

    def back_propagation_if_output_layer(self, ref, speed=1):
        """Выполняет расчет поправки веса по методу обратного распространения ошибки.
        Используется только для связей с нейронами выходного слоя.
        Также считает и запоминает промежуточный параметр  weight_delta, который будет использован при расчете
        поправки весов на предыдущем слое

        :param ref: значение эталона из обучающей выборки
        :param speed: коэффициент скорости обучения (по умолчанию 1)
        :type ref: float
        :type speed: float
        :rtype: Link

        """

        # вычисляем влияние выхода нейрона на потерю
        dl_dy = self.n_to.get_activation(self.n_to.input) - ref

        # вычисляем влияние взвешенного входа нейрона на выход
        dy_dz = self.n_to.get_activation_derivative(self.n_to.input)

        # вычисляем влияние веса связи на взвешенный вход нейрона
        dz_dw = self.n_from.output  # а оно равно значению сигнала, который пуступил на эту связь

        # вычисляем и запоминаем параметр, который понадобится для вычислений на предыдущем слое
        self.weight_delta_param = dl_dy * dy_dz

        # вычисляем и запоминаем поправку к весу (антиградиент, умноженный на коэффициент скорости)
        self.weight_delta = -dl_dy * dy_dz * dz_dw * speed

        return self

    def back_propagation_if_hidden_layer(self, speed=1):
        """Запускает расчет поправок весов по методу обратного распространения ошибки для каждой их входящих связей.
        Используется только для связей с нейронами скрытых слоев.

        :param speed: коэффициент скорости обучения (по умолчанию 1)
        :type speed: float
        :rtype: Link

        """
        self.weight_delta = 0
        self.weight_delta_param = 0

        # сюда будем накапливать влияние выхода нейрона скрытого слоя на общую ошибку сети
        dl_dy = 0

        # для каждой исходящей связи нейрона скрытого слоя с нейронами следующего слоя
        for link in self.n_to.link_output:
            # получим влияние выхода нейрона скрытого слоя на вход нейрона следующего слоя
            dz_dy = link.weight  # а оно равно весу связи

            # накопим влияние выхода нейрона скрытого слоя на общую ошибку сети,
            # используя параметр, который мы запомнили ранее при обсчете следующего слоя
            dl_dy += link.weight_delta_param * dz_dy

        # получим влияние входа нейрона скрытого слоя на выход (производная функции активации)
        dy_dz = self.n_to.get_activation_derivative(self.n_to.input)

        # получим влияние веса связи на взвешенный вход нейрона скрытого слоя
        dz_dw = self.n_from.output  # а оно равно значению на выходе с нейрона предыдущего слоя

        # посчитаем и запомним параметр, который нам понадобится при расчете обратного распространения ошибки
        # на предыдущем слое
        self.weight_delta_param = dl_dy * dy_dz

        # посчитаем поправку к весу связи
        self.weight_delta = -dl_dy * dy_dz * dz_dw * speed

        return self

    def apply_weight_delta(self):
        """Применяет вычисленную поправку к весу
        Также сбрасываеи значения поправки и промежуточного параметра weight_delta_param в 0

        :rtype: Link

        """
        self.weight += self.weight_delta
        self.weight_delta = 0
        self.weight_delta_param = 0
        return self

    def export(self):
        """Экспорт значения связи

        :rtype: float
        """
        return self.weight

    def import_(self, value):
        """Импорт значения связи

        :rtype: Link
        """
        self.weight = value

        return self


class NeuralNetwork:
    """Класс нейронной сети
    Представляет из себя обертку над списком слоев нейронов

    Attributes
    ----------
    layers (list[Layer]): список слоев нейронов

    """
    def __init__(self):
        self.layers = []

    def add_input_layer(self, size):
        """Добавляет в сеть входной слой размером в size нейронов

        :param size: количество нейронов в слое
        :type size: int
        :rtype: NeuralNetwork

        """
        self._add_layer(TransparentNeuron, size)
        return self

    def add_hidden_layer(self, size):
        """Добавляет в сеть скрытый слой размером в size нейронов

        :param size: количество нейронов в слое
        :type size: int
        :rtype: NeuralNetwork

        """
        self._add_layer(Neuron, size)
        return self

    def add_output_layer(self, size):
        """Добавляет в сеть выходной слой размером в size нейронов

        :param size: количество нейронов в слое
        :type size: int
        :rtype: NeuralNetwork

        """
        self._add_layer(Neuron, size)
        return self

    def get_output_layer(self):
        """Возвращает выходной (последний) слой сети

        :rtype: Layer

        """
        return self.layers[-1]

    def run(self, input):
        """Выполняет прямой проход сигналов через нейронную сеть

        :param input: набор значений на входы всех нейронов входного слоя сети
        :type input: list[float]
        :rtype: NeuralNetwork

        """

        # проверяем, что сеть имеет слои
        if len(self.layers) == 0:
            raise Exception('empty network')

        # проверяем, что набор входных данных имеет тот же размер, что и входной слой сети
        if len(input) != len(self.layers[0]):
            raise Exception('bad input')

        # устанавливаем сигналы на вход сети
        self.layers[0].reset(input)

        # для каждой пары соседних слоев сети
        for i in range(0, len(self.layers) - 1):
            # текущий слой
            curr_layer = self.layers[i]

            # следующий слой
            next_layer = self.layers[i + 1]

            # сбрасываем значения на входе следующего слоя
            next_layer.reset()

            # выполняем расчет выходных значений на текущем слое
            curr_layer.calc()

            # отправляем посчитанные выходные значения на входы следующего слоя
            curr_layer.send()

        # выполняем расчет выходных значений на выходном (последнем) слое
        self.layers[-1].calc()

        return self

    def train(self, data, speed):
        """Выполняет обучение сети методом обратного распространения ошибки на основе данных обучающей выборки

        :param data: данные обучающей выборки; имеет формат [[[входы...], [эталоны...]], ...]
        :param speed: коэффициент скорости обучения
        :type data: list
        :type speed: float
        :return: суммарная потеря по обучающей выборке
        :rtype: float

        """
        i = 0

        loss_total = 0  # суммарные потери по всей обучающей выборке

        # для каждого примера из обучающей выборки
        for input, refs in data:
            # выполняем прямой проход сигналов по сети при заданном наборе входных сигналов
            self.run(input)

            # берем выходной слой сети
            output_layer = self.layers[-1]

            # считаем для него потери на основе синалов на выходах сети и соответствуюзих им эталонных данных
            loss_sum = output_layer.get_loss(refs)

            # накапливаем суммарную потерю
            loss_total += loss_sum

            # выводим данные о потере на текущем примере обучающей выборки
            print("item #{} loss: {:.4f}".format(i, loss_sum))

            # считаем поправки весов для входящих связей выходного слоя по методу обратного распространения ошибки
            output_layer.back_propagation_if_output_layer(refs, speed)

            # берем список остальных слоев в обратном порядке (входной слой — последний)
            other_layers = self.layers[:-1][::-1]

            # для каждого слоя из этого списка
            for layer in other_layers:
                # считаем поправки весов для входящих связей слоя по методу обратного распространения ошибки
                layer.back_propagation_if_hidden_layer(speed)

            # когда все поправки весов уже посчитаны, для каждого слоя сети
            for layer in self.layers:
                # применяем поправки к весам
                layer.back_propagation_apply()

            i += 1

        # возвращаем суммарную потерю по обучающей выборке
        return loss_total

    def _add_layer(self, neuron_class, size):
        """Добавление слоя нейронов заданного класса в нейронную сеть

        :param neuron_class: класс нейронов, которые будут использоваться в слое
        :param size: количество нейронов в слое
        :type neuron_class: type
        :type size: int
        :rtype: NeuralNetwork

        """

        # создаем слой
        layer = Layer(neuron_class, size)

        # добавляем в список слоев нейронной сети
        self.layers.append(layer)

        # если добавляемый слой не является первым (читай: входным),
        # нам потребуется создать связи предыдущего слоя с новым
        if len(self.layers) > 1:
            # возьмем предыдущий слой
            prev_layer = self.layers[-2]

            # для всех нейронов предыдущего слоя
            for n_from in prev_layer.neurons:
                # для всех нейронов нового слоя
                for n_to in layer.neurons:
                    # создаем связь с рандомным весом в отрезке [-0.5, 0.5]
                    link = Link(n_from, n_to, random.uniform(-0.5, 0.5))

                    # регистрируем связь, как выходную, для нейрона из предыдущего слоя
                    n_from.add_link_output(link)

                    # регистрируем связь, как входную, для нейрона из нового слоя
                    n_to.add_link_input(link)

        return self

    def export(self):
        """Экспорт данных слоев сети

        :rtype: list
        """
        result = []
        for layer in self.layers:
            result.append(layer.export())

        return result

    def import_(self, layers_data):
        """Импорт данных слоев сети

        :rtype Neuron
        """

        self.add_input_layer(len(layers_data[0]))

        for i in range(1, len(layers_data)-1):
            self.add_hidden_layer(len(layers_data[i]))

        self.add_output_layer(len(layers_data[-1]))

        for i in range(len(layers_data)):
            layer_data = layers_data[i]
            layer = self.layers[i]

            layer.import_(layer_data)

        return self

    def get_output(self):
        """Возвращает выходные сигналы выходного слоя сети

        :rtype: dict
        """
        r = []
        i = 0
        for neuron in self.layers[-1].neurons:
            r.append({i: neuron.output})
            i += 1

        return r

    def __repr__(self):
        return "\n".join([repr(x) for x in self.layers])
