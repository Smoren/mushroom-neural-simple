import math
import random


class Neuron:
    def __init__(self):
        self.input = 0
        self.output = 0
        self.loss = 0
        self.link_output = []
        self.link_input = []

    def add_link_output(self, link):
        self.link_output.append(link)
        return self

    def add_link_input(self, link):
        self.link_input.append(link)
        return self

    def add(self, signal):
        self.input += signal
        return self

    def reset(self, input=None):
        self.input = 0 if input is None else input
        return self

    def activation(self):
        self.output = self.get_activation(self.input)
        return self

    def send(self):
        for link in self.link_output:
            link.send(self.output)
        return self

    def back_propagation_if_output_layer(self, ref, speed):
        for link in self.link_input:
            link.back_propagation_if_output_layer(ref, speed)
        return self

    def back_propagation_if_hidden_layer(self, speed):
        for link in self.link_input:
            link.back_propagation_if_hidden_layer(speed)
        return self

    def back_propagation_apply(self):
        for link in self.link_output:
            link.apply_weight_delta()
        return self

    @staticmethod
    def get_activation(x):
        return 1 / (math.exp(-x) + 1)

    @classmethod
    def get_activation_derivative(cls, x):
        return cls.get_activation(x) * (1 - cls.get_activation(x))

    @staticmethod
    def get_loss(output, ref):
        return 1 / 2 * (output - ref) ** 2

    @staticmethod
    def get_loss_derivative(x, a):
        return x - a

    def __repr__(self):
        return "[{:.4f} => {:.4f}]".format(self.input, self.output)


class TransparentNeuron(Neuron):
    def activation(self):
        self.output = self.input
        return self


class Layer:
    def __init__(self, class_name, size):
        self.neurons = []
        self.loss = 0

        for i in range(0, size):
            self.neurons.append(class_name())

    def reset(self, input=None):
        self.loss = 0
        if input is None:
            for neuron in self.neurons:
                neuron.reset()
        else:
            for i in range(0, len(self.neurons)):
                self.neurons[i].reset(input[i])
        return self

    def calc(self):
        for neuron in self.neurons:
            neuron.activation()
        return self

    def get_loss(self, output_need=None):
        loss = 0

        for i in range(0, len(self.neurons)):
            neuron = self.neurons[i]
            output_need_item = output_need[i]
            loss += neuron.get_loss(neuron.output, output_need_item)

        return loss

    def send(self):
        for neuron in self.neurons:
            neuron.send()
        return self

    def back_propagation_if_output_layer(self, ref, speed):
        i = 0
        for neuron in self.neurons:
            neuron.back_propagation_if_output_layer(ref[i], speed)
            i += 1
        return self

    def back_propagation_if_hidden_layer(self, speed):
        i = 0
        for neuron in self.neurons:
            neuron.back_propagation_if_hidden_layer(speed)
            i += 1
        return self

    def back_propagation_apply(self):
        for neuron in self.neurons:
            neuron.back_propagation_apply()
        return self

    def __len__(self):
        return len(self.neurons)

    def __repr__(self):
        return ', '.join([repr(x) for x in self.neurons])


class Link:
    def __init__(self, n_from, n_to, weight):
        self.n_from = n_from
        self.n_to = n_to
        self.weight = weight
        self.weight_delta = 0
        self.weight_delta_param = 0

    def send(self, signal):
        self.n_to.add(signal * self.weight)
        return self

    def back_propagation_if_output_layer(self, ref, speed=1):
        de_dy = self.n_to.get_activation(self.n_to.input) - ref
        dy_dz = self.n_to.get_activation_derivative(self.n_to.input)
        dz_dw = self.n_from.output

        self.weight_delta_param = de_dy * dy_dz
        self.weight_delta = -de_dy * dy_dz * dz_dw * speed

        return self

    def back_propagation_if_hidden_layer(self, speed=1):
        self.weight_delta = 0
        self.weight_delta_param = 0

        de_dy = 0

        for link in self.n_to.link_output:
            dz_dy = link.weight
            de_dy += link.weight_delta_param * dz_dy

        dy_dz = self.n_to.get_activation_derivative(self.n_to.input)
        dz_dw = self.n_from.output

        self.weight_delta_param = de_dy * dy_dz
        self.weight_delta = -de_dy * dy_dz * dz_dw * speed

        return self

    def apply_weight_delta(self):
        self.weight += self.weight_delta
        self.weight_delta = 0
        self.weight_delta_param = 0
        return self


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def run(self, input):
        if len(self.layers) == 0:
            raise Exception('empty network')

        if len(input) != len(self.layers[0]):
            raise Exception('bad input')

        self.layers[0].reset(input)

        for i in range(0, len(self.layers) - 1):
            curr_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            next_layer.reset()
            curr_layer.calc()
            curr_layer.send()

        self.layers[-1].calc()

    def add_input_layer(self, size):
        self._add_layer(TransparentNeuron, size)

    def add_hidden_layer(self, size):
        self._add_layer(Neuron, size)

    def add_output_layer(self, size):
        self._add_layer(Neuron, size)

    def get_output_layer(self):
        return self.layers[-1]

    def train(self, data, speed):
        i = 0

        loss_total = 0

        for input, ref in data:
            self.run(input)

            output_layer = self.layers[-1]
            loss_sum = output_layer.get_loss(ref)
            output_layer.back_propagation_if_output_layer(ref, speed)

            # print(repr(self))
            # print(ref)
            print("item #{}: {:.4f}".format(i, loss_sum))

            loss_total += loss_sum

            other_layers = self.layers[:-1][::-1]
            for layer in other_layers:
                layer.back_propagation_if_hidden_layer(speed)

            for layer in self.layers:
                layer.back_propagation_apply()

            i += 1

        return loss_total

    def _add_layer(self, class_name, size):
        layer = Layer(class_name, size)
        self.layers.append(layer)

        if len(self.layers) > 1:
            prev_layer = self.layers[-2]

            for n_from in prev_layer.neurons:
                for n_to in layer.neurons:
                    link = Link(n_from, n_to, random.uniform(-0.5, 0.5))
                    n_from.add_link_output(link)
                    n_to.add_link_input(link)

    def __repr__(self):
        return "\n".join([repr(x) for x in self.layers])
