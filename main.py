import math


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

    def loss_gradient_if_output_layer(self, ref):
        for link in self.link_output:
            link.loss_gradient_if_output_layer(ref)
        return self

    def loss_gradient_if_hidden_layer(self):
        for link in self.link_output:
            link.loss_gradient_if_hidden_layer()
        return self

    @staticmethod
    def get_activation(x):
        return 1 / (math.exp(-x) + 1)

    @classmethod
    def get_activation_derivative(cls, x):
        return cls.get_activation(x) * (1 - cls.get_activation(x))

    @staticmethod
    def get_loss(output, output_need):
        return 1 / 2 * (output - output_need) ** 2

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

    def calc_loss(self, output_need=None):
        if output_need is None:
            for neuron in self.neurons:
                self.loss += neuron.calc_loss()
        else:
            for i in range(0, len(self.neurons)):
                neuron = self.neurons[i]
                output_need_item = output_need[i]
                self.loss += neuron.calc_loss(output_need_item)

        return self.loss

    def send(self):
        for neuron in self.neurons:
            neuron.send()
        return self

    def get_output_need(self):
        result = []
        for neuron in self.neurons:
            result.append(neuron.output_need)
        return result

    def loss_gradient_if_output_layer(self, ref):
        for neuron in self.neurons:
            neuron.loss_gradient_if_output_layer(ref)
        return self

    def loss_gradient_if_hidden_layer(self):
        for neuron in self.neuron:
            neuron.loss_gradient_if_hidden_layer()
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

    def loss_gradient_if_output_layer(self, ref):
        de_dy = self.n_to.get_activation(self.n_to.input) - ref
        dy_dz = self.n_to.get_activation_derivative(self.n_to.input)
        dz_dw = self.n_from.output

        self.weight_delta_param = de_dy * dy_dz
        self.weight_delta = -de_dy * dy_dz * dz_dw

        return self

    def loss_gradient_if_hidden_layer(self):
        self.weight_delta = 0
        self.weight_delta_param = 0

        de_dy = 0

        for link in self.n_to.link_output:
            dz_dy = link.weight
            de_dy += link.weight_delta_param * dz_dy

        dy_dz = self.link_to.get_activation_derivative(self.n_to.input)
        dz_dw = self.n_from.output

        self.weight_delta_param = de_dy * dy_dz
        self.weight_delta = de_dy * dy_dz * dz_dw

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

    def train(self, data):
        pass

    def _add_layer(self, class_name, size):
        layer = Layer(class_name, size)
        self.layers.append(layer)

        if len(self.layers) > 1:
            prev_layer = self.layers[-2]

            for n_from in prev_layer.neurons:
                for n_to in layer.neurons:
                    link = Link(n_from, n_to, 0.7)
                    n_from.add_link_output(link)
                    n_to.add_link_input(link)

    def __repr__(self):
        return "\n".join([repr(x) for x in self.layers])


nn = NeuralNetwork()
nn.add_input_layer(3)
nn.add_hidden_layer(3)
nn.add_output_layer(2)

nn.run([1, 1, 1])
print(repr(nn))

# train_data = [
#     [[1, 0, 0], [1, 0]],
#     [[0, 1, 0], [1, 1]],
#     [[0, 0, 1], [0, 1]],
#     [[1, 0, 1], [1, 1]],
#     [[0, 0, 0], [0, 0]],
# ]
# nn.train(train_data)

# print(repr(nn.get_output_layer()))
