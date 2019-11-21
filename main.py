import math


class Neuron:
    def __init__(self):
        self.input = 0
        self.output = 0
        self.output_need = 0
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

    def calc_loss(self, output_need):
        self.output_need = output_need
        self.loss = self.get_loss(self.output, self.output_need)
        return self

    def back_propagation(self, step):
        for link in self.link_input:
            link.weight = step * self.get_loss_derivative(link.n_from.output, link.weight,
                                                          self.output, self.output_need)
        return self

    @staticmethod
    def get_activation(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def get_activation_derivative(x):
        return math.exp(-x)/(1 + math.exp(-x))**2

    @staticmethod
    def get_loss_derivative(x, w, y, a):
        return 2*(y - a) * math.exp(-w*x)/(1 + math.exp(-w*x))**2 * x

    @staticmethod
    def get_loss(output, output_need):
        return (output-output_need)**2

    def __repr__(self):
        return "[{:.4f} => {:.4f}]".format(self.input, self.output)


class TransparentNeuron(Neuron):
    def activation(self):
        self.output = self.input
        return self


class Layer:
    def __init__(self, class_name, size):
        self.neurons = []

        for i in range(0, size):
            self.neurons.append(class_name())

    def reset(self, input=None):
        if input is None:
            for neuron in self.neurons:
                neuron.reset()
        else:
            for i in range(0, len(self.neurons)):
                self.neurons[i].reset(input[i])

    def calc(self):
        for neuron in self.neurons:
            neuron.activation()

    def send(self):
        for neuron in self.neurons:
            neuron.send()

    def __len__(self):
        return len(self.neurons)

    def __repr__(self):
        return ', '.join([repr(x) for x in self.neurons])


class Link:
    def __init__(self, n_from, n_to, weight):
        self.n_from = n_from
        self.n_to = n_to
        self.weight = weight

    def send(self, signal):
        self.n_to.add(signal * self.weight)


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def run(self, input):
        if len(self.layers) == 0:
            raise Exception('empty network')

        if len(input) != len(self.layers[0]):
            raise Exception('bad input')

        self.layers[0].reset(input)

        for i in range(0, len(self.layers)-1):
            curr_layer = self.layers[i]
            next_layer = self.layers[i+1]

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
nn.add_hidden_layer(5)
nn.add_hidden_layer(8)
nn.add_hidden_layer(5)
nn.add_output_layer(3)

nn.run([1, 1, 1])
print(repr(nn))
# print(repr(nn.get_output_layer()))
