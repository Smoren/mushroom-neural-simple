from structs import NeuralNetwork

nn = NeuralNetwork()
nn.add_input_layer(3)
nn.add_hidden_layer(4)
nn.add_hidden_layer(5)
nn.add_output_layer(2)

nn.run([1, 1, 1])
print(repr(nn))

train_data = [
    [[1, 0, 0], [1, 0]],
    [[0, 1, 0], [1, 1]],
    [[0, 0, 1], [0, 1]],
    [[1, 0, 1], [1, 1]],
    [[0, 0, 0], [0, 0]],
]

for i in range(0, 10000):
    print('')
    print('EPOCH #{}'.format(i))
    loss_total = nn.train(train_data, 0.1)
    print('{:.4f}'.format(loss_total))

nn.run([1, 0, 0])
print(repr(nn))

# print(repr(nn.get_output_layer()))
