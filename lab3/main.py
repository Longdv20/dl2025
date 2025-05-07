import random

def sigmoid(x):
    return 1 / (1 + exp(-x))

def exp(x, terms=10):
    result = 1.0
    numerator = 1.0
    denominator = 1.0
    for i in range(1, terms):
        numerator *= x
        denominator *= i
        result += numerator / denominator
    return result

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = [[random.random() for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [[random.random()] for _ in range(output_size)]
    
    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, biases):
        self.biases = biases

    def dot_product(self, matrix, vector):
        result = []
        for row in matrix:
            value = 0
            for i in range(len(row)):
                value += row[i] * vector[i][0]
            result.append([value])
        return result

    def add_vectors(self, a, b):
        return [[a[i][0] + b[i][0]] for i in range(len(a))]

    def apply_sigmoid(self, vector):
        return [[sigmoid(v[0])] for v in vector]

    def forward(self, input_data):
        z = self.dot_product(self.weights, input_data)
        z = self.add_vectors(z, self.biases)
        return self.apply_sigmoid(z)

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def build_from_config(self, config_file_path):
        with open(config_file_path, 'r') as f:
            lines = f.readlines()
        num_layers = int(lines[0].strip())
        layer_sizes = [int(line.strip()) for line in lines[1:num_layers+1]]
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i]))

    def initialize_from_file(self, weight_bias_file):
        with open(weight_bias_file, 'r') as f:
            lines = f.readlines()
        idx = 0
        for layer in self.layers:
            weights = []
            for _ in range(len(layer.weights)):
                weights.append([float(x) for x in lines[idx].strip().split()])
                idx += 1
            layer.set_weights(weights)

            biases = []
            for _ in range(len(layer.biases)):
                biases.append([float(x) for x in lines[idx].strip().split()])
                idx += 1
            layer.set_biases(biases)

    def feedforward(self, input_data):
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

nn = NeuralNetwork()
nn.build_from_config("network_config.txt")
input_vector = [[0.5], [0.8]]
output = nn.feedforward(input_vector)
print("Network output:\n", output)
