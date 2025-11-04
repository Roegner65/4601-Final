import numpy as np
from numpy.typing import NDArray
from numpy import float16

class NeuralNetwork:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.network = []
    
    def add_layer(self, size):
        # TODO: Should generate new random layer of given size, creating random weights for the connections between the previous
        # layer and the new one, plus the biases
        # Generation should look like this {say first layer}: (size, prev_size{or num ins} + 1)
        # So self.network will be: (num_layers, layer_size, prev_size + 1)
        new_layer = 0
        self.network.append(new_layer)

    def num_outputs(self, ) -> int:
        # TODO: get the number of outputs of the final layer
        return 0
    
    def predict(self, inputs: NDArray[float16]) -> NDArray[float16]:
        # TODO
        return np.zeros(self.num_outputs(), dtype=float16)