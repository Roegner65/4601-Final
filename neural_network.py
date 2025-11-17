import numpy as np
from numpy.typing import NDArray
from numpy import float16
from math import e

import pygame
from pygame.gfxdraw import aacircle

def sigmoid(n):
    return 2/(1 + e**(-n)) - 1

class NeuralNetwork:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.network = []
        self.prevSize = num_inputs
    
    def add_layer(self, size):
        new_layer = np.random.rand(size, self.prevSize + 1).astype(float16) * 2 - 1
        self.network.append(new_layer)
        self.prevSize = size

    def num_outputs(self) -> int:
        return len(self.network[-1])
    
    def predict(self, inputs: NDArray[float16]) -> NDArray[float16]:
        current = np.array([1] + list(inputs))
        for layer in self.network[:len(self.network) - 1]:
            next = [1]
            for node_weights in layer:
                next.append(max((current * node_weights).sum(), 0))
                # next.append(sigmoid((current * node_weights).sum()))
            current = np.array(next)
        output = []
        for node_weights in self.network[-1]:
            output.append(sigmoid((current * node_weights).sum()))

        return np.array(output).astype(float16)

    def rand_copy(self):
        copy = []
        for layer in self.network:
            layerCopy = np.random.random(layer.shape).astype(float16) * 2 - 1
            copy.append(layerCopy)
        return copy
    

    def display(self, screen: pygame.Surface):
        min_weight = 0
        max_weight = 0
        for layer in self.network:
            for node in layer:
                for weight in node:
                    if weight < min_weight:
                        min_weight = weight
                    if weight > max_weight:
                        max_weight = weight

        scale = min(screen.get_width(), screen.get_height()) / 200


        layer_dist = (screen.get_width() - 20) / len(self.network)
        previous_positions = []
        for i in range(self.num_inputs):
            if self.num_inputs % 2 == 0:
                node_dists = screen.get_height() / (self.num_inputs + 1)
                previous_positions.append((10, node_dists * i + node_dists*1.125))
            else:
                node_dists = screen.get_height() / self.num_inputs
                previous_positions.append((10, node_dists * i + node_dists/1.5))

            aacircle(screen, int(previous_positions[-1][0]), int(previous_positions[-1][1]), int(2.5 * scale), (255, 255, 255))
            pygame.draw.circle(screen, (255, 255, 255), previous_positions[-1], 3 * scale - 1)


        for i, layer in enumerate(self.network):
            x = layer_dist * (i + 1) + 10
            node_positions = []
            if len(layer) % 2 == 0:
                node_dists = screen.get_height() / (len(layer) + 1)
            else:
                node_dists = screen.get_height() / len(layer)

            for j, node in enumerate(layer):
                if len(layer) % 2 == 0:
                    cur_pos = (x, node_dists * j + node_dists*1.125)
                else:
                    cur_pos = (x, node_dists * j + node_dists/1.5)

                for n in range(1, len(node)):
                    if node[n] > 0:
                        color = (0, 255 * node[n] / abs(max_weight), 0)
                    else:
                        color = (255 * -node[n] / abs(min_weight), 0, 0) if min_weight != 0 else (255 * -node[n], 0, 0)
                    pygame.draw.line(screen, (255, 255, 255), previous_positions[n - 1], cur_pos, int(scale*0.7))
                    pygame.draw.aaline(screen, color, previous_positions[n - 1], cur_pos)
                
                if node[0] > 0:
                    color = (0, 255 * node[0] / abs(max_weight), 0)
                else:
                    color = (255 * -node[0] / abs(min_weight), 0, 0)
                aacircle(screen, int(cur_pos[0]), int(cur_pos[1]), int(2.5 * scale), color)
                pygame.draw.circle(screen, color, cur_pos, 3 * scale - 1)
                node_positions.append(cur_pos)
            previous_positions = node_positions
                
