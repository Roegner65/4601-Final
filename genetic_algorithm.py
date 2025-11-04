from neural_network import NeuralNetwork
from heapq import nlargest
from random import randint
import numpy as np


class Agent(NeuralNetwork):
    def __init__(self, network=None):
        self.score = float('-inf')  # Will be set later, in training
        if network != None:
            self.network = network
    
    def generate_child(self, mutation_chance, mutation_size) -> NeuralNetwork:
        new_layers = []
        for layer in self.network:
            probability_of_mutation = np.random.random(layer.shape)
            mutation_mask = probability_of_mutation < mutation_chance   # Higher mutation_chance = more mutations
            mutation_value = (np.random.random(layer.shape) * 2 - 1)
            new_layers.append(layer + mutation_mask * mutation_value)
        return Agent(new_layers)

class Generation:
    def __init__(self, agents=None,
                 size=100, template=None,   # size is only needed if randomly generating the Generation.
                 num_to_keep=5, mutation_chance=0.2, mutation_size=0.2):
        if agents == None:
            if template == None:
                raise ValueError('A template NeuralNetwork is required to generate random agents. '\
                                 'Try providing a template or a list of pre-made agents.')
            self.agents = self.populate_agents(size, template)
        else:
            self.agents = agents
        self.mutation_chance = mutation_chance
        self.mutation_size = mutation_size
        self.num_to_keep = num_to_keep

    def populate_agents(self, num_agents, template: NeuralNetwork):
        return [template.rand_copy() for i in range(num_agents)]
    

    def next_generation(self):
        best = nlargest(self.num_to_keep, self.agents, key=lambda agent: agent.score)
        next_gen = []
        next_gen.extend(best)
        
        while len(next_gen) < self.get_size():
            parent = best[randint(0, self.num_to_keep)]
            next_gen.append(parent.generate_child(self.mutation_chance, self.mutation_size))

        return Generation(agents=next_gen)


    def get_size(self):
        return len(self.agents)