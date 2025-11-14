import numpy as np
from random import randint
import time
import pygame
from heapq import nlargest

from neural_network import NeuralNetwork
from genetic_algorithm import Generation, Agent

pygame.init()
FIELD_SIZE = (20, 20)
SCALE = 17
INFO_PANE_HEIGHT = 15
FRAME_DELAY = 0.005
network_display = pygame.Surface((200*SCALE/10, 200*SCALE/10))
animation_display = pygame.Surface((FIELD_SIZE[0] * SCALE, FIELD_SIZE[1] * SCALE + INFO_PANE_HEIGHT))
screen = pygame.display.set_mode((animation_display.get_width() + network_display.get_width(),
                                  max(animation_display.get_height(), network_display.get_height())))

GOAL = (FIELD_SIZE[0] / 2, FIELD_SIZE[1] / 2)
TESTER_RADIUS = 2.5

template = NeuralNetwork(2)
template.add_layer(5)
template.add_layer(5)
template.add_layer(2)

gen = Generation(template=template, mutation_chance=0.2, mutation_size=0.03, size=100)
num_gens = 1000

network = gen.agents[0].network
slope = 1
network[0][0] = [slope/2, -slope, 0]
network[0][1] = [slope/2, 0, -slope]


def sign(n):
    return -1 if n < 0 else 1

def update_screen():
    screen.blit(animation_display, (0, 0))
    screen.blit(network_display, (animation_display.get_width(), 0))
    pygame.display.update()


class Tester:
    def __init__(self, x:float, y:float, brain:Agent):
        self.brain = brain
        self.x = x
        self.y = y
    
    def move(self, max_x, max_y):
        output = self.brain.predict(np.array([self.x/FIELD_SIZE[0], self.y/FIELD_SIZE[1]]))
        # print(output)
        if abs(output[0]) > 0.02:
            # print("a")
            self.x += sign(output[0])
        if abs(output[1]) > 0.02:
            self.y += sign(output[1])

        self.x = min(max(self.x, 0), max_x - 1)
        self.y = min(max(self.y, 0), max_y - 1)


def populate_testers(generation:Generation):
    return [Tester(randint(0, FIELD_SIZE[0]), randint(0, FIELD_SIZE[1]), agent) for agent in generation.agents]


def display_info(round_num, gen_num):
    font = pygame.font.SysFont('Arial', INFO_PANE_HEIGHT)
    text_surface = font.render(f'gen {gen_num}, round {round_num}', True, (255, 255, 255))
    animation_display.blit(text_surface, (0, 0))

def display_testers(testers, round_num, gen_num):
    animation_display.fill((0, 0, 0))
    display_info(round_num, gen_num)
    pygame.draw.circle(animation_display, (255, 0, 0), (GOAL[0] * SCALE, GOAL[1] * SCALE + INFO_PANE_HEIGHT), TESTER_RADIUS*SCALE/10)
    for tester in testers:
        color = score(tester) * 255 / ((FIELD_SIZE[0]/2)**2 + (FIELD_SIZE[1]/2)**2)**(1/2)
        pygame.draw.circle(animation_display, (0, color, 0), ((tester.x + TESTER_RADIUS/2.0) * SCALE, (tester.y + TESTER_RADIUS/2.0) * SCALE + INFO_PANE_HEIGHT), TESTER_RADIUS*SCALE/10)
    update_screen()
    time.sleep(FRAME_DELAY)

def score(tester):
    # Score is more negative further from the center of the field, reaching 0 in the corners.
    return ((FIELD_SIZE[0]/2)**2 + (FIELD_SIZE[1]/2)**2)**(1/2) - ((tester.x - FIELD_SIZE[0]/2)**2 + (tester.y - FIELD_SIZE[1]/2)**2)**(1/2)

def run_generation(testers, round_num, gen_num, num_time_steps=10):
    for t in range(num_time_steps):
        for tester in testers:
            tester.move(FIELD_SIZE[0], FIELD_SIZE[1])
        display_testers(testers, round_num, gen_num)  
    for tester in testers:
        tester.brain.score += score(tester)


def randomize_testers(testers):
    for tester in testers:
        tester.x = randint(0, FIELD_SIZE[0])
        tester.y = randint(0, FIELD_SIZE[1])



testers = populate_testers(gen)
display_testers(testers, 0, 0)

running = True
NUM_ROUNDS = 20
for gen_num in range(num_gens):
    for round_num in range(NUM_ROUNDS):
        run_generation(testers, round_num, gen_num, num_time_steps=20)
        randomize_testers(testers,)
    for tester in testers:
        tester.brain.score /= NUM_ROUNDS

    agents = [tester.brain for tester in testers]
    best = nlargest(1, agents, key=lambda agent: agent.score)[0]
    best.display(network_display)
    gen = gen.next_generation()
    testers = populate_testers(gen)
    print(f'NEW GEN! gen {gen_num}/{num_gens}')
    time.sleep(0.05)

    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  
           running = False
    if not running:
        break