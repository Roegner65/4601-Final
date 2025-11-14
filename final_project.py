import numpy as np
from random import randint
import time
import pygame
from heapq import nlargest

from neural_network import NeuralNetwork
from genetic_algorithm import Generation, Agent
from platformer_assets import Player, Vector

pygame.init()
FIELD_SIZE = (200, 200)
SCALE = 1
INFO_PANE_HEIGHT = 15
FRAME_DELAY = 0.005
network_display = pygame.Surface((200*SCALE/10, 200*SCALE/10))
animation_display = pygame.Surface((FIELD_SIZE[0] * SCALE, FIELD_SIZE[1] * SCALE + INFO_PANE_HEIGHT))
screen = pygame.display.set_mode((animation_display.get_width() + network_display.get_width(),
                                  max(animation_display.get_height(), network_display.get_height())))

GOAL = (FIELD_SIZE[0] / 2, FIELD_SIZE[1] / 2)
TESTER_RADIUS = 2.5

template = NeuralNetwork(6)
template.add_layer(5)
template.add_layer(5)
template.add_layer(2)

gen = Generation(template=template, mutation_chance=0.2, mutation_size=0.03, size=100)
num_gens = 1000

# network = gen.agents[0].network
# slope = 1
# network[0][0] = [slope/2, -slope, 0]
# network[0][1] = [slope/2, 0, -slope]


def sign(n):
    return -1 if n < 0 else 1

def update_screen():
    screen.blit(animation_display, (0, 0))
    screen.blit(network_display, (animation_display.get_width(), 0))
    pygame.display.update()


def populate_players(generation: Generation):
    return [Player(Vector(0, 0), Vector(10, 25), agent) for agent in generation.agents]


def display_info(round_num, gen_num):
    font = pygame.font.SysFont('Arial', INFO_PANE_HEIGHT)
    text_surface = font.render(f'gen {gen_num}, round {round_num}', True, (255, 255, 255))
    animation_display.blit(text_surface, (0, 0))


def display_players(players, round_num, gen_num):
    animation_display.fill((0, 0, 0))
    display_info(round_num, gen_num)
    for player in players:
        player.draw(animation_display)
    update_screen()
    time.sleep(FRAME_DELAY)


def run_generation(players: list[Player], round_num, gen_num, num_time_steps=10):
    for t in range(num_time_steps):
        for player in players:
            player.update([])
        display_players(players, round_num, gen_num)  
    for player in players:
        player.brain.score = player.pos.x



players = populate_players(gen)

running = True
NUM_ROUNDS = 20
for gen_num in range(num_gens):
    run_generation(players, 0, gen_num, num_time_steps=20)
    for player in players:
        player.brain.score /= NUM_ROUNDS

    agents = [player.brain for player in players]
    best = nlargest(1, agents, key=lambda agent: agent.score)[0]
    best.display(network_display)
    gen = gen.next_generation()
    testers = populate_players(gen)
    print(f'NEW GEN! gen {gen_num}/{num_gens}')
    time.sleep(0.05)

    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  
           running = False
    if not running:
        break