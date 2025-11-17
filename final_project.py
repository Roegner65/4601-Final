import numpy as np
from random import randint
import time
import pygame
from heapq import nlargest

from neural_network import NeuralNetwork
from genetic_algorithm import Generation, Agent
from platformer_assets import *

pygame.init()
FIELD_SIZE = (600, 400)
SCALE = 1
INFO_PANE_HEIGHT = 15
network_display = pygame.Surface((400, 400))
animation_display = pygame.Surface((FIELD_SIZE[0] * SCALE, FIELD_SIZE[1] * SCALE + INFO_PANE_HEIGHT))
screen = pygame.display.set_mode((animation_display.get_width() + network_display.get_width(),
                                  max(animation_display.get_height(), network_display.get_height())))

GOAL = (FIELD_SIZE[0] / 2, FIELD_SIZE[1] / 2)
TESTER_RADIUS = 2.5

template = NeuralNetwork(19)
template.add_layer(5)
template.add_layer(5)
template.add_layer(2)

gen = Generation(template=template, mutation_chance=0.2, mutation_size=0.1, size=100)
num_gens = 1000

# network = gen.agents[0].network
# slope = 1
# network[0][0] = [slope/2, -slope, 0]
# network[0][1] = [slope/2, 0, -slope]


def sign(n):
    return -1 if n < 0 else 1


def populate_players(generation: Generation):
    return [Player(Vector(3, 325), Vector(10, 25), agent) for agent in generation.agents]


def display_info(round_num, gen_num):
    font = pygame.font.SysFont('Arial', INFO_PANE_HEIGHT)
    text_surface = font.render(f'gen {gen_num}, round {round_num}', True, (255, 255, 255))
    animation_display.blit(text_surface, (0, 0))


def display_players(players):
    for player in players:
        player.draw(animation_display)

def display_platforms(platforms):
    for platform in platforms:
        platform.draw(animation_display)
    
def update_display(players, platforms, round_num, gen_num):
    global running
    animation_display.fill((0, 0, 0))
    
    display_players(players)
    display_platforms(platforms)
    display_info(round_num, gen_num)

    screen.blit(animation_display, (0, 0))
    screen.blit(network_display, (animation_display.get_width(), 0))
    pygame.display.update()
    for event in pygame.event.get():  
        if event.type == pygame.QUIT:
           pygame.quit()
           running = False



def run_generation(players: list[Player], obstacles: list[GameObj], round_num, gen_num, num_time_steps=10):
    RENDER_EVERY_N_STEPS = 50 

    for t in range(num_time_steps):
        if not running:
             break
        
        for player in players:
            if player.is_alive:
                player.update(obstacles)
                if t > 500 and player.pos.x < 100:
                    player.die()
        
        # Only draw one frame every N steps
        if t % RENDER_EVERY_N_STEPS == 0:
            update_display(players, obstacles, round_num, gen_num)
            # TODO: Maybe move the rendering logic out of it's own loop and into here so it doesn't have to run a second loop for the Players?
        
    for player in players:
        player.brain.score = player.pos.x + player.pos.x / 3



obstacles: list[GameObj] = [Platform(Vector(0, FIELD_SIZE[1] - 50), Vector(FIELD_SIZE[0], 50)),
                            Platform(Vector(-10, 0), Vector(11, FIELD_SIZE[1])),
                            Platform(Vector(FIELD_SIZE[0] - 1, 0), Vector(11, FIELD_SIZE[1])),
                            Platform(Vector(320, 200), Vector(50, 300)),
                            Platform(Vector(240, 270), Vector(50, 80)),
                            Platform(Vector(0, 290), Vector(120, 10), 'kill'),
                            Platform(Vector(340, 150), Vector(50, 10), 'kill'),
                            Platform(Vector(140, FIELD_SIZE[1] - 51), Vector(50, 10), 'kill'),
                            Platform(Vector(290, FIELD_SIZE[1] - 51), Vector(320-290, 10), 'kill')]

players = populate_players(gen)

running = True
for gen_num in range(num_gens):
    run_generation(players, obstacles, 0, gen_num, num_time_steps=2000)

    agents = [player.brain for player in players]
    best = nlargest(1, agents, key=lambda agent: agent.score)[0]
    best.display(network_display)
    gen = gen.next_generation()
    players = populate_players(gen)
    print(f'NEW GEN! gen {gen_num}/{num_gens}')
    time.sleep(0.05)

    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  
           running = False
    if not running:
        break