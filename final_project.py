import numpy as np
from random import randint, random
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
RENDER_EVERY_N_STEPS = 7
network_display = pygame.Surface((400, 400))
animation_display = pygame.Surface((FIELD_SIZE[0] * SCALE, FIELD_SIZE[1] * SCALE + INFO_PANE_HEIGHT), pygame.SRCALPHA)
screen = pygame.display.set_mode((animation_display.get_width() + network_display.get_width(),
                                  max(animation_display.get_height(), network_display.get_height())))

template = NeuralNetwork(55)
template.add_layer(32)
# template.add_layer(4)
template.add_layer(2)

gen = Generation(template=template, mutation_chance=0.3, mutation_size=0.3, size=120)
num_gens = 1000


def populate_players(generation: Generation):
    return [Player(Vector(0, 0), Vector(10, 25), agent) for agent in generation.agents]


def display_info(round_num, gen_num):
    font = pygame.font.SysFont('Arial', INFO_PANE_HEIGHT)
    text_surface = font.render(f'gen {gen_num}, round {round_num}', True, (255, 255, 255))
    screen.fill((0, 0, 0))
    screen.blit(text_surface, (0, 0))

def display_players(players):
    for player in players:
        player.draw(animation_display)

def display_platforms(platforms):
    for platform in platforms:
        platform.draw(animation_display)
    
def update_display(players, platforms, goal: Vector, kill_zone: pygame.Rect, round_num, gen_num):
    global running
    animation_display.fill((0, 0, 0))
    
    display_players(players)
    display_platforms(platforms)
    display_info(round_num, gen_num)
    pygame.draw.circle(animation_display, (0, 255, 0), goal.to_tuple(), 5)

    kill_zone_surf = pygame.Surface((kill_zone.width, kill_zone.height), pygame.SRCALPHA)
    pygame.draw.rect(kill_zone_surf, (100, 100, 100, 100), (0, 0, kill_zone.width, kill_zone.height))
    animation_display.blit(kill_zone_surf, (kill_zone.left, kill_zone.top))

    screen.blit(animation_display, (0, INFO_PANE_HEIGHT))
    screen.blit(network_display, (animation_display.get_width(), 0))
    pygame.display.update()
    for event in pygame.event.get():  
        if event.type == pygame.QUIT:
           pygame.quit()
           running = False


def reset_players(players: list[Player], spawn: Vector):
    for player in players:
        player.pos = Vector(spawn.x + random()*2 - 1, spawn.y + random()*2 - 1)
        player.revive()
        player.vel = Vector(0, 0)
        player.min_dist_to_goal = float('inf')
        player.is_on_ground = False

def run_generation(players: list[Player], level: Level, round_num, gen_num, num_time_steps=10):
    for t in range(num_time_steps):
        if not running:
             break
        
        for player in players:
            if player.is_alive:
                if t > 730 and player.get_rect().colliderect(level.kill_zone):
                    player.die()
                player.update(level.platforms, level.goal)
                
        
        # Only draw one frame every N steps
        if t % RENDER_EVERY_N_STEPS == 0:
            update_display(players, level.platforms, level.goal, level.kill_zone, round_num, gen_num)
            # TODO: Maybe move the rendering logic out of it's own loop and into here so it doesn't have to run a second loop for the Players?
        
    for player in players:
        player.brain.score -= (level.goal - player.pos).magnitude()/2.5 + player.min_dist_to_goal#min(player.pos.x, 130) - player.pos.y * 3



l1_obstacles: list[GameObj] = [Platform(0, FIELD_SIZE[1] - 50, FIELD_SIZE[0], 50),
                            Platform(-10, 0, 11, FIELD_SIZE[1]),
                            Platform(FIELD_SIZE[0] - 1, 0, 11, FIELD_SIZE[1]),
                            Platform(320, 200, 50, 300),
                            Platform(240, 270, 50, 80),
                            Platform(0, 290, 120, 10, 'kill'),
                            Platform(340, 150, 50, 10, 'kill'),
                            Platform(140, FIELD_SIZE[1] - 51, 50, 10, 'kill'),
                            Platform(290, FIELD_SIZE[1] - 51, 320-290, 10, 'kill')]
l1_goal = Vector(500, 300)
l1 = Level(l1_obstacles, l1_goal, Vector(4, 325), kill_zone=pygame.Rect(0, 0, 100, 400))

l2_obstacles: list[GameObj] = [Platform(0, FIELD_SIZE[1] - 50, FIELD_SIZE[0], 50),
                            Platform(-10, 0, 11, FIELD_SIZE[1]),
                            Platform(FIELD_SIZE[0] - 1, 0, 11, FIELD_SIZE[1]),
                            Platform(0, 260, 100, 10, 'kill'),
                            Platform(100, 260, 40, 10),
                            Platform(90, 325, 10, 25, 'kill'),
                            
                            Platform(200, 290, 150, 10),
                            Platform(220, 200, 100, 10),
                            Platform(250, 265, 10, 25, 'kill'),
                            Platform(350, 290, 50, 10, 'kill'),
                            Platform(240, 200, 10, 200, 'kill')]
l2_goal = Vector(230, 190)
l2 = Level(l2_obstacles, l2_goal, Vector(4, 325), alternate_spawn=Vector(500, 325), kill_zone=pygame.Rect(0, 0, 80, 400))

l3_obstacles: list[GameObj] = [Platform(0, FIELD_SIZE[1] - 50, FIELD_SIZE[0], 50),
                               Platform(-10, 0, 11, FIELD_SIZE[1]),
                               Platform(170, 0, 11, FIELD_SIZE[1]),
                               
                               Platform(0, 270, 50, 10),
                               Platform(120, 210, 50, 10),
                               Platform(0, 160, 50, 10),
                               Platform(120, 110, 50, 10),
                               Platform(0, 50, 90, 10)]
l3_goal = Vector(80, 10)
l3 = Level(l3_obstacles, l3_goal, Vector(80, 325))

levels = [l1, l2, l3, l1.get_reversed(FIELD_SIZE[0]), l2.get_reversed(FIELD_SIZE[0])]#, l2.get_alt()]
players = populate_players(gen)

running = True
for gen_num in range(num_gens):
    for round_num, level in enumerate(levels):
        reset_players(players, level.spawn_pos)
        run_generation(players, level, round_num, gen_num, num_time_steps=2000)
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