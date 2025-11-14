import pygame
from genetic_algorithm import Generation, Agent
import numpy as np
from numpy.typing import NDArray
from numpy import float16


class Vector:
    def __init__(self, x, y):
        self.x: float = x
        self.y: float = y

    def __add__(self, obj):
        return Vector(self.x + obj.x, self.y + obj.y)
    
    def __sub__(self, obj):
        return Vector(self.x - obj.x, self.y - obj.y)

    def __mul__(self, obj):
        return self.x * obj.x + self.y * obj.y
    
    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other)
    

class GameObj:
    def __init__(self, pos: Vector, dim: Vector, color):
        self.pos = pos
        self.dim = dim
        self.color = color

    def right(self) -> float:
        return self.pos.x
    
    def left(self) -> float:
        return self.pos.x + self.dim.x
    
    def top(self) -> float:
        return self.pos.y
    
    def bottom(self) -> float:
        return self.pos.y + self.dim.y
    
    def get_rect(self):
        return pygame.Rect(int(self.pos.x), int(self.pos.y), int(self.dim.x), int(self.dim.y))

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, self.color, self.get_rect())
    

MOVEMENT_THRESHOLD = 0.2
JUMP_THRESHOLD = 0.4
G = 9.8
FPS = 60
class Player(GameObj):
    def __init__(self, pos: Vector, dim: Vector, brain: Agent):
        self.pos = pos
        self.dim = dim
        self.vel = Vector(0, 0)
        self.color = (50, 200, 100)
        self.brain = brain

    def get_inputs(self, obstacles: list[GameObj]) -> NDArray[float16]:
        return np.zeros(6).astype(float16)
    
    def on_ground(self, obstacles: list[GameObj]) -> bool:
        for platform in obstacles:
            if (platform.top() - self.bottom()) < 0.001:
                return True
        return False
    
    def jump(self, obstacles: list[GameObj]):
        if self.on_ground(obstacles):
            self.vel.y -= 3
    
    def update(self, obstacles: list[GameObj]):
        output = self.brain.predict(self.get_inputs(obstacles))
        horizontal = output[0]
        jump = output[1]

        if abs(horizontal) > MOVEMENT_THRESHOLD:
            self.pos.x += horizontal
        if jump > JUMP_THRESHOLD:
            self.jump(obstacles)

        self.vel.y += G / FPS

        self.pos += self.vel / FPS
    
    


class Platform(GameObj):
    def __init__(self, pos: Vector, dim: Vector, type='standard'):
        self.pos = pos
        self.dim = dim
        self.type = type
        if type == 'standard':
            self.color = (255, 255, 255)
        else:
            self.color = (255, 0, 0)
    
    def get_norm_force(self, player: Player) -> Vector:
        left_dist = abs(self.left() - player.right())
        right_dist = abs(self.right() - player.left())
        top_dist = abs(self.top() - player.bottom())
        bottom_dist = abs(self.bottom() - player.top())
        if left_dist < right_dist and left_dist < top_dist and left_dist < bottom_dist:
            return Vector(-player.vel.x, 0)
        if right_dist < left_dist and right_dist < top_dist and right_dist < bottom_dist:
            return Vector(-player.vel.x, 0)
        if bottom_dist < left_dist and bottom_dist < right_dist and bottom_dist < top_dist:
            return Vector(0, -player.vel.y)
        # if top_dist < left_dist and top_dist < right_dist and top_dist < bottom_dist:
        return Vector(0, -player.vel.y)