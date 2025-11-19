import pygame
from genetic_algorithm import Generation, Agent
import numpy as np
from numpy.typing import NDArray
from numpy import float16
import math


class Vector:
    def __init__(self, x:float, y:float):
        self.x: float = x
        self.y: float = y

    def __add__(self, obj):
        return Vector(self.x + obj.x, self.y + obj.y)
    
    def __sub__(self, obj):
        return Vector(self.x - obj.x, self.y - obj.y)

    def dot(self, obj):
        return self.x * obj.x + self.y * obj.y
    
    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)
    
    def to_tuple(self):
        return (self.x, self.y)
    
    def __repr__(self):
        """A helper for printing and debugging."""
        return f"Vector({self.x}, {self.y})"
    

class GameObj:
    def __init__(self, pos: Vector, dim: Vector, color):
        self.pos = pos
        self.dim = dim
        self.color = color

    def right(self) -> float:
        return self.pos.x + self.dim.x
    
    def left(self) -> float:
        return self.pos.x
    
    def top(self) -> float:
        return self.pos.y
    
    def bottom(self) -> float:
        return self.pos.y + self.dim.y
    
    def center(self) -> Vector:
        return Vector(self.pos.x + self.dim.x / 2, self.pos.y + self.dim.y / 2)
    
    def get_rect(self):
        return pygame.Rect(int(self.pos.x), int(self.pos.y), int(self.dim.x), int(self.dim.y))

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, self.color, self.get_rect())


    

MOVEMENT_THRESHOLD = 0.2
JUMP_THRESHOLD = 0.1
G = 9.8
FPS = 60
MAX_RAY_DISTANCE = 500.0
MAX_JUMP = 40
JUMP_STRENGTH = 40
MAX_MOVE_SPEED = 40
class Player(GameObj):
    def __init__(self, pos: Vector, dim: Vector, brain: Agent):
        self.pos = pos
        self.dim = dim
        self.vel = Vector(0, 0)
        self.color = (50, 200, 100)
        self.brain = brain
        self.is_on_ground = False
        self.is_alive = True

    def cast_ray(self, origin: Vector, direction: Vector, obstacles):
        line_end = origin + direction * MAX_RAY_DISTANCE
        
        point1 = (int(origin.x), int(origin.y))
        point2 = (int(line_end.x), int(line_end.y))
        line = (point1, point2)
        
        closest_dist = MAX_RAY_DISTANCE
        closes_obs = None
        
        for obs in obstacles:
            hit = obs.get_rect().clipline(line)
            
            if hit:
                hit_point = Vector(hit[0][0], hit[0][1])
                
                dist = (hit_point - origin).magnitude()
                
                if dist < closest_dist:
                    closest_dist = dist
                    closes_obs = obs
                    
        return closest_dist, closes_obs

    def get_inputs(self, obstacles: list[GameObj]) -> NDArray[float16]:
        origin = self.center()
        
        directions = [
            Vector(0, -1),   # North
            Vector(1, -1).normalize(),   # North-East
            Vector(1, 0),    # East
            Vector(1, 1).normalize(),    # South-East
            Vector(0, 1),    # South
            Vector(-1, 1).normalize(),   # South-West
            Vector(-1, 0),   # West
            Vector(-1, -1).normalize()   # North-West
        ]
        
        distance_inputs = []
        type_inputs = []
        
        for direction in directions:
            dist, obj = self.cast_ray(origin, direction, obstacles)
            
            distance_inputs.append(dist)
            
            if obj is not None:
                if obj.type == 'standard':
                    type_inputs.append(0.0) # Normal platform
                else:
                    type_inputs.append(1.0) # danger platform
            else:
                type_inputs.append(-1.0) # Empty space
                
        
        norm_distances = np.array(distance_inputs, dtype=float16) / MAX_RAY_DISTANCE
        
        norm_types = np.array(type_inputs, dtype=float16)
        
        vel_x = np.clip(self.vel.x / 70, -1, 1) # max speed (70)
        vel_y = np.clip(self.vel.y / MAX_JUMP, -1, 1)
        on_ground = 1.0 if self.is_on_ground else 0.0
        
        state_inputs = np.array([vel_x, vel_y, on_ground], dtype=float16)

        return np.concatenate((norm_distances, norm_types, state_inputs))
    
    def jump(self, jump_strength):
        self.vel.y -= min(JUMP_STRENGTH * jump_strength, MAX_JUMP)
    
    def die(self):
        self.is_alive = False
        self.brain.score -= 100
        self.color = (20, 100, 60)

    def update(self, obstacles: list):
        output = self.brain.predict(self.get_inputs(obstacles))
        horizontal = output[0]
        jump = output[1]

        self.is_on_ground = False
        for obstacle in obstacles:
            if self.get_rect().colliderect(obstacle.get_rect()):
                if obstacle.type == 'kill':
                    self.die()
                    return
                if self.vel.y > 0:
                    self.pos.y = obstacle.top() - self.dim.y
                    self.vel.y = 0
                    self.is_on_ground = True
                elif self.vel.y < 0:
                    self.pos.y = obstacle.bottom()
                    self.vel.y = 0

        if self.is_on_ground:
            self.vel.x *= 0.88
            if abs(horizontal) > MOVEMENT_THRESHOLD:
                self.vel.x += horizontal * 5
                self.vel.x = np.clip(self.vel.x, -MAX_MOVE_SPEED, MAX_MOVE_SPEED)
            if jump > JUMP_THRESHOLD:
                self.jump(jump)
        else:
            if abs(horizontal) > MOVEMENT_THRESHOLD:
                self.vel.x += horizontal/3
                self.vel.x = np.clip(self.vel.x, -MAX_MOVE_SPEED, MAX_MOVE_SPEED)
            self.vel.x *= 0.98
            self.vel.y += G / FPS


        self.pos.x += self.vel.x / FPS
        for obstacle in obstacles:
            if self.get_rect().colliderect(obstacle.get_rect()):
                if obstacle.type == 'kill':
                    self.die()
                    return
                if self.vel.x > 0:
                    self.pos.x = obstacle.left() - self.dim.x
                    self.vel.x = 0
                elif self.vel.x < 0:
                    self.pos.x = obstacle.right()
                    self.vel.x = 0

        self.pos.y += self.vel.y / FPS
        
    
    


class Platform(GameObj):
    def __init__(self, pos: Vector, dim: Vector, type='standard'):
        self.pos = pos
        self.dim = dim
        self.type = type
        if type == 'standard':
            self.color = (255, 255, 255)
        else:
            self.color = (255, 0, 0)