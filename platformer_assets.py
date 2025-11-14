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
        """Scalar multiplication (Vector * 10)."""
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        """Reverse scalar multiplication (10 * Vector)."""
        return self.__mul__(scalar)

    def magnitude(self) -> float:
        """Returns the length of the vector."""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """Returns a new vector with the same direction and a magnitude of 1."""
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)
    
    def to_tuple(self):
        """Converts the vector to a tuple (x, y) for Pygame functions."""
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
        """Returns the center point of the object as a Vector."""
        return Vector(self.pos.x + self.dim.x / 2, self.pos.y + self.dim.y / 2)
    
    def get_rect(self):
        return pygame.Rect(int(self.pos.x), int(self.pos.y), int(self.dim.x), int(self.dim.y))

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, self.color, self.get_rect())


    

MOVEMENT_THRESHOLD = 0.2
JUMP_THRESHOLD = 0.4
G = 9.8
FPS = 60
MAX_RAY_DISTANCE = 500.0
class Player(GameObj):
    def __init__(self, pos: Vector, dim: Vector, brain: Agent):
        self.pos = pos
        self.dim = dim
        self.vel = Vector(0, 0)
        self.color = (50, 200, 100)
        self.brain = brain
        self.is_on_ground = False

    def cast_ray(self, origin: Vector, direction: Vector, obstacles):
        """
        Casts a single ray from an origin in a direction and finds the
        closest distance to any obstacle.
        """
        # Define the full line segment for the ray (still using floats)
        line_end = origin + direction * MAX_RAY_DISTANCE
        
        # Create integer tuples for the Pygame clipline function
        point1 = (int(origin.x), int(origin.y))
        point2 = (int(line_end.x), int(line_end.y))
        line = (point1, point2)
        
        closest_dist = MAX_RAY_DISTANCE
        closes_obs = None
        
        for obs in obstacles:
            hit = obs.get_rect().clipline(line)
            
            if hit:
                # We have a hit. Get the intersection point.
                # hit[0] is the (x, y) tuple of the intersection point
                # We can use the float-based origin for a more accurate distance
                hit_point = Vector(hit[0][0], hit[0][1])
                
                # Calculate the distance from our float origin to the int hit point
                dist = (hit_point - origin).magnitude()
                
                # We want the *closest* hit, so we track the minimum
                if dist < closest_dist:
                    closest_dist = dist
                    closes_obs = obs
                    
        return closest_dist, closes_obs

    def get_inputs(self, obstacles: list[GameObj]) -> NDArray[float16]:
        """
        Gets the AI's "vision" by casting rays and collecting state.
        Returns a normalized array of all inputs.
        """
        origin = self.center()
        
        # Define 8 directions (N, NE, E, SE, S, SW, W, NW)
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
        
        # --- Create separate lists for each data type ---
        distance_inputs = []
        type_inputs = []
        
        for direction in directions:
            dist, obj = self.cast_ray(origin, direction, obstacles)
            
            # 1. Add distance
            distance_inputs.append(dist)
            
            # 2. Add type
            if obj is not None:
                if obj.type == 'standard':
                    type_inputs.append(0.0) # Normal platform
                else:
                    type_inputs.append(1.0) # e.g., 'danger' platform
            else:
                type_inputs.append(-1.0) # Empty space
                
        # --- Normalize each list correctly ---
        
        # Normalize distances (0.0 to 1.0)
        norm_distances = np.array(distance_inputs, dtype=float16) / MAX_RAY_DISTANCE
        
        # Types are already in a good range (-1, 0, 1)
        norm_types = np.array(type_inputs, dtype=float16)
        
        # Normalize state inputs (-1.0 to 1.0)
        vel_x = np.clip(self.vel.x / 70, -1, 1) # Use your max speed (70)
        vel_y = np.clip(self.vel.y / 80, -1, 1) # Use your max jump (80)
        on_ground = 1.0 if self.is_on_ground else 0.0
        
        state_inputs = np.array([vel_x, vel_y, on_ground], dtype=float16)

        # --- Concatenate all inputs into one final array ---
        return np.concatenate((norm_distances, norm_types, state_inputs))
    
    def jump(self, jump_strength):
        self.vel.y -= min(50 * jump_strength, 80)
    
    def update(self, obstacles: list):
        output = self.brain.predict(self.get_inputs(obstacles))
        horizontal = output[0]
        jump = output[1]

        if self.is_on_ground:
            self.vel.x *= 0.95
            if abs(horizontal) > MOVEMENT_THRESHOLD:
                self.vel.x += horizontal * 20
                # Clamp the velocity between -70 and 70
                self.vel.x = np.clip(self.vel.x, -70, 70)
            if jump > JUMP_THRESHOLD:
                self.jump(jump)
        else:
            self.vel.x *= 0.85
            self.vel.y += G / FPS


        self.pos.x += self.vel.x / FPS
        for obstacle in obstacles:
            if self.get_rect().colliderect(obstacle.get_rect()):
                if self.vel.x > 0:
                    self.pos.x = obstacle.left() - self.dim.x
                    self.vel.x = 0
                elif self.vel.x < 0:
                    self.pos.x = obstacle.right()
                    self.vel.x = 0

        self.pos.y += self.vel.y / FPS
        self.is_on_ground = False
        for obstacle in obstacles:
            if self.get_rect().colliderect(obstacle.get_rect()):
                if self.vel.y > 0:
                    self.pos.y = obstacle.top() - self.dim.y
                    self.vel.y = 0
                    self.is_on_ground = True
                elif self.vel.y < 0:
                    self.pos.y = obstacle.bottom()
                    self.vel.y = 0
    
    


class Platform(GameObj):
    def __init__(self, pos: Vector, dim: Vector, type='standard'):
        self.pos = pos
        self.dim = dim
        self.type = type
        if type == 'standard':
            self.color = (255, 255, 255)
        else:
            self.color = (255, 0, 0)