from __future__ import annotations
import sys
import os
import math
import json
import pygame
import numpy as np
from constant import *
import algorithm
from abc import ABC, abstractmethod
from threading import Thread
from collections import deque

from typing import Iterable



class Command_list:
    def __init__(self, max_len = 10):
        self.queue = deque({})
        self.max_len = max_len

    def append(self, command):
        if len(self.queue) == self.max_len:
            self.queue.popleft()
        self.queue.append(command)

    def check(self, cmd: list):
        cmd_len = len(cmd)
        for start in range(len(self.queue) - cmd_len + 1):
            li = [cmd[i] == self.queue[start + i] for i in range(cmd_len)]
            if False not in li:
                return True
        return False


class Interacter(ABC):
    all_unit = UNIT_SET

    def __init__(self, target: list[bool]):
        self.target = target

    @property
    def check(self, unit: Unit):
        if self.target[unit.number]:
            return True, self
        else:
            return False, self

    @abstractmethod
    def active(self, unit: Unit):
        pass

class killer(Interacter):
    def __init__(self, target):
        super().__init__(target)

    def active(self, unit: Unit, interacter: Interacter):
        if interacter is Victim:
            unit.is_alive = False

class Victim(Interacter):
    def __init__(self, target):
        super().__init__(target)

    def active(self, unit: Unit, interacter: Interacter):
        if interacter is killer:
            unit.is_alive = False

class Giver(Interacter):
    def __init__(self, target):
        super().__init__(target)

class Getter(Interacter):
    pass

class Image:
    def __init__(self, images: list, ratio: list, rate: int = 60):
        self.images_origin = images
        self.images = images

        self.__ratio = ratio

        self.length = len(self.images)
        self.__number = 0

        self.change_max_late = rate
        self.change_late = rate

        self.ratio_update()
        self.__current = images[0]

    @property
    def current(self):
        return self.__current

    @current.setter
    def current(self, n):
        self.__number = n % self.length
        self.__current = self.images[self.__number]

    @property
    def ratio(self):
        return self.__ratio
    @ratio.setter
    def size(self, n):
        self.__ratio = n

    def change(self):
        self.change_late = (self.change_late + 1) % self.change_max_late
        if self.change_late == 0:
            self.__number += 1
        self.current = self.__number

    def turn_to(self, angle):
        for i in range(self.length):
            self.images[i] = pygame.transform.rotate(self.images_origin[i], angle)

    def ratio_update(self):
        for i in range(self.length):
            self.images[i] = pygame.transform.scale(self.images_origin[i], (UNIT_SIZE * self.ratio, UNIT_SIZE * self.ratio))

    def get_rect(self, x, y, z=0):
        return self.images[0].get_rect(center= (x, y))

class All(ABC):
    def __init__(self, coordinate: np.array, direction: float, ratio):
        self.coordinate = coordinate
        self.direction = direction
        self.ratio = ratio

    def __getitem__(self, index):
        if index == 0:
            return self.coordinate[0]
        elif index == 1:
            return self.coordinate[1]

    @abstractmethod
    def draw(self, screen: pygame.display):
        pass


# main three class
class Unit(All):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter, algorithm = algorithm.Unit,
                 images: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1
                 ):
        All.__init__(self, coordinate, direction, ratio)
        
        self.speed = speed
        self.image = Image(images, ratio, rate= rate)
        self.interact = interact
        self.algorithm = algorithm
        
        self.current_point: Point = None

        self.name = name
        self.number = number

        self.wait = False
        self._alive = True
        self.life = -1

    @property
    def alive(self):
        return self._alive
    @alive.setter
    def alive(self, alive: bool):
        if not alive and self.life > 0: 
            self.life -= 1
            self._alive = True
            
    @property
    def can_act(self):
        return self.alive and not self.wait
        

    def move(self, fps):
        if not self.can_act:
            return
        self.coordinate += self.speed/fps

    def move_to(self, x, y, z= 0):
        self.coordinate = np.array([x, y, z])

    def turn_to(self, angle, speed):
        if not self.can_act:
            return False
        self.direction = angle
        rad = angle * math.pi / 180
        self.speed = np.array([speed * math.cos(rad), -speed * math.sin(rad), 0])
        return True

    def draw(self, screen: pygame.surface.Surface):
        screen.blit(self.image.current, self.image.get_rect(*self.coordinate))


class Item(All):
    def __init__(self, coordinate: np.array, direction: float, ratio, interacter: Interacter,
                 images: list = []):
        All.__init__(coordinate, direction, ratio)
        self.interacter = interacter
        self.image = Image(images, ratio)


    def draw(self):
        pass

# --------------------------------
class Line:
    def __init__(self, start, end, color=(0, 0, 0), width = 1):
        self.start = start
        self.end = end

        self.width = width
        self.color = color

    def draw(self, screen):
        pygame.draw.line(screen, self.color, self.start, self.end, self.width)

class Point:
    def __init__(self, x, y, z = 0, radius = 1, color = (0, 0, 0)):
        self.x = x
        self.y = y
        self.z = z

        self.radius = radius
        self.color = color
        
        self._coordinate = (self.x, self.y)
        
    @property
    def coordinate(self):
        return self._coordinate
        
    @coordinate.setter
    def coordinate(self, li):
        self.x = li[0]
        self.y = li[1]
        self._coordinate = (self.x, self.y)
        
    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Point only supports index 0 and 1")

    def distance(self, x, y, z=0):
        return math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z))

    def distance_x(self, x):
        return abs(x - self.x)

    def distance_y(self, y):
        return abs(y - self.y)
    
    def distance_z(self, z):
        return abs(z - self.z)


    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

class Grid(All):
    def __init__(self, coordinate: np.array, direction: float, size: tuple[list], ratio = RATIO,
                 gap: float = Grid_data.BLOCK_GAP):
        super().__init__(coordinate, direction, ratio)
        self.gap = gap * ratio
        self.map_x = size[0] * self.gap
        self.map_y = size[1] * self.gap
        self.block_x = size[0]
        self.block_y = size[1]

        self.color = (255, 255, 255)

        dx = self.block_x/2 * self.gap
        dy = self.block_y/2 * self.gap

        self.start_point = (coordinate[0] - dx, coordinate[1] - dy)
        self.end_point = (coordinate[0] + dx, coordinate[1] + dy)

        self.row = []
        self.column = []
        self.points = []
        
        self.cross_points: list[Point] = []
        self.center_points: list[Point] = []
        self.update()

    def update(self):
        self.row = []
        self.column = []
        self.cross_points: list[Point] = []
        self.center_points: list[Point] = []
        for i in range(self.block_x + 1):
            x = self.start_point[0] + i * self.gap
            self.column.append(Line((x, self.start_point[1]), (x, self.end_point[1]), color= self.color))
        for i in range(self.block_y + 1):
            y = self.start_point[1] + i * self.gap
            self.column.append(Line((self.start_point[0], y), (self.end_point[0], y), color= self.color))

        for i in range(self.block_y):
            y = self.start_point[1] + (i + 0.5) * self.gap
            for j in range(self.block_x):
                x = self.start_point[0] + (j + 0.5) * self.gap
                self.center_points.append(Point(x, y, color= (100,255,100)))
                
        for i in range(self.block_y+1):
            y = self.start_point[1] + i * self.gap
            for j in range(self.block_x+1):
                x = self.start_point[0] + j * self.gap
                self.cross_points.append(Point(x, y, color= (100,255,255), radius=4))

    def get_block_coordinate(self, point: Point):
        x, y = point[0] - self.start_point[0], point[1] - self.start_point[1]
        block_x, block_y = int(x//self.gap), int(y//self.gap)
        return (block_x, block_y)
        

    def check(self, unit: Unit):
        x, y = unit.coordinate[0] - self.start_point[0], unit.coordinate[1] - self.start_point[1]
        block_x, block_y = int(x//self.gap), int(y//self.gap)
        return self.cross_points[min(max(0, block_y), self.block_y) * (self.block_x+1) + min(max(0, block_x), self.block_x)]

    def draw(self, screen):
        for line in self.row:
            line.draw(screen)
        for line in self.column:
            line.draw(screen)
        for point in self.center_points:
            point.draw(screen)
        for point in self.cross_points:
            point.draw(screen)
        pygame.draw.circle(screen, (255, 100, 100), self.start_point, 5)
        pygame.draw.circle(screen, (255, 100, 100), self.end_point, 5)

class Wall(All):
    def __init__(self, coordinate, direction, size, ratio, image: Image):
        super().__init__(coordinate, direction, ratio)
        self.d_size = size
        self.size = size * ratio
        
        self.image = Image(image, ratio)
        self.image.turn_to(direction)
    
    def draw(self, screen: pygame.surface.Surface):
        screen.blit(self.image.current, self.image.get_rect(*self.coordinate))



class Trap(All):
    pass


class Map:
    def __init__(self, grid: Grid, map_data: dict, in_color, out_color, block_gap, ratio, grid_show = False):
        self.grid = grid
        self.map_data = map_data["map"]["map"]
        
        self.size = (map_data["map"]["size_x"], map_data["map"]["size_y"])
        
        self.walls: list[Wall] = self.load_wall(map_data, in_color, out_color, block_gap, ratio)
        self.item: list[All] = self.load_item(map_data)
        
        self.grid_show = grid_show
        
    def is_wall(self, point: Point, direction):
        block_x, block_y = self.grid.get_block_coordinate(point)
        block_x += int(math.cos(direction * math.pi / 180) * 1.5)
        block_y -= int(math.sin(direction * math.pi / 180) * 1.5)
        
        if not (0 <= block_x < self.size[0] and 0 <= block_y < self.size[1]):
            return True
        try:
            value = self.map_data[block_y][block_x]
        except IndexError:
            value = 0

            
        if value != 0:
            return True
        else:
            return False
    
    def check(self, unit: Unit):
        return self.grid.check(unit)
    
    def update(self):
        self.grid.update()
    
    def draw(self, screen):
        for wall in self.walls:
            wall.draw(screen)
            
        if self.grid_show:
            self.grid.draw(screen)
        
            
    def load_wall(self, data, in_color, out_color, block_gap, ratio):
        walls = []
        
        
        return walls

    @staticmethod
    def load_item(data):
        object_data = data["objects"]
        object_list = []
        
        return object_list
        



class PacMan(Unit):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter, algorithm: algorithm.Unit,
                 images: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1):
        super().__init__(coordinate, direction, speed, interact, algorithm, images[1:], ratio, rate, name=name, number=number)
        self.wait_image = Image(images[0:1], ratio)

    def turn_to(self, angle, speed):
        direction = self.direction
        if super().turn_to(angle, speed):
            self.image.turn_to(angle - direction)


class Ghost(Unit):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter,
                 algorithm: algorithm.Ghost, images_eye: list = [],
                 images_body: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1):

        super().__init__(coordinate, direction, speed, interact, algorithm, images_body, ratio, rate, name=name, number=number)
        self.algorithm = algorithm
        self.images_eye = Image(images_eye, ratio, rate=0)

    def turn_to(self, angle, speed):
        if super().turn_to(angle, speed):
            self.images_eye.current = int(angle//(math.pi/2))


    def draw(self, screen):
        super().draw(screen)
        screen.blit(self.images_eye.current, self.image.get_rect(*self.coordinate))

class Unit_generator:
    pass

class Wall_generator:
    pass

class Item_generator:
    pass




class Main:
    def __init__(self, width, height, fps, show_fps = False, show_grid = False):
        self.screen_width, self.screen_height = width, height
        
        self.level = 0
        
        with open(f"data/map/map{self.level}.json", 'r') as f:
            self.data = json.load(f)

        self.thread = Thread(target=self.back_loop)
        self.thread.start()

        self.clock = pygame.time.Clock()
        self.fps = fps
        self.show_fps = show_fps

        self.main_state = 0
        self.running = True

        self.judgment_distance = JUDGMENT_DISTANCE

        self.font = pygame.font.SysFont("Arial", 24)
        
        self.show_grid = show_grid
        self.grid = Grid((self.screen_width/2, self.screen_height/2), 0, (self.data["map"]["size_x"]-1, self.data["map"]["size_y"]-1))
        self.current_point = None
        
        self.map = Map(self.grid, self.data, (255, 0, 0), (0, 255, 0), Grid_data.BLOCK_GAP, RATIO, self.show_grid)
        
        
        

        # --------------------
        # Pac Man init
        # --------------------
        self.pacman = PacMan(PacMan_data.COORDINATE, PacMan_data.DIRECTION, PacMan_data.SPEED,
                             Victim([0, 1, 1, 1, 1 ,1]), algorithm.Unit, PacMan_data.IMAGES,
                             PacMan_data.RATIO,
                             PacMan_data.RATE, name=PacMan_data.NAME, number= PacMan_data.NUMBER)
        self.pacman.move_to(*self.map.grid.cross_points[10].coordinate)
        self.pacman.current_point = self.map.grid.cross_points[10]
        self.pacman.wait = False

        # --------------------
        # Ghost init
        # --------------------
        self.blinky = Ghost((self.screen_width/3, self.screen_height/3, 0), 0, 5,
                            killer([1, 0, 0, 0, 0, 0]), algorithm.Blinky, Blinky_DATA.EYE_IMGAES,
                            Blinky_DATA.BODY_IMAGES, Blinky_DATA.RATIO, PacMan_data.RATE*2, name="Bilnky", number=2)
        self.blinky.move_to(*self.map.grid.cross_points[-1].coordinate)
        self.blinky.current_point = self.map.grid.cross_points[-1]
        
        self.units: list[Unit] = [self.pacman, self.blinky]
        
        self.wall = Wall(self.map.grid.center_points[0].coordinate, 0, 8, RATIO, Wall_data.IMAGES[0:1])
        

        # --------------------
        # command init
        # --------------------

        self.last_move_command = None
        self.command_list = Command_list(max_len= 10)



    def reset(self):
        
        pass

    def back_loop(self):
        pass

    def loop(self, screen: pygame.surface.Surface):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    self.command_list.append(event.key)

                    if event.key == pygame.K_ESCAPE:
                        self.running = False

                    if event.key in [pygame.K_RIGHT, pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN]:
                        self.last_move_command = event.key
            
        

            self.update()
            self.turn_unit()
            self.draw(screen)
            
            self.wall.draw(screen)

            self.clock.tick(self.fps)
            if self.show_fps:
                fps_text = self.font.render(f"FPS: {self.clock.get_fps():0.2f}", True, (255, 255, 255))
                screen.blit(fps_text, (10, 10))
            pygame.display.flip()
            
    def turn_unit(self):
        
        if self.pacman.current_point.distance_x(self.pacman[0]) <= self.judgment_distance:
            if self.last_move_command == pygame.K_DOWN:
                self.pacman.turn_to(270, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman.current_point[0], self.pacman[1])
            elif self.last_move_command == pygame.K_UP:
                self.pacman.turn_to(90, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman.current_point[0], self.pacman[1])

        if self.pacman.current_point.distance_y(self.pacman[1]) <= self.judgment_distance:
            if self.last_move_command == pygame.K_RIGHT:
                self.pacman.turn_to(0, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman[0], self.pacman.current_point[1])
            elif self.last_move_command == pygame.K_LEFT:
                self.pacman.turn_to(180, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman[0], self.pacman.current_point[1])

    def update(self):
        self.map.update()
        self.pacman.current_point = self.map.check(self.pacman)
        
        for unit in self.units:
            if self.map.is_wall(unit.current_point, unit.direction) and unit.current_point.distance(unit[0], unit[1]) <= self.judgment_distance:
                unit.move_to(unit.current_point[0], unit.current_point[1])
            else:
                unit.move(self.fps)
                unit.image.change()

        self.blinky.image.change()
        
    
    def draw(self, screen: pygame.surface.Surface):
        screen.fill(Screen_data.COLOR)

        # map base check
        self.map.draw(screen)
        
        if self.show_grid:
            self.pacman.current_point.radius = 10
            self.pacman.current_point.draw(screen)

        self.pacman.draw(screen)
        pygame.draw.circle(screen, (100, 100, 255), (self.pacman[0], self.pacman[1]), 8)

        self.blinky.draw(screen)

if __name__=="__main__":
    pygame.init()
    screen = pygame.display.set_mode((Screen_data.WIDTH, Screen_data.HEIGHT))
    MainScreen = Main(Screen_data.WIDTH, Screen_data.HEIGHT, 60, show_fps=True, show_grid=True)
    MainScreen.loop(screen)
