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

    def __init__(self, target: list[bool], ):
        self.target = target

    @property
    def check(self, unit: Unit):
        if self.target[unit.number]:
            return True
        else:
            return False

    @abstractmethod
    def active(self, unit: Unit):
        pass

class killer(Interacter):
    pass

class Victim(Interacter):
    pass

class Giver(Interacter):
    pass


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
            self.images[i] = pygame.transform.scale(self.images_origin[i], (PacMan_data.SIZE_X * self.ratio, PacMan_data.SIZE_Y * self.ratio))

    def get_rect(self, x, y, z):
        return self.images[0].get_rect(center= (x, y))

class All(ABC):
    def __init__(self, coordinate: np.array, direction: float):
        self.coordinate = coordinate
        self.direction = direction

    def __getitem__(self, index):
        if index == 0:
            return self.coordinate[0]
        elif index == 1:
            return self.coordinate[1]

    @abstractmethod
    def draw(self, screen: pygame.display):
        pass

# --------------------------------

# main three class
class Unit(All):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter, algorithm = algorithm.Unit,
                 images: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1):
        super().__init__(coordinate, direction)
        self.speed = speed
        self.image = Image(images, ratio, rate= rate)
        self.interact = interact
        self.algorithm = algorithm

        self.name = name
        self.number = number

        self.alive = True
        self.life = -1

    @property
    def is_alive(self):
        return self.alive
    @is_alive.setter
    def is_alive(self, interact: killer):
        if interact.check(self):
            self.life -= 1
            self.alive = False

    def revive(self, coordinate: np.array):
        if not self.alive and self.life > 0:
            self.algorithm.move_to(coordinate)

    def move(self, fps):
        self.coordinate += self.speed/fps

    def move_to(self, x, y, z= 0):
        self.coordinate = np.array([x, y, z])

    def turn_to(self, angle, speed):
        self.image.turn_to(angle - self.direction)
        self.direction = angle
        rad = angle * math.pi / 180
        self.speed = np.array([speed * math.cos(rad), -speed * math.sin(rad), 0])

    def draw(self, screen: pygame.surface.Surface):
        screen.blit(self.image.current, self.image.get_rect(*self.coordinate))


class Structure(All):
    def __init__(self, coordinate: np.array, direction: float):
        super().__init__(coordinate, direction)


    def draw(self):
        pass


class Item(All):
    def __init__(self, coordinate: np.array, direction: float, interacter: Interacter,
                 images: list = []):
        super().__init__(coordinate, direction)
        self.interacter = interacter
        self.image = Image(images)


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
    def __init__(self, x, y, radius = 1, color = (0, 0, 0)):
        self.x = x
        self.y = y

        self.radius = radius
        self.color = color

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Point only supports index 0 and 1")

    def distance(self, x, y, z):
        return math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

    def distance_x(self, x):
        return abs(x - self.x)

    def distance_y(self, y):
        return abs(y - self.y)


    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)


class Grid(Structure):
    def __init__(self, coordinate: np.array, direction: float, size: tuple[list],
                 gap: float = UNIT_SIZE * UNIT_RATIO):
        super().__init__(coordinate, direction)
        self.gap = gap
        self.map_x = size[0] * gap
        self.map_y = size[1] * gap
        self.block_x = size[0]
        self.block_y = size[1]

        self.color = (255, 255, 255)

        dx = self.block_x/2 * gap
        dy = self.block_y/2 * gap

        self.start_point = (coordinate[0] - dx, coordinate[1] - dy)
        self.end_point = (coordinate[0] + dx, coordinate[1] + dy)

        self.row = []
        self.column = []
        self.points = []
        self.update()

    def update(self):
        self.row = []
        self.column = []
        self.points = []
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
                self.points.append(Point(x, y, color= (100,255,100)))

    def check(self, unit: Unit):
        x, y = unit.coordinate[0] - self.start_point[0], unit.coordinate[1] - self.start_point[1]
        block_x, block_y = int(x//self.gap), int(y//self.gap)
        return self.points[min(max(0, block_y), self.block_y-1) * self.block_x + min(max(0, block_x), self.block_x-1)]


    def draw(self, screen):
        for line in self.row:
            line.draw(screen)
        for line in self.column:
            line.draw(screen)
        for point in self.points:
            point.draw(screen)
        pygame.draw.circle(screen, (255, 100, 100), self.start_point, 5)
        pygame.draw.circle(screen, (255, 100, 100), self.end_point, 5)

class wall(Structure):
    def __init__(self, coordinate, direction):
        super().__init__(coordinate, direction)


class Trap(Structure):
    pass



class PacMan(Unit):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter,
                 images: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1):
        super().__init__(coordinate, direction, speed, interact, images, ratio, rate, name=name, number=number)



class Ghost(Unit):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter,
                 algorithm: algorithm.Ghost,
                 images: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1,):
        super().__init__(coordinate, direction, speed, interact, images, ratio, rate, name=name, number=number)
        self.algorithm = algorithm

# red
class Blinky(Ghost):
    pass

# orange
class Clyde(Ghost):
    pass

# cyan
class Lnky(Ghost):
    pass

# pink
class Pinky(Ghost):
    pass




class Main:
    def __init__(self, width, height, fps, show_fps = False):
        self.screen_width, self.screen_height = width, height

        self.thread = Thread(target=self.back_loop)
        self.thread.start()

        self.clock = pygame.time.Clock()
        self.fps = fps
        self.show_fps = show_fps

        self.main_state = 0
        self.running = True

        self.judgment_distance = JUDGMENT_DISTANCE

        self.font = pygame.font.SysFont("Arial", 24)

        self.grid = Grid((self.screen_width/2, self.screen_height/2), 0, (12, 10), UNIT_RATIO * UNIT_SIZE*3/2)

        self.pacman = PacMan(PacMan_data.COORDINATE, PacMan_data.DIRECTION, PacMan_data.SPEED,
                             Victim, PacMan_data.IMAGES, PacMan_data.RATIO,
                             PacMan_data.RATE, name=PacMan_data.NAME, number= PacMan_data.NUMBER)

        self.pacman.move_to(self.grid.points[0][0], self.grid.points[0][1])
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



            screen.fill(Screen_data.COLOR)


            # map base check
            self.grid.draw(screen)
            self.grid.update()


            self.pacman.move(self.fps)
            self.pacman.image.change()
            self.pacman.draw(screen)


            current_point = self.grid.check(self.pacman)
            # pacman coordinate check code
            pygame.draw.circle(screen, (100, 100, 255), (self.pacman[0], self.pacman[1]), 8)
            current_point.radius = 10
            current_point.draw(screen)

            if current_point.distance_x(self.pacman[0]) <= self.judgment_distance:
                if self.last_move_command == pygame.K_DOWN:
                    self.pacman.turn_to(270, PacMan_data.ABS_SPEED)
                    self.pacman.move_to(current_point[0], self.pacman[1])
                elif self.last_move_command == pygame.K_UP:
                    self.pacman.turn_to(90, PacMan_data.ABS_SPEED)
                    self.pacman.move_to(current_point[0], self.pacman[1])

            if current_point.distance_y(self.pacman[1]) <= self.judgment_distance:
                if self.last_move_command == pygame.K_RIGHT:
                    self.pacman.turn_to(0, PacMan_data.ABS_SPEED)
                    self.pacman.move_to(self.pacman[0], current_point[1])
                elif self.last_move_command == pygame.K_LEFT:
                    self.pacman.turn_to(180, PacMan_data.ABS_SPEED)
                    self.pacman.move_to(self.pacman[0], current_point[1])


            self.clock.tick(self.fps)
            if self.show_fps:
                fps_text = self.font.render(f"FPS: {self.clock.get_fps():0.2f}", True, (255, 255, 255))
                screen.blit(fps_text, (10, 10))
            pygame.display.flip()


if __name__=="__main__":
    pygame.init()
    screen = pygame.display.set_mode((Screen_data.WIDTH, Screen_data.HEIGHT))
    MainScreen = Main(Screen_data.WIDTH, Screen_data.HEIGHT, 60, show_fps=True)
    MainScreen.loop(screen)
