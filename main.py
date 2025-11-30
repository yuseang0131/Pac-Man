from __future__ import annotations
import sys
import os
import math
import time
import json
import pygame
import numpy as np
from constant import *
import algorithm
from abc import ABC, abstractmethod
from threading import Thread
from collections import deque
from rotate_image import rotate_image_3d

from typing import Iterable



class Command_list:
    def __init__(self, max_len = 10):
        self.queue = deque({})
        self.max_len = max_len
        
    def __getitem__(self, index):
        return self.queue[index]

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


class Interacter(ABC, pygame.sprite.Sprite):
    def __init__(self, target: list[bool], *groups):
        pygame.sprite.Sprite.__init__(self, *groups)
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
    def __init__(self, target, *groups):
        super().__init__(target, *groups)

    def active(self, unit: Unit, interacter: Interacter):
        if interacter is Victim:
            unit.is_alive = False

class Victim(Interacter):
    def __init__(self, target, *groups):
        super().__init__(target, *groups)


    def active(self, unit: Unit, interacter: Interacter):
        if interacter is killer:
            unit.is_alive = False

class Giver(Interacter):
    def __init__(self, target, *groups):
        super().__init__(target, *groups)

class Getter(Interacter):
    def __init__(self, target, *groups):
        super().__init__(target, *groups)


class Image:
    CAM_DIST = Image_data.CAM_DIST
    F = Image_data.F
    SCREEN_CENTER = (float(Screen_data.WIDTH/2), float(Screen_data.HEIGHT/2), 0.0)
        
    def __init__(self, images: list[pygame.Surface], ratio: list, rate: int = 60, theta = 0):
        self.images_origin = images
        self.images = images

        self.size = images[0].get_size()[0]
        self.__ratio = ratio

        self.length = len(self.images)
        self.__number = 0

        self.change_max_late = rate
        self.change_late = rate

        self.ratio_update()
        self.__current = images[0]
        
        self.theta = theta

    @property
    def current(self):
        return self.__current

    @current.setter
    def current(self, n):
        self.__number = n % self.length
        self.__current = self.images[self.__number]
        
    def get_rotate_state(self, coordinate, axis_point = SCREEN_CENTER, axis_dir = (0.0, 1.0, 0.0)):
        rotated_img, pos = rotate_image_3d(
                self.current,
                img_center_2d=coordinate[:2],
                axis_point_3d=axis_point,
                axis_dir_3d=axis_dir,
                theta_rad=self.theta,
                base_z=coordinate[-1],
                cam_dist=self.CAM_DIST,
            )
        
        return rotated_img, pos

    def rotate(self, d_theta):
        d_theta = d_theta * math.pi / 180
        self.theta = (self.theta + d_theta) % (180 / math.pi)
    
    def theta_to(self, theta):
        self.theta = theta
    
    @property
    def ratio(self):
        return self.__ratio

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
            self.images[i] = pygame.transform.scale(self.images_origin[i], (self.size * self.ratio, self.size * self.ratio))

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


class Unit(All):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter,
                 images: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1
                 ):
        All.__init__(self, coordinate, direction, ratio)
        
        self.speed = speed
        self.image = Image(images, ratio, rate= rate)
        self.image_set = [self.image]
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

    def move_to(self, x, y, z = None):
        if not z: z = self.coordinate[2]
        self.coordinate = np.array([x, y, z])

    def turn_to(self, angle, speed):
        if not self.can_act:
            return False
        self.direction = angle
        rad = angle * math.pi / 180
        self.speed = np.array([speed * math.cos(rad), -speed * math.sin(rad), 0])
        return True

    def draw(self, screen: pygame.surface.Surface):
        img, pos = self.image.get_rotate_state(self.coordinate)
        screen.blit(img, pos)


class Item(All):
    def __init__(self, coordinate: np.array, direction: float, ratio, interacter: Interacter,
                 images: list = []):
        All.__init__(coordinate, direction, ratio)
        self.interacter = interacter
        self.image = Image(images, ratio)
        self.image_set = [self.image]

    def draw(self):
        pass

class dots(Item):
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
    def __init__(self, coordinate: np.array, direction: float, size: tuple[list], ratio, gap):
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
        

    def cross_check(self, unit: Unit):
        x, y = unit.coordinate[0] - self.start_point[0], unit.coordinate[1] - self.start_point[1]
        block_x, block_y = round(x/self.gap), round(y/self.gap)
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
        img, pos = self.image.get_rotate_state(self.coordinate)
        
        screen.blit(img, pos)

class Map:
    def __init__(self, grid: Grid, map_data: dict, color, block_gap, ratio, z = Z, grid_show = False):
        self.grid = grid
        self.map_data = map_data["map"]["map"]
        
        self.size = (map_data["map"]["size_x"], map_data["map"]["size_y"])
        
        self.walls: list[Wall] = self.load_wall(map_data, color, block_gap, ratio, z)
        self.item: list[All] = self.load_item(map_data)
        
        self.collision_data = self.load_collision_data(map_data, self.size)
        
        self.grid_show = grid_show
        
    def is_wall(self, point: Point, direction):
        block_x, block_y = self.grid.get_block_coordinate(point)
        dx = int(math.cos(direction * math.pi / 180) * 1.3)
        dy = -int(math.sin(direction * math.pi / 180) * 1.3)
        
        block_x += dx
        block_y += dy
        
        if not (0 <= block_x <= self.size[0] and 0 <= block_y <= self.size[1]):
            return True
        try:
            value = self.collision_data[block_y][block_x]
        except IndexError:
            value = 0

            
        if value != 0:
            return True
        else:
            return False
    
    def cross_check(self, unit: Unit):
        return self.grid.cross_check(unit)
    
    def update(self):
        self.grid.update()
    
    def draw(self, screen: pygame.surface.Surface):
        if self.grid_show:
            self.grid.draw(screen)
            
        for wall in self.walls:
            wall.draw(screen)
            
    def load_wall(self, data, color, block_gap, ratio, z):
        data = data["map"]["map"]
        walls = []
        for coordinate, theta, img in algorithm.wall_make(data, color):
            point = self.grid.center_points[coordinate[0] * (self.size[0]) + coordinate[1]].coordinate
            point = (point[0], point[1], z)
            walls.append(Wall(point, theta, block_gap, ratio, [img]))
        
        return walls

    @staticmethod
    def load_item(data):
        object_data = data["objects"]
        object_list = []
        
        return object_list
    
    @staticmethod
    def load_collision_data(data, size):
        data = data["map"]["map"]
        
        collision_data = [[0] * (size[0] + 1) for _ in range(size[1] + 1)]
        
        for i in range(size[1]):
            for j in range(size[0]):
                if data[i][j] == 0:
                    continue
                collision_data[i][j] = 1
                collision_data[i+1][j] = 1
                collision_data[i][j+1] = 1
                collision_data[i+1][j+1] = 1
        
        
        
        return collision_data


class PacMan(Unit):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter,
                 images: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1):
        super().__init__(coordinate, direction, speed, interact, images[1:], ratio, rate, name=name, number=number)
        self.wait_image = Image(images[0:1], ratio)
        self.image_set = [self.image, self.wait_image]

    def turn_to(self, angle, speed):
        direction = self.direction
        if super().turn_to(angle, speed):
            self.image.turn_to(angle - direction)


class Ghost(Unit):
    STATE_CASE = [
        "chase",
        "scatter",
        "eaten",
        "frightened"
    ]
    
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter, images_eye: list = [],
                 images_body: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1):

        super().__init__(coordinate, direction, speed, interact, images_body, ratio, rate, name=name, number=number)
        self.images_eye = Image(images_eye, ratio, rate=0)
        self.image_set = [self.image, self.images_eye]
        self.state = None

    def turn_to(self, angle, speed):
        if super().turn_to(angle, speed):
            self.images_eye.current = int(angle//(math.pi/2))
            
    def change_interecter(self, interecter: Interacter):
        self.interact = interecter

    def draw(self, screen):
        super().draw(screen)
        img, pos = self.images_eye.get_rotate_state(self.coordinate)
        screen.blit(img, pos)



class Main:
    def __init__(self, width, height, fps, show_fps = False, show_grid = False):
        self.screen_width, self.screen_height = width, height
        
        self.level = 0
        
        with open(f"data/map/map{self.level}.json", 'r') as f:
            self.data = json.load(f)

        self.clock = pygame.time.Clock()
        self.fps = fps
        self.show_fps = show_fps

        self.main_state = 0
        self.running = True

        self.judgment_distance = JUDGMENT_DISTANCE

        self.font = pygame.font.SysFont("Arial", 24)
        
        self.show_grid = show_grid
        self.grid = Grid((self.screen_width/2, self.screen_height/2), 0, (self.data["map"]["size_x"], self.data["map"]["size_y"]), RATIO, Grid_data.BLOCK_GAP)
        self.current_point = None
        
        self.map = Map(self.grid, self.data, Wall_data.COLOR, Grid_data.BLOCK_GAP, RATIO, z=Z, grid_show=self.show_grid)
        
        
        self.rotate_state = False
        self.rotate_time = 0
        

        # --------------------
        # Pac Man init
        # --------------------
        self.pacman = PacMan(PacMan_data.COORDINATE, PacMan_data.DIRECTION, PacMan_data.SPEED,
                             Victim([0, 1, 1, 1, 1 ,1]), PacMan_data.IMAGES,
                             RATIO,
                             PacMan_data.RATE, name=PacMan_data.NAME, number= PacMan_data.NUMBER)
        self.pacman.move_to(*self.map.grid.cross_points[10].coordinate)
        self.pacman.current_point = self.map.grid.cross_points[10]
        self.pacman.wait = False

        # --------------------
        # Ghost init
        # --------------------
        self.blinky = Ghost(Blinky_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            killer([1, 0, 0, 0, 0, 0]), Blinky_DATA.EYE_IMGAES,
                            Blinky_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Bilnky", number=2,)
        self.blinky.move_to(*self.map.grid.cross_points[-5].coordinate)
        self.blinky.current_point = self.map.grid.cross_points[-5]
        
        self.units: list[Unit] = [self.pacman, self.blinky]
        self.ghosts = [self.blinky]
        
        

        # --------------------
        # command init
        # --------------------
        self.command_list = Command_list(max_len= 10)
        self.last_move_command = None
        
        self.thread = Thread(target=self.back_loop)
        self.thread.start()



    def reset(self):
        pass

    def back_loop(self):
        while self.running:
            if self.rotate_state:
                print(1)
                self.rotate_time -= 1
                if self.rotate_time <= 0:
                    self.rotate_state = False
                    self.rotate_time = 0
                self.rotate(1)
            
            time.sleep(0.01)
        

    def loop(self, screen: pygame.surface.Surface):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    self.command_list.append(event.key)

                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    
                    if event.key == pygame.K_F5:
                        self.reset()
                        if not self.rotate_time:
                            self.rotate_state = True
                            self.rotate_time = 90

                    if event.key in [pygame.K_RIGHT, pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN]:
                        self.last_move_command = event.key

            self.update()
            self.turn_unit()
            self.draw(screen)            
            
            self.clock.tick(self.fps)
            if self.show_fps:
                fps_text = self.font.render(f"FPS: {self.clock.get_fps():0.2f}", True, (255, 255, 255))
                screen.blit(fps_text, (10, 10))
            pygame.display.flip()
            
    def turn_unit(self):
        if self.pacman.current_point.distance_x(self.pacman[0]) <= self.judgment_distance:
            if self.last_move_command == pygame.K_DOWN and not self.map.is_wall(self.pacman.current_point, 270):
                self.pacman.turn_to(270, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman.current_point[0], self.pacman[1])
            elif self.last_move_command == pygame.K_UP and not self.map.is_wall(self.pacman.current_point, 90):
                self.pacman.turn_to(90, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman.current_point[0], self.pacman[1])

        if self.pacman.current_point.distance_y(self.pacman[1]) <= self.judgment_distance:
            if self.last_move_command == pygame.K_RIGHT and not self.map.is_wall(self.pacman.current_point, 0):
                self.pacman.turn_to(0, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman[0], self.pacman.current_point[1])
            elif self.last_move_command == pygame.K_LEFT and not self.map.is_wall(self.pacman.current_point, 180):
                self.pacman.turn_to(180, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman[0], self.pacman.current_point[1])

        for ghost in self.ghosts:
            continue
            if (ghost.current_point.distance_x(ghost[0]) <= self.judgment_distance or
                ghost.current_point.distance_y(ghost[1]) <= self.judgment_distance):
                ghost.turn_to(ghost.direction + 90, Ghost_data.ABS_SPEED)
        
    def update(self):
        self.map.update()
        
        for unit in self.units:
            unit.current_point = self.map.cross_check(unit)
            if self.map.is_wall(unit.current_point, unit.direction) and unit.current_point.distance(unit[0], unit[1]) <= self.judgment_distance:
                unit.move_to(unit.current_point[0], unit.current_point[1])
            else:
                unit.move(self.fps)
                unit.image.change()
                
    def rotate(self, d_theta):
        for unit in self.units:
            for img in unit.image_set:
                img.rotate(d_theta)

        for wall in self.map.walls:
            wall.image.rotate(d_theta)

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
    MainScreen = Main(Screen_data.WIDTH, Screen_data.HEIGHT, 60, show_fps=True, show_grid=False)
    MainScreen.loop(screen)
