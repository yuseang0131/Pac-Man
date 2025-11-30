from __future__ import annotations
import sys
import os
import math
import time
import json
import random
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
    def __init__(self, tartget, rect, *groups):
        pygame.sprite.Sprite.__init__(self, *groups)
        self.rect: pygame.Rect = rect
        
    def move_to(self, x, y):
        self.rect.center = (x, y)

class Image:
    CAM_DIST = Image_data.CAM_DIST
    F = Image_data.F
    SCREEN_CENTER = (float(Screen_data.WIDTH/2), float(Screen_data.HEIGHT/2), 0.0)
        
    def __init__(self, images: list[pygame.Surface], ratio: list, rate: int = 60, theta = 0, is_static=False):
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
        
        # --- 캐싱용 ---
        self.is_static = is_static
        self._cache_theta = None
        self._cache_img = None
        self._cache_pos = None

    @property
    def current(self):
        return self.__current

    @current.setter
    def current(self, n):
        self.__number = n % self.length
        self.__current = self.images[self.__number]
        
    def get_rotate_state(self, coordinate, axis_point = SCREEN_CENTER, axis_dir = (0.0, 1.0, 0.0)):
        theta = math.radians(self.theta)
        
        if self.is_static and self._cache_theta == self.theta and self._cache_img is not None:
            return self._cache_img, self._cache_pos
        
        rotated_img, pos = rotate_image_3d(
                self.current,
                img_center_2d=coordinate[:2],
                axis_point_3d=axis_point,
                axis_dir_3d=axis_dir,
                theta_rad=theta,
                base_z=coordinate[-1],
                cam_dist=self.CAM_DIST,
            )
        
        if self.is_static:
            self._cache_theta = self.theta
            self._cache_img = rotated_img
            self._cache_pos = pos
        
        return rotated_img, pos

    def rotate(self, d_theta):
        d_theta = d_theta
        self.theta = (self.theta + d_theta) % (360)
    
    def theta_to(self, theta):
        self.theta = theta
    
    @property
    def ratio(self):
        return self.__ratio

    def change(self):
        if self.change_max_late == 0:
            return
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
        self.life = 1

    @property
    def alive(self):
        return self._alive
    
    @alive.setter
    def alive(self, alive: bool):
        if not alive and self.life > 0: 
            self.life -= 1
            self._alive = True
        else:
            self._alive = False
            
    @property
    def can_act(self):
        return self.alive and not self.wait
        

    def move(self, fps):
        if not self.can_act:
            return
        self.coordinate += self.speed/fps
        self.interact.move_to(self.coordinate[0], self.coordinate[1])

    def move_to(self, x, y, z = None):
        if not z: z = self.coordinate[2]
        self.coordinate = np.array([x, y, z])
        self.interact.move_to(self.coordinate[0], self.coordinate[1])

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


class Item(All, pygame.sprite.Sprite):
    def __init__(self, coordinate: np.array, direction: float, ratio, rect,
                 images: list = [], *groups):
        All.__init__(self, coordinate, direction, ratio)
        pygame.sprite.Sprite.__init__(self, *groups)
        
        self.rect = rect
        
        self.image = Image(images, ratio, is_static=True)

    def draw(self, screen):
        img, pos = self.image.get_rotate_state(self.coordinate)
        screen.blit(img, pos)
        
    @abstractmethod
    def active(self):
        pass
        
class Dot(Item):
    TOTAL_NUMBER = 0
    SCORE = 0
    def __init__(self, coordinate, direction, ratio, rect, score, images = [], *groups):
        super().__init__(coordinate, direction, ratio, rect, images, *groups)
        self.score = score
        self.number = Dot.TOTAL_NUMBER
        Dot.TOTAL_NUMBER += 1
        
    def active(self, *_):
        Dot.SCORE += 1
        
    @classmethod
    def reset(cls):
        cls.TOTAL_NUMBER = 0
        cls.SCORE = 0

class Power(Item):
    def __init__(self, coordinate, direction, ratio, rect, images = [], *groups):
        super().__init__(coordinate, direction, ratio, rect, images, *groups)
        
    def active(self, pacman: PacMan, ghosts: list[Ghost]):
        for ghost in ghosts:
            ghost.state = "Scared"
        
        pacman.power_on = True
        pacman.power_time = 9 * 100
        

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
            self.row.append(Line((self.start_point[0], y), (self.end_point[0], y), color= self.color))

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
        
        self.image = Image(image, ratio, is_static=True)
        self.image.turn_to(direction)
    
    def draw(self, screen: pygame.surface.Surface):
        img, pos = self.image.get_rotate_state(self.coordinate)
        
        screen.blit(img, pos)

class Map:
    def __init__(self, grid: Grid, map_data: dict, color, block_gap, ratio, theta, z = Z, grid_show = False):
        self.grid = grid
        self.map_data = map_data["map"]["map"]
        
        self.size = (map_data["map"]["size_x"], map_data["map"]["size_y"])
        
        self.collision_data = self.load_collision_data(map_data, self.size)
        self.walls: list[Wall] = self.load_wall(map_data, color, block_gap, ratio, z)
        self.item: list[Item] = self.load_item(map_data, ratio, z)
        
        self.grid_show = grid_show
        
        self.rotate_to(theta)
        
    def rotate_to(self, theta):
        for wall in self.walls:
            wall.image.theta_to(theta)
        
        for item in self.item:
            item.image.theta_to(theta)
        
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

            
        if value == 1:
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

    def load_item(self, data, ratio, z):
        map_data = data["map"]["map"]
        object_data = data["objects"]
        object_list = []
        
        for i, point in enumerate(self.grid.cross_points):
            if self.collision_data[i // (self.size[0] + 1)][i % (self.size[0] + 1)] not in [1, 2]:
                point = point.coordinate
                point = (point[0], point[1], z)
                object_list.append(Dot(point, 0, ratio, Item_data.get_DOT_INTERECT_RECT(point[0], point[1]), 1, [Item_data.DOT_IMAGE]))
        
        for cls, coordinate in object_data:
            point = self.grid.cross_points[coordinate[1] * (self.size[0]) + coordinate[0]]
            point = point.coordinate
            point = (point[0], point[1], z)
            object_list.append(Power(point, 0, ratio, Item_data.get_POWER_INTERECT_RECT(point[0], point[1]), [Item_data.POWER_IMAGE]))
            
        
        return object_list
    
    @staticmethod
    def load_collision_data(data, size):
        data = data["map"]["map"]
        
        collision_data = [[0] * (size[0] + 1) for _ in range(size[1] + 1)]
        
        for i in range(size[1]):
            for j in range(size[0]):
                if data[i][j] == 2:
                    collision_data[i][j] = 2
                    collision_data[i+1][j] = 2
                    collision_data[i][j+1] = 2
                    collision_data[i+1][j+1] = 2
                elif data[i][j] == 1:
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
        
        self.power_on = False
        self.power_time = 0

    def turn_to(self, angle, speed):
        direction = self.direction
        if super().turn_to(angle, speed):
            self.image.turn_to(angle - direction)
            
    def draw(self, screen):
        if self.can_act:
            super().draw(screen)
        else:
            img, pos = self.wait_image.get_rotate_state(self.coordinate)
            screen.blit(img, pos)


class Ghost(Unit):
    STATE_CASE = [
        "chase",
        "scatter",
        "eaten",
        "frightened"
    ]
    
    def __init__(self, coordinate: np.array, direction: float, speed: np.array,
                 interact: Interacter, images_eye: list = [],
                 images_body: list = [], ratio: float = 1, rate: int = 60, name: str = "", number: int = -1, images_scared: list = Ghost_data.SCARED_IMAGES):

        super().__init__(coordinate, direction, speed, interact, images_body[:], ratio, rate, name=name, number=number)
        self.images_eye = Image(images_eye[:], ratio, rate=0)
        self.images_scared = Image(images_scared[:], ratio, rate=rate)
        
        self.image_set = [self.image, self.images_eye, self.images_scared]
        self.state = None

    def turn_to(self, angle, speed):
        if super().turn_to(angle, speed):
            self.images_eye.current = int(angle//(math.pi/2))

    def draw(self, screen):
        if self.state == "Scared" or self.state == "scared":
            img, pos = self.images_scared.get_rotate_state(self.coordinate)
            screen.blit(img, pos)
        else:
            super().draw(screen)
            img, pos = self.images_eye.get_rotate_state(self.coordinate)
            screen.blit(img, pos)


class Main:
    def __init__(self, width, height, fps, show_fps = False, show_grid = False):
        self.screen_width, self.screen_height = width, height
        
        self.level = 0
        self.state = 0
        
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
        
        self.maps = [Map(Grid((self.screen_width/2, self.screen_height/2), 0, (self.data[0]["map"]["size_x"], self.data[0]["map"]["size_y"]), RATIO, Grid_data.BLOCK_GAP), self.data[0], Wall_data.COLOR, Grid_data.BLOCK_GAP, RATIO, 0, z=Z, grid_show=self.show_grid),
                     Map(Grid((self.screen_width/2, self.screen_height/2), 0, (self.data[1]["map"]["size_x"], self.data[1]["map"]["size_y"]), RATIO, Grid_data.BLOCK_GAP), self.data[1], Wall_data.COLOR, Grid_data.BLOCK_GAP, RATIO, -90, z=Z, grid_show=self.show_grid)
                     ]        
        
        self.current_map_number = 0
        self.map_max_number = 1
        
        for map in self.maps:
            map.item = pygame.sprite.Group(*map.item)
        
        
        self.rotate_state = False
        self.rotate_time = 0
        
        # --------------------
        # groups init
        # --------------------
        self.ghost_group = pygame.sprite.Group()
        self.pacman_group = pygame.sprite.Group()
        self.item_gruop = self.maps[self.current_map_number].item
    

        # --------------------
        # Pac Man init
        # --------------------
        self.pacman = PacMan(PacMan_data.COORDINATE, PacMan_data.DIRECTION, PacMan_data.SPEED,
                             Interacter(self.ghost_group, PacMan_data.INTERECTER_RECT, self.pacman_group), PacMan_data.IMAGES,
                             RATIO,
                             PacMan_data.RATE, name=PacMan_data.NAME, number= PacMan_data.NUMBER)
        self.pacman.move_to(*self.maps[self.current_map_number].grid.cross_points[41].coordinate)
        
        
        self.pacman.wait = False       
        
        # --------------------
        # Ghost init
        # --------------------
        self.blinky = Ghost(Blinky_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            Interacter(self.pacman_group, Blinky_DATA.INTERECTER_RECT, self.ghost_group), Blinky_DATA.EYE_IMGAES,
                            Blinky_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Bilnky", number=2,)
        self.blinky.move_to(*self.maps[self.current_map_number].grid.cross_points[-46].coordinate)
        
        '''self.pinky = Ghost(Pinky_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            Interacter(self.pacman_group, Pinky_DATA.INTERECTER_RECT, self.ghost_group), Pinky_DATA.EYE_IMGAES,
                            Pinky_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Pinky", number=5,)
        self.pinky.move_to(*self.maps[self.current_map_number].grid.cross_points[3].coordinate)
        self.pinky.current_point = self.maps[self.current_map_number].grid.cross_points[3]'''
        
        self.clyde = Ghost(Clyde_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            Interacter(self.pacman_group, Clyde_DATA.INTERECTER_RECT, self.ghost_group), Clyde_DATA.EYE_IMGAES,
                            Clyde_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Clyde", number=3,)
        self.clyde.move_to(*self.maps[self.current_map_number].grid.cross_points[-48].coordinate)
        
        '''self.lnky = Ghost(Lnky_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            Interacter(self.pacman_group, Lnky_DATA.INTERECTER_RECT, self.ghost_group), Lnky_DATA.EYE_IMGAES,
                            Lnky_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Lnky", number=4)
        self.lnky.move_to(*self.maps[self.current_map_number].grid.cross_points[-2].coordinate)
        self.lnky.current_point = self.maps[self.current_map_number].grid.cross_points[-2]'''
        
        self.units: list[Unit] = [self.pacman, self.blinky, self.clyde]
        self.ghosts = [self.blinky, self.clyde]
        
        

        # --------------------
        # command init
        # --------------------
        self.command_list = Command_list(max_len= 10)
        self.last_move_command = None
        
        self.thread = Thread(target=self.back_loop)
        self.thread.start()
        self.update()

    def reset(self):
        Dot.reset()
        
        self.state = 1
        self.maps = [Map(Grid((self.screen_width/2, self.screen_height/2), 0, (self.data[0]["map"]["size_x"], self.data[0]["map"]["size_y"]), RATIO, Grid_data.BLOCK_GAP), self.data[0], Wall_data.COLOR, Grid_data.BLOCK_GAP, RATIO, 0, z=Z, grid_show=self.show_grid),
                     Map(Grid((self.screen_width/2, self.screen_height/2), 0, (self.data[1]["map"]["size_x"], self.data[1]["map"]["size_y"]), RATIO, Grid_data.BLOCK_GAP), self.data[1], Wall_data.COLOR, Grid_data.BLOCK_GAP, RATIO, -90, z=Z, grid_show=self.show_grid)
                     ]
        
        self.current_map_number = 0
        self.map_max_number = 1
        
        for map in self.maps:
            map.item = pygame.sprite.Group(*map.item)
        
        self.rotate_state = False
        self.rotate_time = 0
        
        # --------------------
        # groups init
        # --------------------
        self.ghost_group = pygame.sprite.Group()
        self.pacman_group = pygame.sprite.Group()
        self.item_gruop = self.maps[self.current_map_number].item
    
        # --------------------
        # Pac Man init
        # --------------------
        self.pacman = PacMan(PacMan_data.COORDINATE, PacMan_data.DIRECTION, PacMan_data.SPEED,
                             Interacter(self.ghost_group, PacMan_data.INTERECTER_RECT, self.pacman_group), PacMan_data.IMAGES,
                             RATIO,
                             PacMan_data.RATE, name=PacMan_data.NAME, number= PacMan_data.NUMBER)
        self.pacman.move_to(*self.maps[self.current_map_number].grid.cross_points[41].coordinate)
        self.pacman.current_point = self.maps[self.current_map_number].grid.cross_points[41]
        
        self.pacman.wait = False       
        
        # --------------------
        # Ghost init
        # --------------------
        self.blinky = Ghost(Blinky_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            Interacter(self.pacman_group, Blinky_DATA.INTERECTER_RECT, self.ghost_group), Blinky_DATA.EYE_IMGAES,
                            Blinky_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Bilnky", number=2,)
        self.blinky.move_to(*self.maps[self.current_map_number].grid.cross_points[-46].coordinate)
        
        '''self.pinky = Ghost(Pinky_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            Interacter(self.pacman_group, Pinky_DATA.INTERECTER_RECT, self.ghost_group), Pinky_DATA.EYE_IMGAES,
                            Pinky_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Pinky", number=5,)
        self.pinky.move_to(*self.maps[self.current_map_number].grid.cross_points[3].coordinate)
        self.pinky.current_point = self.maps[self.current_map_number].grid.cross_points[3]'''
        
        self.clyde = Ghost(Clyde_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            Interacter(self.pacman_group, Clyde_DATA.INTERECTER_RECT, self.ghost_group), Clyde_DATA.EYE_IMGAES,
                            Clyde_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Clyde", number=3,)
        self.clyde.move_to(*self.maps[self.current_map_number].grid.cross_points[-48].coordinate)
        
        '''self.lnky = Ghost(Lnky_DATA.COORDINATE, 0, Ghost_data.SPEED,
                            Interacter(self.pacman_group, Lnky_DATA.INTERECTER_RECT, self.ghost_group), Lnky_DATA.EYE_IMGAES,
                            Lnky_DATA.BODY_IMAGES, RATIO, PacMan_data.RATE*2, name="Lnky", number=4)
        self.lnky.move_to(*self.maps[self.current_map_number].grid.cross_points[-2].coordinate)
        self.lnky.current_point = self.maps[self.current_map_number].grid.cross_points[-2]'''
        
        self.units: list[Unit] = [self.pacman, self.blinky, self.clyde]
        self.ghosts = [self.blinky, self.clyde]
        
        

        # --------------------
        # command init
        # --------------------
        self.command_list = Command_list(max_len= 10)
        self.last_move_command = None
        
        self.thread = Thread(target=self.back_loop)
        self.thread.start()
        self.update()
    
    def position_reset(self):
        self.state = 1
        self.pacman.move_to(*self.maps[self.current_map_number].grid.cross_points[48])

    def back_loop(self):
        while self.running:
            
            if self.rotate_state:
                    self.rotate_time -= 1
                    if self.rotate_time <= 0:
                        self.rotate_state = False
                        self.rotate_time = 0
                        if self.current_map_number == 0:
                            self.current_map_number = 1
                            self.blinky.move_to(*self.maps[self.current_map_number].grid.cross_points[58])
                            self.clyde.move_to(*self.maps[self.current_map_number].grid.cross_points[64])
                            self.pacman.move_to(*self.maps[self.map_max_number].grid.cross_points[119])
                            
                            
                        elif self.current_map_number == 1:
                            self.current_map_number = 0
                            self.blinky.move_to(*self.maps[self.current_map_number].grid.cross_points[-46])
                            self.clyde.move_to(*self.maps[self.current_map_number].grid.cross_points[-48])
                            self.pacman.move_to(*self.maps[self.map_max_number].grid.cross_points[135])
                            
                        # unit 
                        self.item_gruop = self.maps[self.current_map_number].item
                        self.maps[self.map_max_number].rotate_to(0)
                        for unit in self.units:
                            for image in unit.image_set:
                                image.theta_to(0)
                        
                        
                        
                    if self.current_map_number == 0:
                        self.rotate(1)
                    elif self.current_map_number == 1:
                        self.rotate(-1)
                    
            
            if self.pacman.power_on:
                self.pacman.power_time -= 1
                if self.pacman.power_time <= 0:
                    self.pacman.power_on = False
                    self.pacman.power_time = 0
                    
                    self.blinky.state = None
                    self.clyde.state = None
                    
            
            
            time.sleep(0.01)

    def loop(self, screen: pygame.surface.Surface):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                if self.state == 0:
                    self.handle_opening_event(event)

                if event.type == pygame.KEYDOWN:
                    if self.state == 0:
                        self.state = 1
                        
                    self.command_list.append(event.key)

                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    
                    if event.key == pygame.K_F5:
                        self.reset()
                        
                    if event.key == pygame.K_F6:
                        if not self.rotate_time:
                            self.rotate_state = True
                            self.rotate_time = 90

                    if event.key in [pygame.K_RIGHT, pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN]:
                        self.last_move_command = event.key

            if self.state == 0:
                self.show_opening(screen)
                
            elif self.state == 1:
                self.count_down(screen, 3, 0.4)
                self.state = 2
                
            elif self.state == 2:
                if not self.pacman.alive:
                    self.state = 3
                    continue
            
                self.collide()
                self.update()
                self.turn_unit()
                self.draw(screen)
                
                if Dot.SCORE == Dot.TOTAL_NUMBER:
                    self.state = 4
                
                x, y = self.maps[self.current_map_number].grid.get_block_coordinate(self.pacman.current_point)
                if self.maps[self.current_map_number].collision_data[y][x] == 2 and self.maps[self.current_map_number].is_wall(self.pacman.current_point, self.pacman.direction) and not self.rotate_state:
                    self.rotate_state = True
                    self.rotate_time = 90
                    self.item_gruop = pygame.sprite.Group(self.maps[0].item, self.maps[1].item)
                

                score_text = self.font.render(f"SCORE: {Dot.SCORE}", True, (255, 255, 255))
                screen.blit(score_text, (self.screen_width - 200, 50))
                
            elif self.state == 3:
                self.game_over(screen)
                
            elif self.state == 4:
                self.show_game_clear(screen)
            
            self.clock.tick(self.fps)
            if self.show_fps:
                fps_text = self.font.render(f"FPS: {self.clock.get_fps():0.2f}", True, (255, 255, 255))
                screen.blit(fps_text, (10, 10))
            pygame.display.flip()
            
    def turn_unit(self):
        if self.pacman.current_point.distance_x(self.pacman[0]) <= self.judgment_distance:
            if self.last_move_command == pygame.K_DOWN and not self.maps[self.current_map_number].is_wall(self.pacman.current_point, 270):
                self.pacman.turn_to(270, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman.current_point[0], self.pacman[1])
            elif self.last_move_command == pygame.K_UP and not self.maps[self.current_map_number].is_wall(self.pacman.current_point, 90):
                self.pacman.turn_to(90, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman.current_point[0], self.pacman[1])

        if self.pacman.current_point.distance_y(self.pacman[1]) <= self.judgment_distance:
            if self.last_move_command == pygame.K_RIGHT and not self.maps[self.current_map_number].is_wall(self.pacman.current_point, 0):
                self.pacman.turn_to(0, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman[0], self.pacman.current_point[1])
            elif self.last_move_command == pygame.K_LEFT and not self.maps[self.current_map_number].is_wall(self.pacman.current_point, 180):
                self.pacman.turn_to(180, PacMan_data.ABS_SPEED)
                self.pacman.move_to(self.pacman[0], self.pacman.current_point[1])

        # -------------------
        # Ghost 자동 방향 전환
        # -------------------
        for ghost in self.ghosts:
            # 교차점 근처에 있을 때만 방향 결정을 한다.
            if (ghost.current_point.distance_x(ghost[0]) > self.judgment_distance and
                ghost.current_point.distance_y(ghost[1]) > self.judgment_distance):
                continue

            # 고스트의 현재 방향을 0/90/180/270 중 가장 가까운 값으로 정규화
            current_dir = ghost.direction % 360
            current_dir = (round(current_dir / 90) * 90) % 360

            directions = [0, 90, 180, 270]

            # 네 방향에 대해 벽 여부 검사
            wall_info = {}
            for d in directions:
                wall_info[d] = self.maps[self.current_map_number].is_wall(ghost.current_point, d)

            # 앞 방향(현재 진행 방향)에 벽이 있는지
            front_wall = self.maps[self.current_map_number].is_wall(ghost.current_point, current_dir)

            # 벽인 방향 개수
            wall_count = sum(1 for v in wall_info.values() if v)

            # 조건:
            # 1) 네 방향 중 벽이 3개 이상이거나
            # 2) 현재 방향 앞에 벽이 있는 경우
            if wall_count >= 3 or front_wall:
                # 갈 수 있는(벽이 아닌) 방향 후보
                candidates = [d for d, is_wall in wall_info.items() if not is_wall]

                # 모두 막혀 있는 경우(이론상 거의 없지만 방어 코드)
                if not candidates:
                    continue

                new_dir = random.choice(candidates)

                # 방향 전환 및 타일 중앙으로 스냅
                if ghost.turn_to(new_dir, Ghost_data.ABS_SPEED):
                    ghost.move_to(ghost.current_point[0], ghost.current_point[1])       
        
    def collide(self):
        item_collide = pygame.sprite.groupcollide(self.pacman_group, self.item_gruop, False, True)
        if item_collide:
            for item in item_collide.values():
                for i in item:
                    i.active(self.pacman, [self.blinky, self.clyde])
        
        ghost_collide = pygame.sprite.groupcollide(self.pacman_group, self.ghost_group, False, False)
        if ghost_collide:
            if not self.pacman.power_on:
                self.pacman.alive = False
            
                if self.pacman.alive:
                    self.state = 3
                else:
                    self.position_reset(self.current_map_number)
            elif False:
                for ghost in ghost_collide.values():
                    for i in ghost:
                        i.move_to(*self.maps[self.current_map_number].grid.cross_points[-5].coordinate)
                        i.current_point = self.maps[self.current_map_number].grid.cross_points[-5]
        
    def update(self):
        self.maps[self.current_map_number].update()
        
        for unit in self.units:
            unit.current_point = self.maps[self.current_map_number].cross_check(unit)
            if self.maps[self.current_map_number].is_wall(unit.current_point, unit.direction) and unit.current_point.distance(unit[0], unit[1]) <= self.judgment_distance:
                unit.move_to(unit.current_point[0], unit.current_point[1])
            else:
                unit.move(self.fps)
                for image in unit.image_set:
                    image.change()
                
    def rotate(self, d_theta):
        for unit in self.units:
            for img in unit.image_set:
                img.rotate(d_theta)

        for map in self.maps:
            for wall in map.walls:
                wall.image.rotate(d_theta)
            
            for item in map.item:
                item.image.rotate(d_theta)

    def draw(self, screen: pygame.surface.Surface):
        screen.fill(Screen_data.COLOR)

        # map base check
        if self.rotate_state:
            for map in self.maps:
                map.draw(screen)
        else:
            self.maps[self.current_map_number].draw(screen)
            
        for item in self.maps[self.current_map_number].item:
            item.draw(screen)

        
        for unit in self.units:
            unit.draw(screen)
            
        if self.show_grid:
            for unit in self.units:
                unit.current_point.radius = 10
                unit.current_point.draw(screen)

    def show_opening(self, screen: pygame.surface.Surface):
        # --- 폰트 설정 ---
        title_font = pygame.font.SysFont("Arial", 72, True)
        sub_font   = pygame.font.SysFont("Arial", 32, True)
        info_font  = pygame.font.SysFont("Arial", 18)

        # --- START 버튼 Rect 초기화 (한 번만 계산) ---
        if not hasattr(self, "start_button_rect"):
            btn_w, btn_h = 260, 80
            self.start_button_rect = pygame.Rect(0, 0, btn_w, btn_h)
            self.start_button_rect.center = (
                self.screen_width // 2,
                self.screen_height * 3 // 4,
            )

        screen.fill(Screen_data.COLOR)

        # ----- 타이틀 -----
        title_surf = title_font.render("CAP-MAN", True, (255, 255, 0))
        title_rect = title_surf.get_rect(
            center=(self.screen_width // 2, self.screen_height // 4)
        )
        screen.blit(title_surf, title_rect)

        # ----- START 버튼 -----
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.start_button_rect.collidepoint(mouse_pos)

        # 배경 색 (호버 시 살짝 밝게)
        base_color   = (40, 40, 40)
        hover_color  = (80, 80, 80)
        border_color = (255, 255, 255)

        pygame.draw.rect(
            screen,
            hover_color if hovered else base_color,
            self.start_button_rect,
            border_radius=10,
        )
        pygame.draw.rect(
            screen,
            border_color,
            self.start_button_rect,
            width=3,
            border_radius=10,
        )

        # 버튼 텍스트
        start_surf = sub_font.render("START", True, (255, 255, 255))
        start_rect = start_surf.get_rect(center=self.start_button_rect.center)
        screen.blit(start_surf, start_rect)

        # 하단에 조작 설명
        info_surf = info_font.render("ESC: exit   F5: restart", True, (200, 200, 200))
        info_rect = info_surf.get_rect(
            center=(self.screen_width // 2, self.screen_height - 30)
        )
        screen.blit(info_surf, info_rect)

    def handle_opening_event(self, event: pygame.event.EventType):
        """
        오프닝(state == 0) 상태에서만 호출해서
        START 버튼 클릭을 감지하는 함수.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # start_button_rect 위를 클릭했는지 검사
            if hasattr(self, "start_button_rect") and \
            self.start_button_rect.collidepoint(event.pos):
                self.state = 1  # 게임 시작 상태로 전환

    def count_down(self, screen, start_number, late = 1):
        # 반투명 어두운 오버레이
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)  # 어두운 정도 (0~255)
        overlay.fill((0, 0, 0))

        big_font = pygame.font.SysFont("Arial", 96, True)
        small_font = pygame.font.SysFont("Arial", 64, True)

        # --------- 숫자 카운트다운 ---------
        for num in range(start_number, 0, -1):
            start_time = pygame.time.get_ticks()

            while pygame.time.get_ticks() - start_time < int(1000 * late):  # 각 숫자 1초씩
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return

                # 기존 게임 화면 먼저 그림
                self.draw(screen)

                # 그 위에 어두운 오버레이 + 숫자
                screen.blit(overlay, (0, 0))

                text_surf = big_font.render(str(num), True, (255, 255, 255))
                text_rect = text_surf.get_rect(center=(self.screen_width // 2,
                                                       self.screen_height // 2))
                screen.blit(text_surf, text_rect)

                pygame.display.flip()
                self.clock.tick(self.fps)

        # --------- START! 표시 ---------
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < int(800 * late):  # 0.8초 정도
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

            self.draw(screen)
            screen.blit(overlay, (0, 0))

            start_surf = small_font.render("START!", True, (255, 255, 0))
            start_rect = start_surf.get_rect(center=(self.screen_width // 2,
                                                     self.screen_height // 2))
            screen.blit(start_surf, start_rect)

            pygame.display.flip()
            self.clock.tick(self.fps)
    
    def game_over(self, screen: pygame.surface.Surface):
        """
        게임 오버 화면:
        - 화면을 어둡게 덮고
        - GAME OVER + SCORE
        - R: 다시 시작 / ESC: 종료
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)  # 어두운 정도
        overlay.fill((0, 0, 0))

        big_font = pygame.font.SysFont("Arial", 96, True)
        small_font = pygame.font.SysFont("Arial", 32, True)

        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # 다시 시작
                        self.reset()
                        # 카운트다운 함수 만들어놨으면 여기서 써도 됨
                        # self.countdown(screen, start_number=3)
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        # 종료
                        self.running = False
                        return

            # 현재 게임 화면 그리기
            self.draw(screen)

            # 어둡게 덮기
            screen.blit(overlay, (0, 0))

            # GAME OVER 텍스트
            text_surf = big_font.render("GAME OVER", True, (255, 0, 0))
            text_rect = text_surf.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 - 40)
            )
            screen.blit(text_surf, text_rect)

            # 점수 표시
            score_surf = small_font.render(f"SCORE: {Dot.SCORE}", True, (255, 255, 255))
            score_rect = score_surf.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + 20)
            )
            screen.blit(score_surf, score_rect)

            # 안내 문구
            info_surf = small_font.render("R: restart   ESC: exit", True, (200, 200, 200))
            info_rect = info_surf.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + 80)
            )
            screen.blit(info_surf, info_rect)

            pygame.display.flip()
            self.clock.tick(self.fps)

    def show_game_clear(self, screen: pygame.surface.Surface):
        """
        게임 클리어 화면:
        - 화면을 어둡게 덮고
        - STAGE CLEAR! + SCORE
        - R: 다시 시작 / ESC: 종료
        (나중에 N: 다음 스테이지 추가 가능)
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))

        big_font = pygame.font.SysFont("Arial", 88, True)
        small_font = pygame.font.SysFont("Arial", 32, True)

        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # 같은 맵 다시 시작
                        self.reset()
                        # 필요하면 카운트다운
                        # self.countdown(screen, start_number=3)
                        waiting = False

                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                        return

                    # 나중에 여러 스테이지 만들면:
                    # elif event.key == pygame.K_n:
                    #     self.level += 1
                    #     self.load_level(self.level)
                    #     self.countdown(screen, start_number=3)
                    #     waiting = False

            # 현재 게임 화면 그리기
            self.draw(screen)

            # 어둡게 덮기
            screen.blit(overlay, (0, 0))

            # STAGE CLEAR 텍스트
            clear_surf = big_font.render("GAME CLEAR!", True, (0, 255, 0))
            clear_rect = clear_surf.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 - 40)
            )
            screen.blit(clear_surf, clear_rect)

            # 점수 표시
            score_surf = small_font.render(f"SCORE: {Dot.SCORE}", True, (255, 255, 255))
            score_rect = score_surf.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + 20)
            )
            screen.blit(score_surf, score_rect)

            # 안내 문구
            info_surf = small_font.render("R: restart   ESC: exit", True, (200, 200, 200))
            info_rect = info_surf.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + 80)
            )
            screen.blit(info_surf, info_rect)

            pygame.display.flip()
            self.clock.tick(self.fps)

    
if __name__=="__main__":
    pygame.init()
    screen = pygame.display.set_mode((Screen_data.WIDTH, Screen_data.HEIGHT))
    MainScreen = Main(Screen_data.WIDTH, Screen_data.HEIGHT, 40, show_fps=True, show_grid=False)
    MainScreen.loop(screen)
