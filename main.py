import sys
import os
import math
import json
import pygame
import numpy as np
from constant import *
from abc import ABC, abstractmethod
from threading import Thread



class Interacter:
    def __init__(self):
        pass

    def check(self):
        pass

class killer(Interacter):
    pass


class giver(Interacter):
    pass


class getter(Interacter):
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

    @abstractmethod
    def draw(self, screen: pygame.display):
        pass

# --------------------------------

# main three class
class Unit(All):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array, images: list = [], ratio: float = 1):
        super().__init__(coordinate, direction)
        self.speed = speed
        self.image = Image(images, ratio)

    def move(self, fps):
        self.coordinate += self.speed/fps

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
    def __init__(self, coordinate: np.array, direction: float, interacter: Interacter, images: list = []):
        super().__init__(coordinate, direction)
        self.interacter = interacter
        self.image = Image(images)


    def draw(self):
        pass


# --------------------------------


class wall(Structure):
    def __init__(self):
        super().__init__()


class Trap(Structure):
    def __init__(self):
        super().__init__()



class PacMan(Unit):
    def __init__(self, coordinate: np.array, direction: float, speed: np.array, images: list = [], ratio: float = 1):
        super().__init__(coordinate, direction, speed, images, ratio)



class ghost(Unit):
    def __init__(self):
        super().__init__()


class Main:
    def __init__(self, fps, show_fps = False):
        self.thread = Thread(target=self.back_loop)
        self.thread.start()

        self.clock = pygame.time.Clock()
        self.fps = fps
        self.show_fps = show_fps

        self.main_state = 0
        self.running = True

        self.font = pygame.font.SysFont("Arial", 24)

        self.pacman = PacMan(*PacMan_data.INIT_PACK)
        self.last_command = ""


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
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

                    elif event.key == pygame.K_DOWN:
                        self.pacman.turn_to(270, PacMan_data.ABS_SPEED)
                    elif event.key == pygame.K_LEFT:
                        self.pacman.turn_to(180, PacMan_data.ABS_SPEED)
                    elif event.key == pygame.K_UP:
                        self.pacman.turn_to(90, PacMan_data.ABS_SPEED)
                    elif event.key == pygame.K_RIGHT:
                        self.pacman.turn_to(0, PacMan_data.ABS_SPEED)

            screen.fill(Screen_data.COLOR)

            #self.pacman.move(self.fps)

            self.pacman.image.change()
            self.pacman.draw(screen)




            self.clock.tick(self.fps)
            if self.show_fps:
                fps_text = self.font.render(f"FPS: {self.clock.get_fps():0.2f}", True, (255, 255, 255))
                screen.blit(fps_text, (10, 10))
            pygame.display.flip()


if __name__=="__main__":
    pygame.init()
    screen = pygame.display.set_mode((Screen_data.WIDTH, Screen_data.HEIGHT))
    MainScreen = Main(60, show_fps=True)
    MainScreen.loop(screen)
