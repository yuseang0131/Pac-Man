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

class Image:
    def __init__(self, images: list, rate: int = 60):
        self.images = images
        self.__current = images[0]

        self.length = len(self.images)
        self.__number = 0

        self.__change_max_late = rate
        self.__change_late = rate

    @property
    def change_max_late(self):
        return self.__change_max_late

    @change_max_late.setter
    def change_max_late(self, n):
        self.__change_max_late = n

    @property
    def current(self):
        return self.__current

    @current.setter
    def current(self, n):
        self.__number = n % self.length
        self.__current = self.images[self.__number]

    def change(self, n):
        self.__change_late = (self.__change_late + 1) % self.__change_max_late
        if self.__change_late == 0:
            self.current = self.__number + 1

class All(ABC):
    def __init__(self, coordinate: np.array[float], direction: float):
        self.coordinate = coordinate
        self.direction = direction

    @abstractmethod
    def draw(self, screen: pygame.display):
        pass

# --------------------------------

# main three class
class Unit(All):
    def __init__(self, coordinate: np.array[float], direction: float, speed: np.array[float], images: list = []):
        super().__init__(coordinate, direction)
        self.speed = speed
        self.image = Image(images)

    def move(self, fps):
        self.coordinate += self.speed/fps

    def turn(self, angle):
        self.direction = (self.direction + angle) % (2 * math.pi)

    def draw(self, screen: pygame.display):
        screen.bilt(self.image.current)


class Structure(All):
    def __init__(self, coordinate: np.array[float], direction: float):
        super().__init__(coordinate, direction)


    def draw(self):
        pass


class Item(All):
    def __init__(self, coordinate: np.array[float], direction: float, interacter: Interacter, images: list = []):
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
    def __init__(self):
        super().__init__()
        pass



class ghost(Unit):
    def __init__(self):
        super().__init__()


class Main:
    def __init__(self):
        self.thread = Thread(target=self.back_loop)
        self.thread.start()

        self.main_state = 0
        pass

    def reset(self):
        pass

    def back_loop(self):
        pass

    def loop(self, screen):
        pass


if "__name__"=="__main__":
    pygame.init()
    SCREEN = pygame.display
