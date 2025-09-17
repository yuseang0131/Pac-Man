import sys
import os
import json
import pygame
import numpy
from constant import *
from abc import ABc, abstractmethod


map = json.load(map)

class Unit:
    def __init__(self):
        pass

    def move(self):
        pass

    def turn(self, angle):
        pass



class Object(ABc):
    def __init__(self):
        pass


class interacter:
    def __init__(self):
        pass

    def check(self):
        pass

class killer(interacter):
    pass

class giver(interacter):
    pass





class PacMan(Unit):
    def __init__(self):
        super().__init__()
        pass



class ghost(Unit):
    def __init__(self):
        super().__init__()


