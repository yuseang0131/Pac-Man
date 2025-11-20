import pygame
import math
import numpy as np

pygame.init()


RATIO = 4
UNIT_SIZE = 14
IMG_PATH = "data/imgs"
JUDGMENT_DISTANCE = 5
UNIT_SET = ["PacMan", "LadyPacMan", "Blinky", "Clyde", "Lnky", "Pinky"]

# ----------------------
# Screen
# ----------------------
class Screen_data:
    INFO = pygame.display.Info()
    WIDTH, HEIGHT = INFO.current_w, INFO.current_h

    COLOR = (0, 0, 0)


class Grid_data:
    BLOCK_GAP = 8


# ----------------------
# Pac-Man
# ----------------------
class PacMan_data:
    NUMBER = 0
    NAME = "PacMan"

    IMAGES = [
        pygame.image.load(f"{IMG_PATH}/Pac-Man/1.png"),
        pygame.image.load(f"{IMG_PATH}/Pac-Man/2.png")
    ]
    ABS_SPEED = float(UNIT_SIZE * RATIO * 5)
    SPEED = np.array([ABS_SPEED, 0.0, 0.0])
    COORDINATE = np.array([Screen_data.WIDTH/2, Screen_data.HEIGHT/2, 0])
    DIRECTION = 0
    RATIO = RATIO
    SIZE_X = UNIT_SIZE
    SIZE_Y = UNIT_SIZE

    RATE = 5



class Ghost_data:
    BLINKY_COLOR = (237, 28, 36)
    CLYDE_COLOR = (222, 184, 70)
    LNKY_COLOR = (57, 222, 203)
    PINKY_COLOR = (255, 174, 201)

    EYE_IMAGES_ORDER = ["right.png", "up.png", "left.png", "down.png"]
    BODY_IMAGES_ORDER = ["1.png", "2.png"]


    def __init__(self, color):
        self.COLOR = color
        self.RATIO = RATIO
        self.EYE_IMGAES = []
        self.BODY_IMAGES = []

        for i in Ghost_data.BODY_IMAGES_ORDER:
            image = pygame.image.load(f"{IMG_PATH}/Ghost/{i}")
            image.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
            self.BODY_IMAGES.append(image)

        for i in Ghost_data.EYE_IMAGES_ORDER:
            self.EYE_IMGAES.append(pygame.image.load(f"{IMG_PATH}/Ghost/{i}"))


Blinky_DATA = Ghost_data(Ghost_data.BLINKY_COLOR)
Clyde_DATA = Ghost_data(Ghost_data.CLYDE_COLOR)
Lnky_DATA = Ghost_data(Ghost_data.LNKY_COLOR)
Pinky_DATA = Ghost_data(Ghost_data.PINKY_COLOR)
