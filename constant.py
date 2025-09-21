import pygame
import math
import numpy as np

pygame.init()


UNIT_RATIO = 2
UNIT_SIZE = 26
IMG_PATH = "data/imgs"
JUDGMENT_DISTANCE = 5

# ----------------------
# Screen
# ----------------------
class Screen_data:
    INFO = pygame.display.Info()
    WIDTH, HEIGHT = INFO.current_w, INFO.current_h

    COLOR = (0, 0, 0)

# ----------------------
# Pac-Man
# ----------------------
class PacMan_data:
    IMAGES = [
        pygame.image.load(f"{IMG_PATH}/Pac-Man/1.png"),
        pygame.image.load(f"{IMG_PATH}/Pac-Man/2.png")
    ]
    ABS_SPEED = UNIT_SIZE * UNIT_RATIO * 5
    SPEED = np.array([ABS_SPEED, 0, 0])
    COORDINATE = np.array([Screen_data.WIDTH/2, Screen_data.HEIGHT/2, 0])
    DIRECTION = 0
    RATIO = UNIT_RATIO
    SIZE_X = UNIT_SIZE
    SIZE_Y = UNIT_SIZE

    RATE = 5

    INIT_PACK = (COORDINATE, DIRECTION, SPEED, IMAGES, RATIO, RATE)

