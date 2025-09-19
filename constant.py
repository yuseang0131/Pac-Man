import pygame
import math
import numpy as np

pygame.init()


RATIO = 10


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
        pygame.image.load("data/imgs/Pac-Man/1.png")
    ]
    ABS_SPEED = 180
    SPEED = np.array([ABS_SPEED, 0, 0])
    COORDINATE = np.array([Screen_data.WIDTH/2, Screen_data.HEIGHT/2, 0])
    DIRECTION = 0
    RATIO = RATIO
    SIZE_X = 26
    SIZE_Y = 26

    INIT_PACK = (COORDINATE, DIRECTION, SPEED, IMAGES, RATIO)

