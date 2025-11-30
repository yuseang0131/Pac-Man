import pygame
import math
import numpy as np

pygame.init()


RATIO = 5
IMG_PATH = "data/imgs"
JUDGMENT_DISTANCE = RATIO
UNIT_SET = ["PacMan", "LadyPacMan", "Blinky", "Clyde", "Lnky", "Pinky"]

Z = -200

# ----------------------
# Screen
# ----------------------
class Screen_data:
    INFO = pygame.display.Info()
    WIDTH, HEIGHT = INFO.current_w, INFO.current_h
    WIDTH, HEIGHT = 1080, 720

    COLOR = (0, 0, 0)

class Image_data:
    CAM_DIST = 900.0
    F = 700.0
    SCRENN_CENTER = (Screen_data.WIDTH/2, Screen_data.HEIGHT/2)

class Grid_data:
    BLOCK_GAP = 8

class Wall_data:
    COLOR = (33, 33, 222)
    IMAGES = {
        "end00": pygame.image.load(f"{IMG_PATH}/Wall/end00.png"),
        "end01": pygame.image.load(f"{IMG_PATH}/Wall/end01.png"),
        "end10": pygame.image.load(f"{IMG_PATH}/Wall/end10.png"),
        "end11": pygame.image.load(f"{IMG_PATH}/Wall/end11.png"),
        
        "l00": pygame.image.load(f"{IMG_PATH}/Wall/line00.png"),
        "l01": pygame.image.load(f"{IMG_PATH}/Wall/line01.png"),
        "l10": pygame.image.load(f"{IMG_PATH}/Wall/line10.png"),
        "l11": pygame.image.load(f"{IMG_PATH}/Wall/line11.png"),
        
        "turn": pygame.image.load(f"{IMG_PATH}/Wall/turn.png"),
        
        10: pygame.image.load(f"{IMG_PATH}/Wall/t_line01.png"),
        11: pygame.image.load(f"{IMG_PATH}/Wall/t_line10.png"),
        12: pygame.image.load(f"{IMG_PATH}/Wall/t_line00.png"),
        17: pygame.image.load(f"{IMG_PATH}/Wall/t_line11.png"),
        
        13: pygame.image.load(f"{IMG_PATH}/Wall/t_turn_1.png"),
        14: pygame.image.load(f"{IMG_PATH}/Wall/t_turn_2.png"),
        
    }

# ----------------------
# Pac-Man
# ----------------------
class PacMan_data:
    NUMBER = 0
    NAME = "PacMan"
    SIZE = 13

    IMAGES = [
        pygame.image.load(f"{IMG_PATH}/Pac-Man/0.png"),
        pygame.image.load(f"{IMG_PATH}/Pac-Man/1.png"),
        pygame.image.load(f"{IMG_PATH}/Pac-Man/2.png")
    ]
    ABS_SPEED = float(Grid_data.BLOCK_GAP * RATIO * 4)
    SPEED = np.array([ABS_SPEED, 0.0, 0.0])
    COORDINATE = np.array([Screen_data.WIDTH/2, Screen_data.HEIGHT/2, Z])
    DIRECTION = 0
    RATE = 5



class Ghost_data:
    BLINKY_COLOR = (237, 28, 36)
    CLYDE_COLOR = (222, 184, 70)
    LNKY_COLOR = (57, 222, 203)
    PINKY_COLOR = (255, 174, 201)
    
    ABS_SPEED = float(Grid_data.BLOCK_GAP * RATIO * 4)
    SPEED = np.array([ABS_SPEED, 0.0, 0.0])
    
    SIZE = 14

    EYE_IMAGES_ORDER = ["right.png", "up.png", "left.png", "down.png"]
    BODY_IMAGES_ORDER = ["1.png", "2.png"]

    COORDINATE = np.array([0, 0, Z])

    def __init__(self, color):
        self.COLOR = color
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
