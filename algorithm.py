import math
import random
import numpy as np
from collections import deque
import pygame
from constant import Wall_data



def add_img(li, n, theta, coordinate, color=Wall_data.COLOR, wall_data=Wall_data.IMAGES):
    img = wall_data[n]
    img = pygame.transform.rotate(img, theta)
    img.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
    li.append((coordinate, theta, img))
    




def wall_make(map_data: list[list[int]], color=Wall_data.COLOR, wall_data=Wall_data.IMAGES):
    x_len = len(map_data[0])
    y_len = len(map_data)
    same = lambda i, j, dy, dx, t: 0 <= i + dy < y_len and 0 <= j + dx < x_len and map_data[i+dy][j+dx] in t
    data = []
    
    for i in range(y_len):
        for j in range(x_len):
            if map_data[i][j] == 1:
                
                v = [1, 2]
                N  = same(i, j, -1,  0, v)
                NE = same(i, j, -1,  1, v)
                E  = same(i, j,  0,  1, v)
                SE = same(i, j,  1,  1, v)
                S  = same(i, j,  1,  0, v)
                SW = same(i, j,  1, -1, v)
                W = same(i, j,  0, -1, v)
                NW = same(i, j, -1, -1, v)
                
                card = {"N": N, "E": E, "S": S, "W": W}
                diag = {"NE": NE, "SE": SE, "SW": SW, "NW": NW}

                card_count = sum(card.values())
                diag_count = sum(diag.values())
                
                if card_count == 1:                    
                    
                    if N:
                        if NE and NW:
                            add_img(data, "end11", 180, (i, j))
                        elif not NE and NW:
                            add_img(data, "end01", 180, (i, j))
                        elif NE and not NW:
                            add_img(data, "end10", 180, (i, j))
                        elif not NE and not NW:
                            add_img(data, "end00", 180, (i, j))
                    
                    elif E:
                        if NE and SE:
                            add_img(data, "end11", 90, (i, j))
                        elif not NE and SE:
                            add_img(data, "end10", 90, (i, j))
                        elif NE and not SE:
                            add_img(data, "end01", 90, (i, j))
                        elif not NE and not SE:
                            add_img(data, "end00", 90, (i, j))

                    elif S:
                        if SE and SW:
                            add_img(data, "end11", 0, (i, j))
                        elif not SE and SW:
                            add_img(data, "end10", 0, (i, j))
                        elif SE and not SW:
                            add_img(data, "end01", 0, (i, j))
                        elif not SE and not SW:
                            add_img(data, "end00", 0, (i, j))
                            
                    elif W:
                        if NW and SW:
                            add_img(data, "end11", 270, (i, j))
                        elif not NW and SW:
                            add_img(data, "end01", 270, (i, j))
                        elif NW and not SW:
                            add_img(data, "end10", 270, (i, j))
                        elif not NW and not SW:
                            add_img(data, "end00", 270, (i, j))
                    
                elif card_count == 2:
                    if N and E:
                        add_img(data, "turn", 0, (i, j))
                    elif E and S:
                        add_img(data, "turn", 270, (i, j))
                    elif S and W:
                        add_img(data, "turn", 180, (i, j))
                    elif W and N:
                        add_img(data, "turn", 90, (i, j))
                    else:
                        if N and S:
                            # right
                            if not NE and not SE:
                                add_img(data, "l00", 90, (i, j))
                            elif not NE and SE:
                                add_img(data, "l10", 90, (i, j))
                            elif NE and not SE:
                                add_img(data, "l01", 90, (i, j))
                            elif NE and SE:
                                add_img(data, "l11", 90, (i, j))

                            #left
                            if not NW and not SW:
                                add_img(data, "l00", 270, (i, j))
                            elif not NW and SW:
                                add_img(data, "l01", 270, (i, j))
                            elif NW and not SW:
                                add_img(data, "l10", 270, (i, j))
                            elif NW and SW:
                                add_img(data, "l11", 270, (i, j))
                                
                        elif E and W:
                            #up
                            if not NE and not NW:
                                add_img(data, "100", 180, (i, j))
                            elif NE and not NW:
                                add_img(data, "110", 180, (i, j))
                            elif not NE and NW:
                                add_img(data, "101", 180, (i, j))
                            elif NE and NW:
                                add_img(data, "111", 180, (i, j))

                            #down
                            if not SE and not SW:
                                add_img(data, "100", 0, (i, j))
                            elif SE and not SW:
                                add_img(data, "l01", 0, (i, j))
                            elif not SE and SW:
                                add_img(data, "110", 0, (i, j))
                            elif SE and SW:
                                add_img(data, "111", 0, (i, j))
                
                elif card_count == 3:
                    a, b = 0, 0
                    if not N:
                        if NE: a = 1
                        if NW: b = 1
                        add_img(data, f"l{a}{b}", 180, (i, j))
                    elif not E:
                        if NE: a = 1
                        if SE: b = 1
                        add_img(data, f"l{b}{a}", 90, (i, j))
                    elif not S:
                        if SE: a = 1
                        if SW: b = 1
                        add_img(data, f"l{b}{a}", 0, (i, j))
                    elif not W:
                        if NW: a = 1
                        if SW: b = 1
                        add_img(data, f"l{a}{b}", 270, (i, j))
                
            elif map_data[i][j] == 2:
                
                v = [2]
                N  = same(i, j, -1,  0, v)
                NE = same(i, j, -1,  1, v)
                E  = same(i, j,  0,  1, v)
                SE = same(i, j,  1,  1, v)
                S  = same(i, j,  1,  0, v)
                SW = same(i, j,  1, -1, v)
                W = same(i, j,  0, -1, v)
                NW = same(i, j, -1, -1, v)
                
                card = {"N": N, "E": E, "S": S, "W": W}
                diag = {"NE": NE, "SE": SE, "SW": SW, "NW": NW}

                card_count = sum(card.values())
                diag_count = sum(diag.values())
                
    
    return data
                
                

