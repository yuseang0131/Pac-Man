import math
import numpy as np
from collections import deque

class Unit:
    def move_to(self, current_coordinate, target_coordinate):
        pass

    def targeting(self, coordinate: np.array):
        pass


# red
class Blinky(Unit):
    pass

# pink
class Pinky(Unit):
    pass

# cyan
class Lnky(Unit):
    pass

# orange
class Clyde(Unit):
    pass


def make_wall(data):
    len_x = data["map"]["size_x"]
    len_y = data["map"]["size_y"]
    map_data = data["map"]["map"]
    
    walls = []
    
    visited = [[[0, 0] for _ in range(len_x)] for _ in range(len_y)]
    for i in range(len_y):
        for j in range(len_x):
            if map_data[i][j] == 0:
                continue
            
            wall = []
            queue = deque()
            queue.append((j, i))
            
            while queue:
                x, y = queue.popleft()
                
                if not visited[y][x][0]: #row
                    visited[y][x][0] = 1
                    end = [(x, y), (x, y)]
                    
                    last = None
                    w = 1
                    while 0 <= x+w < len_x and not visited[y][x+w][0] and map_data[y][x+w] == 1:
                        visited[y][x+w][0] = 1
                        queue.append((x+w, y))
                        last = (x+w, y)
                        w += 1
                    if queue and last: end[0] = last
                    
                    last = None
                    w = -1
                    while 0 <= x+w < len_x and not visited[y][x+w][0] and map_data[y][x+w] == 1:
                        visited[y][x+w][0] = 1
                        queue.append((x+w, y))
                        last = (x+w, y)
                        w -= 1
                    if queue and last: end[1] = last
                    
                    if end[0][0] != end[1][0] or end[0][1] != end[1][1]:
                        wall.append(end)
                    
                
                if not visited[y][x][1]: #column
                    visited[y][x][1] = 1
                    end = [(x, y), (x, y)]
                    
                    last = None
                    w = 1
                    while 0 <= y+w < len_y and not visited[y+w][x][1] and map_data[y+w][x] == 1:
                        visited[y+w][x][1] = 1
                        queue.append((x, y+w))
                        last = (x, y+w)
                        w += 1
                    if queue and last: end[0] = last
                    
                    last = None
                    w = -1
                    while 0 <= y+w < len_y and not visited[y+w][x][1] and map_data[y+w][x] == 1:
                        visited[y+w][x][1] = 1
                        queue.append((x, y+w))
                        last = (x, y+w)
                        w -= 1
                    if queue and last: end[1] = last
                    
                    if end[0][0] != end[1][0] or end[0][1] != end[1][1]:
                        wall.append(end)
                
            if wall:
                walls.append(wall)
                        
    
    return walls