#!/usr/bin/python -tt
import pygame as pg
import math
import numpy as np
from queue import PriorityQueue
import time
from enum import Enum
from collections import deque
import random
import torch
from model import Linear_QNet,QTrainer
from helper import plot
import csv

# Window set up

WIDTH = 600
WIN = pg.display.set_mode((WIDTH,WIDTH)) # Screen Dimension
pg.display.set_caption("Path Planning") # Title
pg.time.Clock()

# Color code

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)
PURPLE = (128,0,128)
ORANGE = (255,165,0)
GREY = (128,128,128)
L_GREY = (200,200,200)
TURQUOISE = (64,224,208)

# Set up for machine learning
t1=0
MAX_MEMORY = 50_000
BATCH_SIZE = 512
LR = 0.0001
        
class Spot:
    def __init__(self,row,col,width,total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        
    def get_pos(self):
        return self.row, self.col
    
    def get_x(self):
        return (self.x)
    
    def get_y(self):
        return (self.y)
    
    def is_closed(self):
        return self.color == L_GREY
    
    def is_open(self):
        return self.color == GREEN
    
    def is_barrier(self):
        return self.color == BLACK
    
    def is_start(self):
        return self.color == ORANGE
    
    def is_end(self):
        return self.color == TURQUOISE
    
    def reset(self):
        self.color = WHITE
    
    def make_start(self):
        self.color = ORANGE
    
    def make_closed(self):
        self.color = L_GREY
        
    def make_open(self):
        self.color = GREEN
        
    def make_barrier(self):
        self.color = BLACK
        
    def make_end(self):
        self.color = TURQUOISE
        
    def make_path(self):
        self.color = RED
        
    def start_rect(self):
        rect = pg.Rect(self.x, self.y, self.width, self.width)
        return rect
        
    def end_rect(self):
        rect = pg.Rect(self.x, self.y, self.width, self.width)
        return rect
        
    def obs_rect(self):
        rect = pg.Rect(self.x, self.y, self.width, self.width)
        return rect
        
    def draw(self,win):
        self.rectangle_list = []
        self.rect = pg.Rect(self.x, self.y, self.width, self.width)
        self.rectangle_list.append(self.rect)
        pg.draw.rect(win, self.color, self.rect)
        
    def update_neighbors(self, grid):
        self.neighbors = []
        
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # Down
            self.neighbors.append(grid[self.row + 1][self.col])
                        
        if self.row < self.total_rows - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_barrier(): # Down left
            self.neighbors.append(grid[self.row + 1][self.col - 1])
                        
        if self.col < self.total_rows - 1 and self.row < self.total_rows - 1 and not grid[self.row + 1][self.col + 1].is_barrier(): # Down Right
            self.neighbors.append(grid[self.row + 1][self.col + 1])
            
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # Up
            self.neighbors.append(grid[self.row - 1][self.col])
                        
        if self.col > 0 and self.row > 0 and not grid[self.row - 1][self.col - 1].is_barrier(): # Up Left
            self.neighbors.append(grid[self.row - 1][self.col - 1])
                        
        if self.row > 0 and self.col < self.total_rows - 1 and not grid[self.row - 1][self.col + 1].is_barrier(): # Up Right
            self.neighbors.append(grid[self.row - 1][self.col + 1])
            
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # Left
            self.neighbors.append(grid[self.row][self.col - 1])
            
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # Right
            self.neighbors.append(grid[self.row ][self.col + 1])
            
    def update_barrier(self, grid):
        self.barriers = []
        
        if grid[self.row][self.col].is_barrier():
            self.barriers.append(grid[self.row][self.col])
        
        return self.barriers
            
    def __lt__(self, other):
        return False
    
def h(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs (y1 - y2)

def A_star_algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}
    
    while not open_set.empty():
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                
        current = open_set.get()[2]
        open_set_hash.remove(current)
        
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
                    
        draw()
        
        if current != start:
            current.make_closed()     
    return False

def A_star_clear(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}
    
    while not open_set.empty():
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                
        current = open_set.get()[2]
        open_set_hash.remove(current)
        
        if current == end:
            clear_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.reset()
                    
        draw()
        
        if current != start:
            current.reset()
            
    return False
                
    
def make_grid(rows,width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Spot(i, j, gap, rows)
            grid[i].append(node)
            
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pg.draw.line(win, GREY, (0,i * gap), (width, i * gap))
        for j in range(rows):
            pg.draw.line(win, GREY, (j * gap, 0), (j * gap, width))
            
def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		draw()
        
def clear_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.reset()
        draw()
            
def draw(win, grid, rows, width):
    win.fill((255,0,0))
    
    for row in grid:
        for node in row:
            node.draw(win)
            
    draw_grid(win, rows, width)
    pg.display.update()
    
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    x, y = pos
    
    row = x // gap
    col = y // gap
    
    return row, col

class RRT:
    def __init__(self,start,end, width, height, rectang):
        self.start = start
        self.end = end
        self.start_x,self.start_y = start.center
        self.goal_x,self.goal_y = end.center
        self.width = width
        self.height = height
        self.x = []
        self.y = []
        self.parent = []
        self.rectang_list = rectang
                
        # initialize the tree
        self.x.append(self.start_x)
        self.y.append(self.start_y)
        self.parent.append(0)

        # path
        self.goalFlag = False
        self.goalstate = None
        self.path = []
        
        
    def add_node(self,n,x,y):
        self.x.insert(n,x)
        self.y.append(y)
    
    def remove_node(self,n):
        self.x.pop(n)
        self.y.pop(n)
        
    def number_of_node(self):
        return len(self.x)
    
    def sample_envir(self):
        x = int(np.random.uniform(0,self.width))
        y = int(np.random.uniform(0,self.height))
        return x,y

    def isFree(self,n,x,y):
        for rectang in self.rectang_list:
            if rectang.collidepoint(x, y):
                self.remove_node(n)
                return False
        return True
    
    def add_edge(self, parent, child):
        self.parent.insert(child, parent)

    def remove_edge(self, n):
        self.parent.pop(n)
        
    def distance(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        px = (float(x1) - float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2
        return (px + py) ** (0.5)
    
    def nearest(self, n):
        dmin = self.distance(0, n)
        nnear = 0
        for i in range(0, n):
            if self.distance(i, n) < dmin:
                dmin = self.distance(i, n)
                nnear = i
        return nnear
    
    def crossObstacle(self, x1, x2, y1, y2):
        for rectang in self.rectang_list:
            for i in range(0, 101):
                u = i / 100
                x = x1 * u + x2 * (1 - u)
                y = y1 * u + y2 * (1 - u)
                if rectang.collidepoint(x, y):
                    return True
        return False
    
    def connect(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        if self.crossObstacle(x1, x2, y1, y2):
            self.remove_node(n2)
            return False
        else:
            self.add_edge(n1, n2)
            return True
        
    def bias(self):
        n = self.number_of_node()
        self.add_node(n,self.goal_x, self.goal_y)
        nnear = self.nearest(n)
        self.step(nnear, n)
        self.connect(nnear, n)
        return self.x, self.y, self.parent

    def expand(self):
        n = self.number_of_node()
        x, y = self.sample_envir()
        self.add_node(n, x, y)
        if self.isFree(n,x,y):
            xnearest = self.nearest(n)
            self.step(xnearest, n)
            self.connect(xnearest, n)
        return self.x, self.y, self.parent

    def path_to_goal(self):
        if self.goalFlag:
            self.path = []
            self.path.append(self.goalstate)
            newpos = self.parent[self.goalstate]
            while (newpos != 0):
                self.path.append(newpos)
                newpos = self.parent[newpos]
            self.path.append(0)
        return self.goalFlag

    def getPathCoords(self):
        pathCoords = []
        for node in self.path:
            x, y = (self.x[node], self.y[node])
            pathCoords.append((x, y))
        return pathCoords

    def cost(self, n):
        ninit = 0
        n = n
        parent = self.parent[n]
        c = 0
        while n is not ninit:
            c = c + self.distance(n, parent)
            n = parent
            if n is not ninit:
                parent = self.parent[n]
        return c

    def getTrueObs(self, obs):
        TOBS = []
        for ob in obs:
            TOBS.append(ob.inflate(-50, -50))
        return TOBS

    def waypoints2path(self):
        oldpath = self.getPathCoords()
        path = []
        for i in range(0, len(self.path) - 1):
            print(i)
            if i >= len(self.path):
                break
            x1, y1 = oldpath[i]
            x2, y2 = oldpath[i + 1]
            print('---------')
            print((x1, y1), (x2, y2))
            for i in range(0, 5):
                u = i / 5
                x = int(x2 * u + x1 * (1 - u))
                y = int(y2 * u + y1 * (1 - u))
                path.append((x, y))
                print((x, y))

        return path
    
    def step(self, nnear, nrand, dmax= 20, dgoal = 15):
        d = self.distance(nnear, nrand)
        if d > dmax:
            (xnear, ynear) = (self.x[nnear], self.y[nnear])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py, px)
            (x, y) = (int(xnear + dmax * math.cos(theta)),
                      int(ynear + dmax * math.sin(theta)))
            self.remove_node(nrand)
            if abs(x - self.goal_x) <= dgoal and abs(y - self.goal_y) <= dgoal:
                self.add_node(nrand, self.goal_x, self.goal_y)
                self.goalstate = nrand
                self.goalFlag = True
            else:
                self.add_node(nrand, x, y)
                
    def drawPath(self,path):
        for node in path:
            pg.draw.circle(WIN,RED,node,3,0)
    
def rrt_algorithm(start,end,width,height,obstacle):
    start_time = time.time()
    graph = RRT(start,end,width,height,obstacle)
    iteration = 0
    t1=time.time()
    clock = pg.time.Clock()
    while(not graph.path_to_goal()):
        pg.event.pump()
        elapsed=time.time()-t1
        t1=time.time()
        if elapsed > 10:
            raise
        if iteration % 10 == 0:
            x,y,Parent = graph.bias()
            pg.draw.circle(WIN,GREY,(x[-1],y[-1]),2,0)
            pg.draw.line(WIN,BLUE,(x[-1],y[-1]),(x[Parent[-1]],y[Parent[-1]]),2)
        
        else:
            x,y,Parent = graph.expand()
            pg.draw.circle(WIN,GREY,(x[-1],y[-1]),2,0)
            pg.draw.line(WIN,BLUE,(x[-1],y[-1]),(x[Parent[-1]],y[Parent[-1]]),2)
            
        if iteration % 5 == 0:
            pg.display.update()
            clock.tick(10)
        iteration +=1
    graph.drawPath(graph.getPathCoords())  
    pg.display.update()
    end_time = time.time()
    print('Computational time: ',end_time - start_time)
    pg.event.clear()
    pg.event.wait()
    
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    HOLD = 5
    
class Game:
    def __init__(self,x,y,win,obstacle_list,width,goal_x,goal_y,rows,start,end,row):
        self.width = width
        self.height = width
        self.display = win
        self.rows = rows
        self.start_x = x
        self.start_y = y
        self.win = win
        self.start = start
        self.end = end
        self.obstacle_list = obstacle_list
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.speed = 10000
        self.blocksize = width/self.rows
        self.reset()
    
    def reset(self):
        self.clock = pg.time.Clock()
        self.direction = Direction.HOLD
        self.x = self.start_x
        self.y = self.start_y
        self.player = pg.Rect(self.x,self.y,self.blocksize,self.blocksize)
        self.goal = pg.Rect(self.goal_x,self.goal_y,self.blocksize,self.blocksize)
        self.frame_iteration = 0
        self.reward = 0
        self.randomobslist = []

        
    def add_uncertainty(self):
        self.randomobs1 = self.uncertainty()
        self.randomobs2 = self.uncertainty()
        self.randomobs3 = self.uncertainty()
        self.randomobs4 = self.uncertainty()
        self.randomobs5 = self.uncertainty()
        self.randomobslist.append(self.randomobs1)
        self.randomobslist.append(self.randomobs2)
        self.randomobslist.append(self.randomobs3)
        self.randomobslist.append(self.randomobs4)
        self.randomobslist.append(self.randomobs5)
        
    def is_add(self):
        if self.randomobslist == []:
            return False
        return True
        
    def uncertainty(self):
        x = random.randint(0,self.rows)
        y = random.randint(0,self.rows)
        multiplier = self.blocksize
        randomobs = pg.Rect(x*multiplier,y*multiplier,self.blocksize,self.blocksize)
        for obstacle in self.obstacle_list:
            if pg.Rect.colliderect(randomobs, obstacle) or pg.Rect.colliderect(randomobs, self.player) or pg.Rect.colliderect(randomobs, self.goal):
                randomobs = pg.Rect(-self.blocksize,-self.blocksize,self.blocksize,self.blocksize)
        return randomobs
    
            
    def play_step(self, action):
        self.frame_iteration +=1
        self.p_player = pg.Rect(self.x,self.y,self.blocksize,self.blocksize)
        # move
        self._move(action) # update the head
         
        # check if game over
        game_over = False
        if self._is_collision(self.player):
            game_over = True
            self.reward += -10
            return self.reward, game_over
        
        if self._out_of_bound(self.player):
            game_over = True
            self.reward += -10
            return self.reward, game_over
        
        
        if self.frame_iteration > 675:
            game_over = True
            self.reward += -10
            return self.reward, game_over
        
        if self._reach_goal():
            game_over = True
            self.reward += 20
            return self.reward, game_over
        

        # update ui and clock
        self.update()
        self.clock.tick(self.speed)
        # 6. return game over
        return self.reward, game_over
        
    def update(self):
        
        self.display.fill(WHITE)
        draw_grid(self.display, self.rows, self.width)
        for obstacle in self.obstacle_list:
            pg.draw.rect(self.display,BLACK,obstacle)
        pg.draw.rect(self.display,ORANGE,self.start)
        pg.draw.rect(self.display,TURQUOISE,self.end)
        pg.draw.rect(self.display,BLUE,self.player)
        if self.is_add():
            pg.draw.rect(self.display,RED,self.randomobs1)
            pg.draw.rect(self.display,RED,self.randomobs2)
            pg.draw.rect(self.display,RED,self.randomobs3)
            pg.draw.rect(self.display,RED,self.randomobs4)
            pg.draw.rect(self.display,RED,self.randomobs5)
        pg.display.update()
        
    def _is_collision(self,rect):
        # hit obstacle
        for obstacle in self.obstacle_list:
            if pg.Rect.colliderect(rect, obstacle):
                return True
        for randomobs in self.randomobslist:
            if pg.Rect.colliderect(rect, randomobs):
                return True
        return  False
    
    def _out_of_bound(self,rect):
        # out of bound
        if rect.x > (self.width - self.blocksize) or rect.x < 0 or rect.y > (self.height - self.blocksize) or rect.y < 0:
            return True
        return False
    
    def _reach_goal(self):
        # reach goal
        if pg.Rect.collidepoint(self.player, (self.goal_x, self.goal_y)):
            return True
        return False
    
    def _move(self, action):
        
        new_dir = None
        
        if np.array_equal(action,[1,0,0,0]):
            new_dir = Direction.UP #UP
            
        elif np.array_equal(action,[0,1,0,0]):
            new_dir = Direction.RIGHT
            
        elif np.array_equal(action,[0,0,1,0]):
            new_dir = Direction.DOWN
            
        elif np.array_equal(action,[0,0,0,1]):
            new_dir = Direction.LEFT
            
        elif np.array_equal(action,[0,0,0,0]):
            new_dir = Direction.HOLD
            
        self.direction = new_dir

        if self.direction == Direction.RIGHT:
            self.x += self.blocksize
        elif self.direction == Direction.LEFT:
            self.x -= self.blocksize
        elif self.direction == Direction.DOWN:
            self.y += self.blocksize
        elif self.direction == Direction.UP:
            self.y -= self.blocksize
        else:
            self.x += 0
            self.y += 0
            
        self.player = pg.Rect(self.x,self.y,self.blocksize,self.blocksize)
        
class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 1 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(12, 96, 4)
        self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)
        self.blocksize = 30


    def get_state(self, game):
        player = game.player
        
        left_rect = pg.Rect(player.x - self.blocksize, player.y,self.blocksize,self.blocksize)
        right_rect = pg.Rect(player.x + self.blocksize, player.y,self.blocksize,self.blocksize)
        up_rect = pg.Rect(player.x, player.y - self.blocksize,self.blocksize,self.blocksize)
        down_rect = pg.Rect(player.x, player.y + self.blocksize,self.blocksize,self.blocksize)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        
        state = [

            # Danger up
            (game._is_collision(up_rect)) or (game._out_of_bound(up_rect)),
            
            
            # Danger right
            (game._is_collision(right_rect)) or (game._out_of_bound(right_rect)),
            
            
            # Danger down
            (game._is_collision(down_rect)) or (game._out_of_bound(down_rect)),
            
            
            # Danger left
            (game._is_collision(left_rect)) or (game._out_of_bound(left_rect)),

            # Move direction
            dir_u,
            dir_r,
            dir_d,
            dir_l,
            
            # goal location 
            game.goal_x < game.x,  # goal left
            game.goal_x > game.x,  # goal right
            game.goal_y < game.y,  # goal up
            game.goal_y > game.y   # goal down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        EPS_START = 1
        EPS_END = 0.05
        EPS_DECAY = 200
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.n_games / EPS_DECAY)
        self.sample = random.random()
        final_move = [0,0,0,0]
        if self.sample <= self.epsilon:
            
            if state[4] == 0 and state[5] == 0 and state[6] == 0 and state[7] == 0: # No heading
            
                # 0 barrier
                if state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(0, 3)
                    final_move[move] = 1
                    
                # 1 barrier    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(1, 3)
                    final_move[move] = 1
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = random.randint(1, 3)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [1,0,0,0]
                    
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                    if final_move == [0,0,1,0]:
                        final_move = [0,0,0,1]
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                    
                # 2 barriers    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = random.randint(2, 3)
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                    if final_move == [0,0,1,0]:
                        final_move = [0,0,0,1]
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                        
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,0,1]
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,1,0]
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    
                # 3 barriers
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 1:
                    move = 0
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = 1
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = 2
                    final_move[move] = 1
    
                        
                elif state[0] == 1 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = 3
                    final_move[move] = 1
            
            if state[4] == 1 and state[5] == 0 and state[6] == 0 and state[7] == 0: # Heading up
            
                # 0 barrier
                if state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                    if final_move == [0,0,1,0]:
                        final_move = [0,0,0,1]
                    
                # 1 barrier    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                    if final_move == [0,0,1,0]:
                        final_move = [0,0,0,1]
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,0,1]
                    
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                    if final_move == [0,0,1,0]:
                        final_move = [0,0,0,1]
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    
                # 2 barriers    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = 3
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                    if final_move == [0,0,1,0]:
                        final_move = [0,0,0,1]
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = 1
                    final_move[move] = 1
                        
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,0,1]
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = 0
                    final_move[move] = 1
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    
                # 3 barriers
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 1:
                    move = 0
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = 1
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                    if final_move == [0,0,1,0]:
                        final_move = [0,0,0,1]
    
                elif state[0] == 1 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = 3
                    final_move[move] = 1
                    
            if state[4] == 0 and state[5] == 1 and state[6] == 0 and state[7] == 0: # Heading right
            
                # 0 barrier
                if state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                    
                # 1 barrier    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,1,0]
                    
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                    
                # 2 barriers    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = 2
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = 1
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                        
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = 0
                    final_move[move] = 1
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,1,0]
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    
                # 3 barriers
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 1:
                    move = 0
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = 1
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = 2
                    final_move[move] = 1
    
                        
                elif state[0] == 1 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                    
            if state[4] == 0 and state[5] == 0 and state[6] == 1 and state[7] == 0: # Heading down
            
                # 0 barrier
                if state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(1, 3)
                    final_move[move] = 1
                    
                # 1 barrier    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(1, 3)
                    final_move[move] = 1
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = random.randint(2, 3)
                    final_move[move] = 1
                    
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                    if final_move == [0,0,1,0]:
                        final_move = [0,0,0,1]
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                    
                # 2 barriers    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = random.randint(2, 3)
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = 3
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = random.randint(1, 2)
                    final_move[move] = 1
                        
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = 3
                    final_move[move] = 1
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = 2
                    final_move[move] = 1
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = 1
                    final_move[move] = 1
                    
                # 3 barriers
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 1:
                    move = random.randint(1, 3)
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = 1
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = 2
                    final_move[move] = 1
    
                        
                elif state[0] == 1 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = 3
                    final_move[move] = 1
                    
            if state[4] == 0 and state[5] == 0 and state[6] == 0 and state[7] == 1: # Heading left
            
                # 0 barrier
                if state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(1, 3)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [1,0,0,0]
                    
                # 1 barrier    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 0:
                    move = random.randint(2, 3)
                    final_move[move] = 1
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = random.randint(1, 3)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [1,0,0,0]
                    
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,0,1]
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,1,0]
                    
                # 2 barriers    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 0:
                    move = random.randint(2, 3)
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 0:
                    move = 3
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 0 and state[3] == 1:
                    move = 2
                    final_move[move] = 1
                        
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,0,1]
                    
                elif state[0] == 0 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = random.randint(0, 1)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [0,0,1,0]
                        
                elif state[0] == 0 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = 0
                    final_move[move] = 1
                    
                # 3 barriers
                elif state[0] == 0 and state[1] == 1 and state[2] == 1 and state[3] == 1:
                    move = 0
                    final_move[move] = 1
                    
                elif state[0] == 1 and state[1] == 0 and state[2] == 1 and state[3] == 1:
                    move = random.randint(1, 3)
                    final_move[move] = 1
                    if final_move == [0,1,0,0]:
                        final_move = [1,0,0,0]
                    
                elif state[0] == 1 and state[1] == 1 and state[2] == 0 and state[3] == 1:
                    move = 2
                    final_move[move] = 1
    
                        
                elif state[0] == 1 and state[1] == 1 and state[2] == 1 and state[3] == 0:
                    move = 3
                    final_move[move] = 1
            
            
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def csv_save(episode_list,plot_reward):
    fields = []
    move_allepisode = []
    a = 'RUN'
    b = 'Reward'
    fields.append(a)
    fields.append(b)
    for i in range (499):
        b = str(i+1)
        c = 'STEP ' + b
        fields.append(c)
    h = 0
    for episode in episode_list:
        move = []
        h += 1
        move.append(h)
        move.append(plot_reward[h-1])
        for action in episode:
            if action == [1,0,0,0]:
                direction = 'UP'
            if action == [0,1,0,0]:
                direction = 'RIGHT'
            if action == [0,0,1,0]:
                direction = 'DOWN'
            if action == [0,0,0,1]:
                direction = 'LEFT'
            if action == [0,0,0,0]:
                direction = 'STOP'
            move.append(direction)
        move_allepisode.append(move)
    
    rows = move_allepisode
    
    with open('Move','w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)
        
    

def train(x,y,win,obstacle_list,width,goal_x,goal_y,rows,start,end):
    start_time=time.time()
    plot_reward = []
    plot_mean_reward = []
    total_reward = 0
    agent = Agent()
    game = Game(x,y,win,obstacle_list,width,goal_x,goal_y,rows,start,end,rows)
    action_list = []
    episode_list = []
    
    while agent.n_games<=2500:
        pg.event.pump()
        # get old state
        state_old = agent.get_state(game)
        

        # get move
        final_move = agent.get_action(state_old)
        action_list.append(final_move)
        

        # perform move and get new state
        reward, done = game.play_step(final_move)
        state_new = agent.get_state(game)



        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
                    
        if done:
            # train long memory, plot result
            game.reset()
            if agent.n_games > 2500:
                game.add_uncertainty()
            agent.n_games += 1
            agent.train_long_memory()

            print('Game', agent.n_games,'Reward', reward)
            print('Epsilon :', agent.epsilon)
            episode_list.append(action_list)
            action_list = []
            plot_reward.append(reward)
            total_reward += reward
            mean_score = total_reward / agent.n_games
            plot_mean_reward.append(mean_score)
            plot(plot_reward, plot_mean_reward)
    end_time = time.time()   
     
    print('End')
    print('Computational time: ', end_time - start_time)
    csv_save(episode_list,plot_reward)
    
    
def q_algorithm(x,y,win,rectang_list,width,goal_x,goal_y,rows,start,end):
    train(x,y,win,rectang_list,width,goal_x,goal_y,rows,start,end)

    
def main(win, width):
    ROWS = 15
    grid = make_grid(ROWS,width)
    
    start = None
    end = None
    
    run = True
    
    rectang_list = []

    while run:
        draw(win, grid, ROWS, width)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

            
            if pg.mouse.get_pressed()[0]: # LEFT
                pos = pg.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                if not start and node != end:
                    start = node
                    start.make_start()
                    print(start.get_x(),start.get_y())
                    
                elif not end and node != start:
                    end = node
                    end.make_end()
                    
                elif node != end and node != start:
                    obstacles = node
                    rectang_list.append(obstacles.obs_rect())
                    obstacles.make_barrier()
                    
            elif pg.mouse.get_pressed()[2]: # RIGHT
                pos = pg.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None
                    
            elif pg.mouse.get_pressed()[1]: # middle
                pos = pg.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                path = node
                path.make_open()
            
            
            if event.type == pg.KEYDOWN:
                
                if event.key == pg.K_a and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                            
                    A_star_algorithm (lambda: draw(win, grid, ROWS, width), grid, start, end)
                    
                if event.key == pg.K_c:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    start_time = time.time()
                    A_star_clear (lambda: draw(win, grid, ROWS, width), grid, start, end)
                    end_time = time.time()
                    print('Computational time :', end_time - start_time)
                    
                if event.key == pg.K_r:                            
                    rrt_algorithm(start.rect,end.rect,width,width,rectang_list)
                    
                if event.key == pg.K_t:
                    q_algorithm(start.get_x(),start.get_y(),win,rectang_list,width,end.x,end.y,ROWS,start.rect,end.rect)

                    
                    
                
                  
    pg.quit()

main(WIN, WIDTH)