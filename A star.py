#!/usr/bin/python -tt
import pygame as pg
import math
import numpy as np
from queue import PriorityQueue
import time

WIDTH = 600 
WIN = pg.display.set_mode((WIDTH,WIDTH)) # Screen Dimension
pg.display.set_caption("A* Path Finding Algorithm") # Title
obstaclesSurface = pg.Surface((WIDTH, WIDTH))

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

t1=0
        
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
            u = dmax / d
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
    graph = RRT(start,end,width,height,obstacle)
    iteration = 0
    t1=time.time()
    while(not graph.path_to_goal()):
        elapsed=time.time()-t1
        t1=time.time()
        if elapsed > 10:
            raise
        if iteration % 2 == 0:
            x,y,Parent = graph.bias()
            pg.draw.circle(WIN,GREY,(x[-1],y[-1]),2,0)
            pg.draw.line(WIN,BLUE,(x[-1],y[-1]),(x[Parent[-1]],y[Parent[-1]]),2)
        
        else:
            x,y,Parent = graph.expand()
            pg.draw.circle(WIN,GREY,(x[-1],y[-1]),2,0)
            pg.draw.line(WIN,BLUE,(x[-1],y[-1]),(x[Parent[-1]],y[Parent[-1]]),2)
            
        if iteration % 5 == 0:
            pg.display.update()
        iteration +=1
    graph.drawPath(graph.getPathCoords())   
    pg.display.update()
    pg.event.clear()
    pg.event.wait()
    

def main(win, width):
    ROWS = 30
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
                    
                    A_star_clear (lambda: draw(win, grid, ROWS, width), grid, start, end)
                    
                if event.key == pg.K_r:                            
                    rrt_algorithm(start.rect,end.rect,width,width,rectang_list)
                  
    pg.quit()

main(WIN, WIDTH)