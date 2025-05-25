import numpy as np
import matplotlib.pyplot as plt
import heapq
from matplotlib.colors import ListedColormap

WIDTH, HEIGHT = 20, 20

FREE = 0
OBSTACLE = 1
START = 2
GOAL = 3
PATH = 4

grid = np.zeros((HEIGHT, WIDTH), dtype=int)

start = (2, 3)
goal = (17, 15)

grid[start] = START
grid[goal] = GOAL

obstacles = (
    [(5, i) for i in range(5, 15)] +
    [(10, i) for i in range(3, 17)] +
    [(i, 12) for i in range(5, 10)]
)
for obs in obstacles:
    grid[obs] = OBSTACLE

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < HEIGHT and 0 <= neighbor[1] < WIDTH and
                grid[neighbor] != OBSTACLE):

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return None

path = astar(grid, start, goal)
if path:
    for cell in path:
        if grid[cell] == FREE:
            grid[cell] = PATH

def plot_grid(grid):
    colors = ['white', 'black', 'blue', 'red', 'orange']
    cmap = ListedColormap(colors)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap, origin='lower')
    plt.colorbar(ticks=range(5), label="Cell Type")
    plt.clim(-0.5, 4.5)
    plt.xticks(np.arange(WIDTH))
    plt.yticks(np.arange(HEIGHT))
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.title("A* Pathfinding on a Grid")
    plt.show()

plot_grid(grid)
