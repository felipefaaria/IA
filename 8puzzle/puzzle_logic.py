import heapq
import math
from collections import deque
import random

# Estado final desejado
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic_cost = heuristic_cost
        self.f_cost = path_cost + heuristic_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def get_path(self):
        path = []
        node = self
        while node:
            path.append((node.action, node.state))
            node = node.parent
        return path[::-1]

def get_neighbors(state):
    neighbors = []
    zero_idx = state.index(0)
    row, col = divmod(zero_idx, 3)
    moves = {'Cima': (-1, 0), 'Baixo': (1, 0), 'Esquerda': (0, -1), 'Direita': (0, 1)}

    for action, (dr, dc) in moves.items():
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            neighbors.append((tuple(new_state), action))
    return neighbors

def is_solvable(state):
    inversions = 0
    state_list = [x for x in state if x != 0]
    for i in range(len(state_list)):
        for j in range(i + 1, len(state_list)):
            if state_list[i] > state_list[j]:
                inversions += 1
    return inversions % 2 == 0

# --- HEURÍSTICAS ---
def h1_misplaced(state):
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != GOAL_STATE[i])

def h2_manhattan(state):
    dist = 0
    for i in range(9):
        val = state[i]
        if val != 0:
            target_idx = GOAL_STATE.index(val)
            curr_row, curr_col = divmod(i, 3)
            target_row, target_col = divmod(target_idx, 3)
            dist += abs(curr_row - target_row) + abs(curr_col - target_col)
    return dist

def h3_euclidean(state):
    dist = 0
    for i in range(9):
        val = state[i]
        if val != 0:
            target_idx = GOAL_STATE.index(val)
            curr_row, curr_col = divmod(i, 3)
            target_row, target_col = divmod(target_idx, 3)
            dist += math.sqrt((curr_row - target_row)**2 + (curr_col - target_col)**2)
    return dist

# --- ALGORITMOS ---
def solve_bfs(start_state):
    start_node = Node(start_state)
    if start_state == GOAL_STATE: return start_node, 0
    frontier = deque([start_node])
    explored = {start_state}
    nodes_expanded = 0

    while frontier:
        node = frontier.popleft()
        nodes_expanded += 1
        for neighbor, action in get_neighbors(node.state):
            if neighbor not in explored:
                child = Node(neighbor, node, action, node.path_cost + 1)
                if neighbor == GOAL_STATE: return child, nodes_expanded
                frontier.append(child)
                explored.add(neighbor)
    return None, nodes_expanded

def solve_greedy(start_state, heuristic):
    start_node = Node(start_state, heuristic_cost=heuristic(start_state))
    frontier = []
    heapq.heappush(frontier, start_node)
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = heapq.heappop(frontier)
        nodes_expanded += 1
        if node.state == GOAL_STATE: return node, nodes_expanded
        explored.add(node.state)
        for neighbor, action in get_neighbors(node.state):
            if neighbor not in explored:
                h = heuristic(neighbor)
                child = Node(neighbor, node, action, heuristic_cost=h)
                child.f_cost = h
                if child.state not in [n.state for n in frontier]: # Simplificado
                    heapq.heappush(frontier, child)
    return None, nodes_expanded

def solve_astar(start_state, heuristic):
    start_node = Node(start_state, path_cost=0, heuristic_cost=heuristic(start_state))
    frontier = []
    heapq.heappush(frontier, start_node)
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = heapq.heappop(frontier)
        if node.state in explored: continue
        nodes_expanded += 1
        explored.add(node.state)
        if node.state == GOAL_STATE: return node, nodes_expanded
        for neighbor, action in get_neighbors(node.state):
            if neighbor not in explored:
                g = node.path_cost + 1
                h = heuristic(neighbor)
                child = Node(neighbor, node, action, g, h)
                heapq.heappush(frontier, child)
    return None, nodes_expanded