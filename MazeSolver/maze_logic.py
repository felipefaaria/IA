import heapq
import math
from collections import deque

class Node:
    def __init__(self, position, parent=None, action=None, path_cost=0, heuristic_cost=0):
        self.position = position  # Tupla (row, col)
        self.parent = parent
        self.path_cost = path_cost # g(n)
        self.heuristic_cost = heuristic_cost # h(n)
        self.f_cost = path_cost + heuristic_cost # f(n)

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def get_path(self):
        path = []
        node = self
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

def get_neighbors(position, grid, rows, cols):
    """Retorna vizinhos válidos (não são paredes e estão dentro do grid)"""
    r, c = position
    neighbors = []
    # Cima, Baixo, Esquerda, Direita (Sem diagonal para facilitar comparação com Manhattan)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        # Verifica limites e se não é parede (1 = parede)
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            neighbors.append((nr, nc))
    return neighbors

# --- HEURÍSTICAS ---
def h1_manhattan(pos, goal):
    """Distância de Manhattan: |x1-x2| + |y1-y2|"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def h2_euclidean(pos, goal):
    """Distância Euclidiana: hipotenusa"""
    return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

def h3_weighted_manhattan(pos, goal):
    """Manhattan ponderada (não admissível, mas muito rápida). Multiplica por 2."""
    return (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])) * 2.0

# --- ALGORITMOS ---

def solve_bfs(grid, start, goal):
    """Busca em Largura"""
    rows, cols = len(grid), len(grid[0])
    start_node = Node(start)
    if start == goal: return start_node, 0, []

    frontier = deque([start_node])
    explored = {start}
    nodes_expanded = 0
    visited_order = [] # Para animação da busca (opcional)

    while frontier:
        node = frontier.popleft()
        nodes_expanded += 1
        visited_order.append(node.position)

        if node.position == goal:
            return node, nodes_expanded, visited_order

        for neighbor_pos in get_neighbors(node.position, grid, rows, cols):
            if neighbor_pos not in explored:
                explored.add(neighbor_pos)
                child = Node(neighbor_pos, node, path_cost=node.path_cost + 1)
                frontier.append(child)
    
    return None, nodes_expanded, visited_order

def solve_greedy(grid, start, goal, heuristic_func):
    """Busca Gulosa"""
    rows, cols = len(grid), len(grid[0])
    start_node = Node(start, heuristic_cost=heuristic_func(start, goal))
    frontier = []
    heapq.heappush(frontier, start_node)
    explored = {start}
    nodes_expanded = 0
    visited_order = []

    while frontier:
        node = heapq.heappop(frontier)
        nodes_expanded += 1
        visited_order.append(node.position)

        if node.position == goal:
            return node, nodes_expanded, visited_order

        for neighbor_pos in get_neighbors(node.position, grid, rows, cols):
            if neighbor_pos not in explored:
                explored.add(neighbor_pos)
                h = heuristic_func(neighbor_pos, goal)
                # Gulosa usa apenas h(n) como custo
                child = Node(neighbor_pos, node, heuristic_cost=h) 
                child.f_cost = h 
                heapq.heappush(frontier, child)

    return None, nodes_expanded, visited_order

def solve_astar(grid, start, goal, heuristic_func):
    """Algoritmo A*"""
    rows, cols = len(grid), len(grid[0])
    # f(n) = g(n) + h(n)
    start_node = Node(start, path_cost=0, heuristic_cost=heuristic_func(start, goal))
    
    frontier = []
    heapq.heappush(frontier, start_node)
    
    # Dicionário para guardar o menor custo g(n) encontrado para cada célula
    g_costs = {start: 0}
    nodes_expanded = 0
    visited_order = []

    while frontier:
        node = heapq.heappop(frontier)
        
        # Se achamos um caminho melhor para este nó já, ignoramos este antigo da fila
        if node.path_cost > g_costs.get(node.position, float('inf')):
            continue

        nodes_expanded += 1
        visited_order.append(node.position)

        if node.position == goal:
            return node, nodes_expanded, visited_order

        for neighbor_pos in get_neighbors(node.position, grid, rows, cols):
            new_g = node.path_cost + 1
            
            # Se achamos um caminho melhor para o vizinho (ou é a primeira vez)
            if new_g < g_costs.get(neighbor_pos, float('inf')):
                g_costs[neighbor_pos] = new_g
                h = heuristic_func(neighbor_pos, goal)
                child = Node(neighbor_pos, node, path_cost=new_g, heuristic_cost=h)
                heapq.heappush(frontier, child)

    return None, nodes_expanded, visited_order