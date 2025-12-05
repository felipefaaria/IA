import pygame
import sys
import time
import maze_logic as logic

# --- CONFIGURAÇÕES VISUAIS ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
GRID_ROWS, GRID_COLS = 20, 30
CELL_SIZE = 30
MARGIN_LEFT = 250 # Espaço para o painel lateral
MARGIN_TOP = 50

# Cores
COLOR_BG = (30, 33, 40)
COLOR_GRID_BG = (255, 255, 255)
COLOR_WALL = (50, 50, 60)
COLOR_START = (46, 204, 113) # Verde
COLOR_GOAL = (231, 76, 60)   # Vermelho
COLOR_PATH = (52, 152, 219)  # Azul
COLOR_VISITED = (174, 214, 241) # Azul claro (nós explorados)
COLOR_LINE = (200, 200, 200)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (100, 149, 237)
COLOR_RESET_BTN = (220, 53, 69)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Labirinto IA Solver - A*, BFS, Gulosa")
font_ui = pygame.font.SysFont("arial", 18)
font_title = pygame.font.SysFont("arial", 24, bold=True)
font_stats = pygame.font.SysFont("arial", 16)

class Button:
    def __init__(self, x, y, w, h, text, callback, color=COLOR_BUTTON):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.base_color = color
        self.hovered = False

    def draw(self, surface):
        color = self.base_color
        if self.hovered:
            color = (min(color[0]+30, 255), min(color[1]+30, 255), min(color[2]+30, 255))
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        text_surf = font_ui.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def check_click(self, mouse_pos):
        if self.hovered and self.callback:
            self.callback()

class MazeApp:
    def __init__(self):
        # 0=Vazio, 1=Parede
        self.grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.start_pos = (2, 2)
        self.goal_pos = (GRID_ROWS-3, GRID_COLS-3)
        self.path = []
        self.visited_cells = [] # Para animação da exploração
        self.is_running = False
        self.is_animating = False
        self.animation_idx = 0
        self.stats_msg = ""
        self.status_msg = ""
        self.drag_mode = None # 'start', 'goal', 'wall'

    def clear_path(self):
        self.path = []
        self.visited_cells = []
        self.is_animating = False
        self.stats_msg = ""
        self.status_msg = "Caminho limpo."

    def reset_grid(self):
        self.grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.clear_path()
        self.status_msg = "Grid resetado."
    def create_trap_maze(self):
        """Cria um labirinto projetado para diferenciar os algoritmos"""
        self.reset_grid()
        self.start_pos = (2, 2)
        self.goal_pos = (10, 26)
        
        # 1. A Grande Barreira Central (Força a dar a volta)
        # Bloqueia a linha reta entre inicio e fim
        for r in range(0, 16):
            self.grid[r][15] = 1 
            
        # 2. O "U" da Morte (Armadilha para a Gulosa)
        # Uma parede em forma de C ao redor do objetivo
        # A Gulosa vai tentar entrar no U e ficar presa
        for r in range(5, 15):
            self.grid[r][22] = 1 # Parede esquerda do U
        for c in range(22, 28):
            self.grid[5][c] = 1  # Parede superior do U
            self.grid[14][c] = 1 # Parede inferior do U
            
        # 3. Beco sem saída enganoso
        # Um corredor que parece levar ao objetivo mas fecha
        for c in range(5, 12):
            self.grid[10][c] = 1
            self.grid[12][c] = 1
        self.grid[11][11] = 1 # Fecha a ponta
            
        self.status_msg = "Labirinto de Teste Gerado!"
    def run_algorithm(self, algo_name, heuristic_name=None):
        self.clear_path()
        self.status_msg = f"Calculando {algo_name}..."
        draw_app(self, []) 
        pygame.display.flip()

        # ALTERAÇÃO 1: Usar perf_counter para alta precisão
        start_time = time.perf_counter()
        
        node = None
        expanded = 0
        visited_order = []

        if algo_name == "BFS":
            node, expanded, visited_order = logic.solve_bfs(self.grid, self.start_pos, self.goal_pos)
        elif algo_name == "Gulosa":
            h = logic.h1_manhattan 
            node, expanded, visited_order = logic.solve_greedy(self.grid, self.start_pos, self.goal_pos, h)
        elif algo_name == "A*":
            h = getattr(logic, heuristic_name)
            node, expanded, visited_order = logic.solve_astar(self.grid, self.start_pos, self.goal_pos, h)

        end_time = time.perf_counter()
        
        if node:
            self.path = node.get_path()
            # ALTERAÇÃO 2: Converter para Milissegundos (ms)
            duration_ms = (end_time - start_time) * 1000 
            
            # ALTERAÇÃO 3: Mostrar em ms com mais precisão
            self.stats_msg = f"[{algo_name}] Nós: {expanded} | Tempo: {duration_ms:.4f}ms | Passos: {len(self.path)}"
            
            self.visited_cells = visited_order
            self.status_msg = "Solução encontrada! Animando..."
            self.is_animating = True
        else:
            self.status_msg = "Sem solução para este labirinto!"
            self.visited_cells = visited_order
            self.is_animating = True

    # --- Mouse Handling ---
    def handle_mouse_down(self, pos):
        if self.is_animating: return
        
        # Converte pixel para grid coords
        c = (pos[0] - MARGIN_LEFT) // CELL_SIZE
        r = (pos[1] - MARGIN_TOP) // CELL_SIZE
        
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            if (r, c) == self.start_pos:
                self.drag_mode = 'start'
            elif (r, c) == self.goal_pos:
                self.drag_mode = 'goal'
            else:
                self.drag_mode = 'wall'
                # Toggle parede
                self.grid[r][c] = 1 if self.grid[r][c] == 0 else 0

    def handle_mouse_motion(self, pos):
        if self.is_animating or not self.drag_mode: return

        c = (pos[0] - MARGIN_LEFT) // CELL_SIZE
        r = (pos[1] - MARGIN_TOP) // CELL_SIZE
        
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            if self.drag_mode == 'start' and (r, c) != self.goal_pos and self.grid[r][c] != 1:
                self.start_pos = (r, c)
            elif self.drag_mode == 'goal' and (r, c) != self.start_pos and self.grid[r][c] != 1:
                self.goal_pos = (r, c)
            elif self.drag_mode == 'wall':
                if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                    self.grid[r][c] = 1 # Desenha parede

    def handle_mouse_up(self, pos):
        self.drag_mode = None

def draw_app(app, buttons):
    screen.fill(COLOR_BG)
    
    # 1. Desenha Grid
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x = MARGIN_LEFT + c * CELL_SIZE
            y = MARGIN_TOP + r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            
            color = COLOR_GRID_BG
            if app.grid[r][c] == 1: color = COLOR_WALL
            
            # Desenha células visitadas (Exploração)
            if app.is_animating and (r, c) in app.visited_cells[:app.animation_idx]:
                if color == COLOR_GRID_BG: color = COLOR_VISITED
            elif not app.is_animating and (r, c) in app.visited_cells: # Se acabou animação, mantém
                 if color == COLOR_GRID_BG: color = COLOR_VISITED

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, COLOR_LINE, rect, 1) # Borda

    # 2. Desenha Caminho Final (sobrepõe visitados)
    if not app.is_animating or app.animation_idx >= len(app.visited_cells):
        for (r, c) in app.path:
            x = MARGIN_LEFT + c * CELL_SIZE
            y = MARGIN_TOP + r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLOR_PATH, rect)
            pygame.draw.rect(screen, COLOR_LINE, rect, 1)

    # 3. Desenha Start e Goal
    sx, sy = MARGIN_LEFT + app.start_pos[1]*CELL_SIZE, MARGIN_TOP + app.start_pos[0]*CELL_SIZE
    pygame.draw.rect(screen, COLOR_START, (sx, sy, CELL_SIZE, CELL_SIZE))
    
    gx, gy = MARGIN_LEFT + app.goal_pos[1]*CELL_SIZE, MARGIN_TOP + app.goal_pos[0]*CELL_SIZE
    pygame.draw.rect(screen, COLOR_GOAL, (gx, gy, CELL_SIZE, CELL_SIZE))

    # 4. Interface Lateral
    title = font_title.render("Maze Solver", True, (255, 255, 255))
    screen.blit(title, (20, 20))
    
    status = font_stats.render(app.status_msg, True, (200, 200, 200))
    screen.blit(status, (20, 60))
    
    if app.stats_msg:
        stats_bg = pygame.Rect(MARGIN_LEFT, 10, 600, 30)
        pygame.draw.rect(screen, (40, 44, 52), stats_bg, border_radius=5)
        stats = font_stats.render(app.stats_msg, True, (46, 204, 113))
        screen.blit(stats, (MARGIN_LEFT + 10, 15))

    for btn in buttons:
        btn.draw(screen)

def main():
    clock = pygame.time.Clock()
    app = MazeApp()
    
    # Botões
    bx, by = 20, 100
    bw, bh = 210, 40
    gap = 50
    
    buttons = [
        Button(bx, by, bw, bh, "Limpar Caminho", app.clear_path),
        Button(bx, by+gap, bw, bh, "Resetar Paredes", app.reset_grid, color=COLOR_RESET_BTN),
        
        # --- NOVO BOTÃO AQUI ---
        Button(bx, by+gap*2, bw, bh, "Gerar Labirinto Teste", app.create_trap_maze, color=(255, 165, 0)), # Laranja
        
        # Ajuste os 'gap' dos botões abaixo para +20 ou +gap para não encavalar
        Button(bx, by+gap*3+20, bw, bh, "BFS (Largura)", lambda: app.run_algorithm("BFS")),
        Button(bx, by+gap*4+20, bw, bh, "Gulosa (Manhattan)", lambda: app.run_algorithm("Gulosa")),
        
        Button(bx, by+gap*5+40, bw, bh, "A* (Manhattan)", lambda: app.run_algorithm("A*", "h1_manhattan")),
        Button(bx, by+gap*6+40, bw, bh, "A* (Euclidiana)", lambda: app.run_algorithm("A*", "h2_euclidean")),
        Button(bx, by+gap*7+40, bw, bh, "A* (Ponderada x2)", lambda: app.run_algorithm("A*", "h3_weighted_manhattan")),
    ]

    running = True
    speed = 1 # Nós por frame na animação

    while running:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for btn in buttons: btn.check_click(mouse_pos)
                    app.handle_mouse_down(mouse_pos)
            if event.type == pygame.MOUSEMOTION:
                app.handle_mouse_motion(mouse_pos)
                for btn in buttons: btn.check_hover(mouse_pos)
            if event.type == pygame.MOUSEBUTTONUP:
                app.handle_mouse_up(mouse_pos)

        # Lógica de Animação
        if app.is_animating:
            # Acelera a animação se tiver muitos nós
            speed = 5 if len(app.visited_cells) > 500 else 2
            app.animation_idx += speed
            if app.animation_idx >= len(app.visited_cells):
                app.animation_idx = len(app.visited_cells)
                # Não para flag is_animating para manter o desenho, mas para o loop
                # Na prática, apenas garante que desenhou tudo

        draw_app(app, buttons)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()