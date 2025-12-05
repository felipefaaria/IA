import pygame
import sys
import time
import random
import puzzle_logic as logic

# --- CONFIGURAÇÕES VISUAIS ---
SCREEN_WIDTH, SCREEN_HEIGHT = 950, 600 
BOARD_SIZE = 400
TILE_SIZE = BOARD_SIZE // 3
GAP = 5
OFFSET_X = 50
OFFSET_Y = 100

# Paleta de Cores
COLOR_BG = (40, 44, 52)
COLOR_BOARD = (187, 173, 160)
COLOR_TILE = (238, 228, 218)
COLOR_TEXT = (119, 110, 101)
COLOR_ACCENT = (237, 197, 63)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (100, 149, 237)
COLOR_RESET_BTN = (220, 53, 69)

# --- INICIALIZAÇÃO ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("8-Puzzle IA Solver - Comparativo")

# Fontes
font_tile = pygame.font.SysFont("arial", 60, bold=True)
font_ui = pygame.font.SysFont("arial", 18) 
font_title = pygame.font.SysFont("arial", 35, bold=True)
font_stats = pygame.font.SysFont("arial", 16) 

class Button:
    def __init__(self, x, y, w, h, text, callback, color=COLOR_BUTTON):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.base_color = color
        self.hovered = False

    def draw(self, surface):
        current_color = self.base_color
        if self.hovered:
            current_color = (min(self.base_color[0]+30, 255), 
                             min(self.base_color[1]+30, 255), 
                             min(self.base_color[2]+30, 255))
        
        pygame.draw.rect(surface, current_color, self.rect, border_radius=8)
        
        # Centralização do texto
        text_surf = font_ui.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def check_click(self, mouse_pos):
        if self.hovered and self.callback:
            self.callback()

class GameState:
    def __init__(self):
        self.state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.saved_state = (1, 2, 3, 4, 5, 6, 7, 8, 0) # Para o Reset
        self.solution_path = []
        self.is_animating = False
        self.status_msg = "Bem-vindo! Embaralhe para começar."
        self.stats_msg = ""
        self.temp_stats_msg = "" # Guarda os stats para mostrar só no final
        self.dragging_tile = None
        self.drag_offset = (0, 0)
        self.drag_pos = (0, 0)

    def shuffle_board(self):
        if self.is_animating: return
        self.status_msg = "Gerando estado..."
        draw_game(self, []) # Redesenha rápido
        pygame.display.flip()
        
        nums = list(range(9))
        random.shuffle(nums)
        state = tuple(nums)
        while not logic.is_solvable(state) or state == logic.GOAL_STATE:
            random.shuffle(nums)
            state = tuple(nums)
        
        self.state = state
        self.saved_state = state # SALVA O ESTADO PARA O RESET
        self.solution_path = []
        self.status_msg = "Tabuleiro Embaralhado."
        self.stats_msg = ""

    def reset_board(self):
        if self.is_animating: return
        self.state = self.saved_state
        self.solution_path = []
        self.status_msg = "Tabuleiro Resetado (Mesmo estado)."
        self.stats_msg = ""

    def run_algorithm(self, algo_name, heuristic_name=None):
        if self.is_animating: return
        
        # Se o tabuleiro já estiver resolvido, avisa e não roda
        if self.state == logic.GOAL_STATE:
            self.status_msg = "O tabuleiro já está resolvido!"
            return

        self.status_msg = f"Calculando com {algo_name}..."
        self.stats_msg = "" # Limpa stats anteriores
        draw_game(self, [])
        pygame.display.flip()

        start_time = time.time()
        node = None
        expanded = 0

        # Seleciona Lógica
        if algo_name == "BFS":
            node, expanded = logic.solve_bfs(self.state)
        elif algo_name == "Gulosa":
            h = logic.h2_manhattan
            node, expanded = logic.solve_greedy(self.state, h)
        elif algo_name == "A*":
            h = getattr(logic, heuristic_name)
            node, expanded = logic.solve_astar(self.state, h)

        end_time = time.time()
        
        if node:
            self.solution_path = node.get_path()
            self.solution_path.pop(0)
            duration = end_time - start_time
            
            # GUARDA OS DADOS
            self.temp_stats_msg = f"[{algo_name}] Nós: {expanded} | Tempo IA: {duration:.4f}s | Passos: {len(self.solution_path)}"
            self.status_msg = "Resolvendo visualmente..."
            self.is_animating = True
        else:
            self.status_msg = "Falha/Timeout."

    # --- Lógica de Mouse (Drag & Drop) ---
    def handle_mouse_down(self, pos):
        if self.is_animating: return
        for i, val in enumerate(self.state):
            if val == 0: continue
            row, col = divmod(i, 3)
            x = OFFSET_X + col * TILE_SIZE
            y = OFFSET_Y + row * TILE_SIZE
            if pygame.Rect(x, y, TILE_SIZE, TILE_SIZE).collidepoint(pos):
                zero_idx = self.state.index(0)
                z_row, z_col = divmod(zero_idx, 3)
                if abs(row - z_row) + abs(col - z_col) == 1:
                    self.dragging_tile = (i, val)
                    self.drag_offset = (x - pos[0], y - pos[1])
                    self.drag_pos = (x, y)
                    break

    def handle_mouse_up(self, pos):
        if self.dragging_tile:
            idx, val = self.dragging_tile
            zero_idx = self.state.index(0)
            z_row, z_col = divmod(zero_idx, 3)
            target_rect = pygame.Rect(OFFSET_X + z_col*TILE_SIZE, OFFSET_Y + z_row*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            
            if target_rect.inflate(20, 20).collidepoint(pos):
                new_state = list(self.state)
                new_state[idx], new_state[zero_idx] = new_state[zero_idx], new_state[idx]
                self.state = tuple(new_state)
                # Atualiza o saved_state se o usuário mexer manualmente, 
                # senão o reset volta para antes do movimento dele
                self.saved_state = self.state 
                
                if self.state == logic.GOAL_STATE:
                    self.status_msg = "Resolvido manualmente!"
            self.dragging_tile = None

    def handle_mouse_motion(self, pos):
        if self.dragging_tile:
            self.drag_pos = (pos[0] + self.drag_offset[0], pos[1] + self.drag_offset[1])

# --- DESENHO ---
def draw_board(surface, game):
    pygame.draw.rect(surface, COLOR_BOARD, (OFFSET_X - 5, OFFSET_Y - 5, BOARD_SIZE + 10, BOARD_SIZE + 10), border_radius=10)
    for i, val in enumerate(game.state):
        if game.dragging_tile and game.dragging_tile[0] == i: continue
        row, col = divmod(i, 3)
        x = OFFSET_X + col * TILE_SIZE + GAP
        y = OFFSET_Y + row * TILE_SIZE + GAP
        pygame.draw.rect(surface, (205, 193, 180), (x, y, TILE_SIZE - 2*GAP, TILE_SIZE - 2*GAP), border_radius=5)
        if val != 0:
            color = COLOR_ACCENT if val == (i + 1) else COLOR_TILE
            pygame.draw.rect(surface, color, (x, y, TILE_SIZE - 2*GAP, TILE_SIZE - 2*GAP), border_radius=5)
            text = font_tile.render(str(val), True, COLOR_TEXT)
            text_rect = text.get_rect(center=(x + TILE_SIZE//2 - GAP, y + TILE_SIZE//2 - GAP))
            surface.blit(text, text_rect)

    if game.dragging_tile:
        val = game.dragging_tile[1]
        x, y = game.drag_pos
        pygame.draw.rect(surface, COLOR_TILE, (x, y, TILE_SIZE-2*GAP, TILE_SIZE-2*GAP), border_radius=5)
        pygame.draw.rect(surface, (255, 255, 255), (x, y, TILE_SIZE-2*GAP, TILE_SIZE-2*GAP), width=3, border_radius=5)
        text = font_tile.render(str(val), True, COLOR_TEXT)
        surface.blit(text, text.get_rect(center=(x + TILE_SIZE//2, y + TILE_SIZE//2)))

def draw_ui(surface, game, buttons):
    title = font_title.render("8-Puzzle Solver", True, (255, 255, 255))
    surface.blit(title, (OFFSET_X, 30))

    # Status e Stats (Stats agora desenhados com cor de destaque)
    status = font_ui.render(game.status_msg, True, (200, 200, 200))
    surface.blit(status, (OFFSET_X, BOARD_SIZE + OFFSET_Y + 20))
    
    if game.stats_msg:
        # Fundo escuro para os stats para destacar
        stats_bg = pygame.Rect(OFFSET_X, BOARD_SIZE + OFFSET_Y + 50, BOARD_SIZE, 30)
        pygame.draw.rect(surface, (30, 30, 30), stats_bg, border_radius=5)
        stats = font_stats.render(game.stats_msg, True, COLOR_ACCENT)
        stats_rect = stats.get_rect(center=stats_bg.center)
        surface.blit(stats, stats_rect)

    panel_x = OFFSET_X + BOARD_SIZE + 50
    for btn in buttons:
        btn.draw(surface)

def draw_game(game, buttons):
    screen.fill(COLOR_BG)
    draw_board(screen, game)
    draw_ui(screen, game, buttons)

def main():
    clock = pygame.time.Clock()
    game = GameState()
    
    # --- LAYOUT DOS BOTÕES ---
    bx = OFFSET_X + BOARD_SIZE + 40
    start_y = 100
    bw, bh = 260, 45
    gap = 55 
    
    buttons = [
        Button(bx, start_y, bw, bh, "Embaralhar (Novo Jogo)", game.shuffle_board),
        Button(bx, start_y + gap, bw, bh, "Resetar (Mesmo Jogo)", game.reset_board, color=COLOR_RESET_BTN),
        Button(bx, start_y + gap*2 + 10, bw, bh, "Busca em Largura (BFS)", lambda: game.run_algorithm("BFS")),
        Button(bx, start_y + gap*3 + 10, bw, bh, "Gulosa (Manhattan)", lambda: game.run_algorithm("Gulosa")),
        Button(bx, start_y + gap*4 + 10, bw, bh, "A* (Misplaced)", lambda: game.run_algorithm("A*", "h1_misplaced")),
        Button(bx, start_y + gap*5 + 10, bw, bh, "A* (Manhattan)", lambda: game.run_algorithm("A*", "h2_manhattan")),
        Button(bx, start_y + gap*6 + 10, bw, bh, "A* (Euclidiana)", lambda: game.run_algorithm("A*", "h3_euclidean")),
    ]

    running = True
    animation_timer = 0
    ANIMATION_SPEED = 150 # ms

    while running:
        dt = clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for btn in buttons: btn.check_click(mouse_pos)
                    game.handle_mouse_down(mouse_pos)
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: game.handle_mouse_up(mouse_pos)
            if event.type == pygame.MOUSEMOTION:
                game.handle_mouse_motion(mouse_pos)
                for btn in buttons: btn.check_hover(mouse_pos)

        # Lógica de Animação
        if game.is_animating and game.solution_path:
            animation_timer += dt
            if animation_timer > ANIMATION_SPEED:
                _, next_state = game.solution_path.pop(0)
                game.state = next_state
                animation_timer = 0
                
                # VERIFICAÇÃO DE FINAL DE ANIMAÇÃO
                if not game.solution_path:
                    game.is_animating = False
                    game.status_msg = "Finalizado!"
                    game.stats_msg = game.temp_stats_msg

        draw_game(game, buttons)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()