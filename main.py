import pygame
import numpy as np
import random
import time
from enum import Enum
from typing import List, Tuple, Optional, Set

TILE_SIZE = 30
HEADER_HEIGHT = 60
PANEL_WIDTH = 250  
COLORS = {
    1: (0, 0, 255),   
    2: (0, 128, 0),    
    3: (255, 0, 0),    
    4: (0, 0, 128),    
    5: (128, 0, 0),    
    6: (0, 128, 128),  
    7: (0, 0, 0),      
    8: (128, 128, 128) 
}

class GameState(Enum):
    PLAYING = 0
    WIN = 1
    LOSE = 2

class GameMode(Enum):
    CLASSIC = 0
    AI_CHALLENGE = 1

class TileState(Enum):
    HIDDEN = 0
    REVEALED = 1
    FLAGGED = 2

class GameHistory:
    """Class to track game state history for undo/redo functionality"""
    
    def __init__(self):
        self.history = []
        self.current_index = -1
    
    def push(self, board_state, tile_states):
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        board_copy = np.copy(board_state)
        tile_states_copy = np.copy(tile_states)
        
        self.history.append((board_copy, tile_states_copy))
        self.current_index = len(self.history) - 1
    
    def can_undo(self):
        return self.current_index > 0
    
    def can_redo(self):
        return self.current_index < len(self.history) - 1
    
    def undo(self):
        if self.can_undo():
            self.current_index -= 1
            return self.history[self.current_index]
        return None
    
    def redo(self):
        if self.can_redo():
            self.current_index += 1
            return self.history[self.current_index]
        return None

class PlayerStats:
    """Tracks player performance to adjust difficulty"""
    
    def __init__(self):
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.avg_completion_time = 0
        self.last_few_results = []  
        
    def record_game(self, win: bool, duration: float):
        self.total_games += 1
        if win:
            self.wins += 1
        else:
            self.losses += 1
            
        if self.avg_completion_time == 0:
            self.avg_completion_time = duration
        else:
            self.avg_completion_time = (self.avg_completion_time * (self.total_games - 1) + duration) / self.total_games
            
        self.last_few_results.append(win)
        if len(self.last_few_results) > 5:
            self.last_few_results.pop(0)
    
    def get_win_rate(self):
        if self.total_games == 0:
            return 0.5 
        return self.wins / self.total_games
    
    def get_recent_win_rate(self):
        if not self.last_few_results:
            return 0.5  
        return sum(1 for x in self.last_few_results if x) / len(self.last_few_results)

class MinesweeperAI:
    """Handles AI features: adaptive difficulty and hint system"""
    
    def __init__(self, game):
        self.game = game
        
    def get_hint(self) -> Optional[Tuple[int, int]]:
        """Returns coordinates of the safest move based on probability"""
        safe_moves = self._find_safe_moves()
        if safe_moves:
            return random.choice(list(safe_moves))
        
        return self._find_lowest_probability_move()
    
    def _find_safe_moves(self) -> Set[Tuple[int, int]]:
        """Find moves that are guaranteed to be safe using constraint satisfaction"""
        safe_moves = set()
        
        for i in range(self.game.height):
            for j in range(self.game.width):
                if (self.game.tile_states[i][j] == TileState.REVEALED and 
                    self.game.board[i][j] > 0):
                    
                    hidden = []
                    flagged = 0
                    for ni, nj in self._get_neighbors(i, j):
                        if self.game.tile_states[ni][nj] == TileState.HIDDEN:
                            hidden.append((ni, nj))
                        elif self.game.tile_states[ni][nj] == TileState.FLAGGED:
                            flagged += 1
                    
                    if flagged == self.game.board[i][j] and hidden:
                        safe_moves.update(hidden)
        
        return safe_moves
    
    def _find_lowest_probability_move(self) -> Optional[Tuple[int, int]]:
        """Find the move with lowest probability of containing a mine"""
        probabilities = {}
        
        for i in range(self.game.height):
            for j in range(self.game.width):
                if self.game.tile_states[i][j] == TileState.HIDDEN:
                    prob = self._calculate_mine_probability(i, j)
                    if prob is not None:  
                        probabilities[(i, j)] = prob
        
        if probabilities:
            min_prob_tile = min(probabilities.items(), key=lambda x: x[1])[0]
            return min_prob_tile
        
        hidden_tiles = [(i, j) for i in range(self.game.height) 
                      for j in range(self.game.width) 
                      if self.game.tile_states[i][j] == TileState.HIDDEN]
        if hidden_tiles:
            return random.choice(hidden_tiles)
        
        return None
    
    def _calculate_mine_probability(self, i: int, j: int) -> Optional[float]:
        """Calculate probability this tile contains a mine based on neighbors"""
        default_prob = 0.5
        
        informative_neighbors = []
        for ni, nj in self._get_neighbors(i, j):
            if (self.game.tile_states[ni][nj] == TileState.REVEALED and 
                self.game.board[ni][nj] > 0):
                informative_neighbors.append((ni, nj))
        
        if not informative_neighbors:
            return default_prob
        
        probabilities = []
        for ni, nj in informative_neighbors:
            hidden = []
            flagged = 0
            for nni, nnj in self._get_neighbors(ni, nj):
                if self.game.tile_states[nni][nnj] == TileState.HIDDEN:
                    hidden.append((nni, nnj))
                elif self.game.tile_states[nni][nnj] == TileState.FLAGGED:
                    flagged += 1
            
            if hidden and self.game.board[ni][nj] > 0:
                remaining_mines = max(0, self.game.board[ni][nj] - flagged)
                if (i, j) in hidden:
                    prob = remaining_mines / len(hidden)
                    probabilities.append(prob)
        
        if probabilities:
            return sum(probabilities) / len(probabilities)
        
        return default_prob
    
    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get valid neighboring coordinates"""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (0 <= ni < self.game.height and 
                    0 <= nj < self.game.width):
                    neighbors.append((ni, nj))
        return neighbors
    
    def adjust_difficulty(self) -> int:
        """Adjust number of mines based on player performance"""
        win_rate = self.game.player_stats.get_recent_win_rate()
        
        total_cells = self.game.width * self.game.height
        
        print(f"Adjusting difficulty. Win rate: {win_rate:.2f}")
        
        base_density = 0.15
        
        consecutive_losses = 0
        consecutive_wins = 0
        for result in reversed(self.game.player_stats.last_few_results):
            if not result:  
                consecutive_losses += 1
                consecutive_wins = 0
            else:  
                consecutive_wins += 1
                consecutive_losses = 0
            if consecutive_wins > 0 or consecutive_losses > 0:
                break  
        
        if win_rate > 0.7 or consecutive_wins > 0:  
            density = base_density + 0.1  
            print(f"Player is winning consistently (consecutive wins: {consecutive_wins}) - increasing difficulty")
        elif win_rate < 0.4 or consecutive_losses >= 2: 
            reduction = 0.1
            if consecutive_losses >= 2:
                reduction = 0.15 
                
            density = max(0.08, base_density - reduction)  
            print(f"Player is struggling (consecutive losses: {consecutive_losses}) - decreasing difficulty")
        else:
            density = base_density
            print("Player performance is balanced - maintaining difficulty")
            
        mines = int(total_cells * density)
        
        min_mines = max(5, total_cells // 20)
        max_mines = min(int(total_cells * 0.35), total_cells - 9) 
        mines = max(min_mines, min(mines, max_mines))
        
        print(f"New mine count: {mines} ({(mines/total_cells)*100:.1f}% density)")
        
        return mines
    
    def redistribute_mines(self):
        """Redistribute mines for AI Challenge mode while preserving current game state"""
        if self.game.first_click_made and self.game.game_state == GameState.PLAYING:
            self.game.hint_tile = None
            
            revealed = set()
            for i in range(self.game.height):
                for j in range(self.game.width):
                    if self.game.tile_states[i][j] == TileState.REVEALED:
                        revealed.add((i, j))
            
            hidden_tiles = [(i, j) for i in range(self.game.height) 
                          for j in range(self.game.width) 
                          if (i, j) not in revealed]
            
            if len(hidden_tiles) <= self.game.mines:
                return 
            
            self.game.board = np.zeros((self.game.height, self.game.width), dtype=int)
            mine_positions = random.sample(hidden_tiles, self.game.mines)
            
            for i, j in mine_positions:
                self.game.board[i][j] = -1
            
            self.game._calculate_numbers()
            
            for i, j in revealed:
                self.game.tile_states[i][j] = TileState.REVEALED

class StartupMenu:
    """Startup menu for selecting game parameters"""
    
    def __init__(self, screen, window_width, window_height):
        self.screen = screen
        self.window_width = window_width
        self.window_height = window_height
        
        self.width = 10
        self.height = 10
        self.mines = 15
        self.mode = GameMode.CLASSIC
        
        self.title_font = pygame.font.SysFont('Arial', 32)
        self.menu_font = pygame.font.SysFont('Arial', 24)
        self.button_font = pygame.font.SysFont('Arial', 18)
        
        self.grid_buttons = [
            {'rect': pygame.Rect(self.window_width // 2 - 150, 150, 80, 40), 'size': (10, 10), 'text': "10x10"},
            {'rect': pygame.Rect(self.window_width // 2 - 50, 150, 80, 40), 'size': (15, 15), 'text': "15x15"},
            {'rect': pygame.Rect(self.window_width // 2 + 50, 150, 80, 40), 'size': (20, 20), 'text': "20x20"}
        ]
        
        self.selected_grid = 0 
        
        self.buttons = {
            'mines_minus': pygame.Rect(self.window_width // 2 - 150, 250, 40, 40),
            'mines_plus': pygame.Rect(self.window_width // 2 + 110, 250, 40, 40),
            'mode_toggle': pygame.Rect(self.window_width // 2 - 100, 300, 200, 40),
            'start': pygame.Rect(self.window_width // 2 - 100, 400, 200, 50)
        }
    
    def draw(self):
        """Draw the startup menu"""
        self.screen.fill((50, 50, 50))
        
        title_surface = self.title_font.render("AI-Enhanced Minesweeper", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(self.window_width // 2, 60))
        self.screen.blit(title_surface, title_rect)
        
        size_label = self.menu_font.render("Select Grid Size:", True, (255, 255, 255))
        size_label_rect = size_label.get_rect(midleft=(self.window_width // 2 - 150, 120))
        self.screen.blit(size_label, size_label_rect)
        
        for i, btn in enumerate(self.grid_buttons):
            color = (100, 200, 100) if i == self.selected_grid else (150, 150, 150)
            pygame.draw.rect(self.screen, color, btn['rect'])
            
            text = self.button_font.render(btn['text'], True, (0, 0, 0))
            text_rect = text.get_rect(center=btn['rect'].center)
            self.screen.blit(text, text_rect)
        
        mines_text = self.menu_font.render(f"Mines: {self.mines}", True, (255, 255, 255))
        mines_rect = mines_text.get_rect(center=(self.window_width // 2, 250))
        self.screen.blit(mines_text, mines_rect)
        
        total_cells = self.width * self.height
        density = (self.mines / total_cells) * 100
        density_text = self.menu_font.render(f"Mine Density: {density:.1f}%", True, (255, 255, 255))
        density_rect = density_text.get_rect(center=(self.window_width // 2, 350))
        self.screen.blit(density_text, density_rect)
        
        self._draw_button('mines_minus', "-", (200, 100, 100))
        self._draw_button('mines_plus', "+", (100, 200, 100))
        
        mode_text = "Classic Mode" if self.mode == GameMode.CLASSIC else "AI Challenge"
        mode_color = (150, 150, 150) if self.mode == GameMode.CLASSIC else (200, 100, 100)
        pygame.draw.rect(self.screen, mode_color, self.buttons['mode_toggle'])
        mode_surface = self.button_font.render(mode_text, True, (0, 0, 0))
        mode_rect = mode_surface.get_rect(center=self.buttons['mode_toggle'].center)
        self.screen.blit(mode_surface, mode_rect)
        
        pygame.draw.rect(self.screen, (0, 200, 0), self.buttons['start'])
        start_surface = self.button_font.render("Start Game", True, (0, 0, 0))
        start_rect = start_surface.get_rect(center=self.buttons['start'].center)
        self.screen.blit(start_surface, start_rect)
        
        pygame.display.flip()
    
    def _draw_button(self, name, text, color):
        """Draw a button with text"""
        pygame.draw.rect(self.screen, color, self.buttons[name])
        text_surface = self.button_font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.buttons[name].center)
        self.screen.blit(text_surface, text_rect)
    
    def handle_event(self, event):
        """Handle events for the startup menu"""
        if event.type == pygame.QUIT:
            return False, None
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, btn in enumerate(self.grid_buttons):
                if btn['rect'].collidepoint(event.pos):
                    self.selected_grid = i
                    self.width, self.height = btn['size']
                    self._adjust_mines()
                    return False, None
            
            if self.buttons['mines_minus'].collidepoint(event.pos):
                self.mines = max(1, self.mines - 1)
            elif self.buttons['mines_plus'].collidepoint(event.pos):
                max_mines = min(int(self.width * self.height * 0.5), self.width * self.height - 9)
                self.mines = min(max_mines, self.mines + 1)
                
            elif self.buttons['mode_toggle'].collidepoint(event.pos):
                self.mode = GameMode.AI_CHALLENGE if self.mode == GameMode.CLASSIC else GameMode.CLASSIC
                
            elif self.buttons['start'].collidepoint(event.pos):
                return True, {
                    'width': self.width,
                    'height': self.height,
                    'mines': self.mines,
                    'mode': self.mode
                }
                
        return False, None
    
    def _adjust_mines(self):
        """Adjust mine count when board size changes to maintain reasonable density"""
        total_cells = self.width * self.height
        max_mines = min(int(total_cells * 0.5), total_cells - 9)
        self.mines = min(self.mines, max_mines)

class Minesweeper:
    """Main game class"""
    
    def __init__(self, width=10, height=10, mines=15, mode=GameMode.CLASSIC):
        self.width = width
        self.height = height
        self.mines = mines
        self.mode = mode
        self.player_stats = PlayerStats()
        self.ai = MinesweeperAI(self)
        self.history = GameHistory()
        
        # Game state variables
        self.game_state = GameState.PLAYING
        self.start_time = None
        self.end_time = None
        self.first_click_made = False
        self.hint_tile = None
        
        self.initialize_game()
        
    def initialize_game(self):
        """Initialize or reset the game board"""
        self.board = np.zeros((self.height, self.width), dtype=int)
        
        self.tile_states = np.full((self.height, self.width), TileState.HIDDEN, dtype=object)
        
        self.game_state = GameState.PLAYING
        self.start_time = None
        self.end_time = None
        self.first_click_made = False
        self.hint_tile = None
        
        self.history = GameHistory()
        
        if (self.mode == GameMode.AI_CHALLENGE and 
            hasattr(self, 'player_stats') and 
            self.player_stats.total_games > 0):
            
            old_mines = self.mines
            
            self.mines = self.ai.adjust_difficulty()
            
            if old_mines != self.mines:
                print(f"Difficulty adjusted: {old_mines} -> {self.mines} mines")
        else:
            print(f"Using default mine count: {self.mines}")
        
    
    def place_mines(self, first_i: int, first_j: int):
        """Place mines randomly but ensure first click is safe"""
        all_positions = [(i, j) for i in range(self.height) for j in range(self.width)]
        
        safe_zone = [(first_i + di, first_j + dj) for di in [-1, 0, 1] for dj in [-1, 0, 1]]
        valid_positions = [(i, j) for i, j in all_positions 
                           if 0 <= i < self.height and 0 <= j < self.width and (i, j) not in safe_zone]
        
        if len(valid_positions) < self.mines:
            self.mines = len(valid_positions)
        
        mine_positions = random.sample(valid_positions, self.mines)
        
        for i, j in mine_positions:
            self.board[i][j] = -1
        
        self._calculate_numbers()
        
        self.start_time = time.time()
        self.first_click_made = True
        
        self.history.push(self.board, self.tile_states)
    
    def _calculate_numbers(self):
        """Calculate numbers for non-mine cells based on adjacent mines"""
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] != -1:  # Skip mines
                    mine_count = 0
                    for ni, nj in self.ai._get_neighbors(i, j):
                        if self.board[ni][nj] == -1:
                            mine_count += 1
                    self.board[i][j] = mine_count
    
    def reveal(self, i: int, j: int) -> bool:
        """Reveal a tile and handle cascading reveals for empty cells"""
        if (self.game_state != GameState.PLAYING or
            self.tile_states[i][j] != TileState.HIDDEN):
            return False
        
        if not self.first_click_made:
            self.place_mines(i, j)
        
        self.history.push(self.board, self.tile_states)
        
        self.hint_tile = None
        
        if self.board[i][j] == -1:
            self.tile_states[i][j] = TileState.REVEALED
            self.game_state = GameState.LOSE
            self.end_time = time.time()
            
            duration = self.end_time - self.start_time
            self.player_stats.record_game(win=False, duration=duration)
            return True
        
        self.tile_states[i][j] = TileState.REVEALED
        
        if self.board[i][j] == 0:
            self._cascade_reveal(i, j)
        
        if self._check_win():
            self.game_state = GameState.WIN
            self.end_time = time.time()
            
            duration = self.end_time - self.start_time
            self.player_stats.record_game(win=True, duration=duration)
            
            for i in range(self.height):
                for j in range(self.width):
                    if (self.board[i][j] == -1 and 
                        self.tile_states[i][j] == TileState.HIDDEN):
                        self.tile_states[i][j] = TileState.FLAGGED
        
        return True
    
    def _cascade_reveal(self, i: int, j: int):
        """Recursively reveal adjacent empty cells"""
        for ni, nj in self.ai._get_neighbors(i, j):
            if (self.tile_states[ni][nj] == TileState.HIDDEN):
                self.tile_states[ni][nj] = TileState.REVEALED
                if self.board[ni][nj] == 0:
                    self._cascade_reveal(ni, nj)
    
    def toggle_flag(self, i: int, j: int) -> bool:
        """Toggle flag on a hidden tile"""
        if (self.game_state != GameState.PLAYING or
            self.tile_states[i][j] == TileState.REVEALED):
            return False
        
        self.history.push(self.board, self.tile_states)
        
        self.hint_tile = None
        
        if self.tile_states[i][j] == TileState.HIDDEN:
            self.tile_states[i][j] = TileState.FLAGGED
        else:
            self.tile_states[i][j] = TileState.HIDDEN
        
        return True
    
    def get_hint(self) -> bool:
        """Get a hint for the next move"""
        if self.game_state != GameState.PLAYING:
            return False
        
        hint_coords = self.ai.get_hint()
        if hint_coords:
            if self.tile_states[hint_coords[0]][hint_coords[1]] == TileState.HIDDEN:
                self.hint_tile = hint_coords
                return True
        return False
    
    def undo(self) -> bool:
        """Undo the last move"""
        if not self.history.can_undo():
            return False
        
        result = self.history.undo()
        if result:
            self.board, self.tile_states = result
            self.hint_tile = None
            if self.game_state != GameState.PLAYING:
                self.game_state = GameState.PLAYING
                self.end_time = None
            return True
        return False
    
    def redo(self) -> bool:
        """Redo the last undone move"""
        if not self.history.can_redo():
            return False
        
        result = self.history.redo()
        if result:
            self.board, self.tile_states = result
            self.hint_tile = None
            if self._check_win():
                self.game_state = GameState.WIN
                if not self.end_time:
                    self.end_time = time.time()
            return True
        return False
    
    def _check_win(self) -> bool:
        """Check if all non-mine tiles are revealed"""
        for i in range(self.height):
            for j in range(self.width):
                if (self.board[i][j] != -1 and  # Not a mine
                    self.tile_states[i][j] != TileState.REVEALED): 
                    return False
        return True
    
    def get_game_duration(self) -> float:
        """Get the current game duration in seconds"""
        if not self.start_time:
            return 0
        
        if self.end_time:
            return self.end_time - self.start_time
        
        return time.time() - self.start_time
    
    def change_mode(self, mode: GameMode):
        """Change game mode and reset the board"""
        self.mode = mode
        self.initialize_game()
    
    def resize_board(self, width: int, height: int, mine_density: float = None):
        """Resize the board with optional mine density"""
        self.width = max(5, min(30, width))
        self.height = max(5, min(24, height))
        
        if mine_density is not None:
            total_cells = self.width * self.height
            self.mines = int(total_cells * max(0.1, min(0.5, mine_density)))
        else:
            old_density = self.mines / (self.width * self.height)
            total_cells = self.width * self.height
            self.mines = int(total_cells * old_density)
        
        min_mines = max(5, total_cells // 20)
        max_mines = min(total_cells // 4, total_cells - 9)
        self.mines = max(min_mines, min(self.mines, max_mines))
        
        self.initialize_game()

class MinesweeperGUI:
    """Graphical user interface for Minesweeper game"""
    
    def __init__(self, game=None):
        pygame.init()
        
        screen_info = pygame.display.Info()
        self.max_screen_width = screen_info.current_w
        self.max_screen_height = screen_info.current_h
        
        self.window_width = min(1200, self.max_screen_width - 50)
        self.window_height = min(900, self.max_screen_height - 50)
        
        self.board_width = 0
        self.board_height = 0
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("AI-Enhanced Minesweeper")
        
        self.header_font = pygame.font.SysFont('Arial', 24)
        self.tile_font = pygame.font.SysFont('Arial', 16)
        self.button_font = pygame.font.SysFont('Arial', 14)
        
        self.show_startup_menu = True
        self.startup_menu = StartupMenu(self.screen, self.window_width, self.window_height)
        if game:
            self.game = game
            self.show_startup_menu = False
            self.calculate_window_size() 
            self.update_button_positions()
        else:
            self.game = None
        
        self.images = {
            'hidden': None,
            'flag': None,
            'mine': None,
            'hint': None,
        }
        
        self.buttons = {}
        
        self.autoplay_active = False
        self.autoplay_speed = 0.5 
        self.last_autoplay_move = 0
        
        self.running = True

        self.notification = None
        self.notification_end_time = 0

        self.last_hint_time = 0
        self.hint_cooldown = 0.5 

    def calculate_window_size(self):
        """Calculate window dimensions based on current board size"""
        global PANEL_WIDTH
        
        raw_board_width = self.game.width * TILE_SIZE
        raw_board_height = self.game.height * TILE_SIZE + HEADER_HEIGHT
        
        adjusted_panel_width = max(PANEL_WIDTH, min(raw_board_width // 3, 300))
        
        self.board_width = min(raw_board_width, self.max_screen_width - adjusted_panel_width - 50)
        self.board_height = raw_board_height
        
        self.window_width = min(self.board_width + adjusted_panel_width, self.max_screen_width - 50)
        self.window_height = min(max(self.board_height, 600), self.max_screen_height - 50)
        
        PANEL_WIDTH = adjusted_panel_width

    def update_button_positions(self):
        """Update button positions after window resize"""
        self.buttons = {
            'new_game': pygame.Rect(self.board_width + 20, 100, PANEL_WIDTH - 40, 40),
            'hint': pygame.Rect(self.board_width + 20, 150, PANEL_WIDTH - 40, 40),
            'undo': pygame.Rect(self.board_width + 20, 200, (PANEL_WIDTH - 50) // 2, 40),
            'redo': pygame.Rect(self.board_width + (PANEL_WIDTH - 50) // 2 + 30, 200, (PANEL_WIDTH - 50) // 2, 40),
            'mode_toggle': pygame.Rect(self.board_width + 20, 250, PANEL_WIDTH - 40, 40),
            'size_10x10': pygame.Rect(self.board_width + 20, 300, (PANEL_WIDTH - 50) // 3, 40),
            'size_15x15': pygame.Rect(self.board_width + (PANEL_WIDTH - 50) // 3 + 25, 300, (PANEL_WIDTH - 50) // 3, 40),
            'size_20x20': pygame.Rect(self.board_width + 2 * ((PANEL_WIDTH - 50) // 3) + 30, 300, (PANEL_WIDTH - 50) // 3, 40),
            'autoplay': pygame.Rect(self.board_width + 20, 350, PANEL_WIDTH - 40, 40),
        }

    def draw(self):
        """Draw the game screen"""
        self.screen.fill((240, 240, 240))
        
        if self.show_startup_menu or not self.game:
            return
        
        self.draw_header()
        
        self.draw_board()
        
        self.draw_panel()
        
        self.draw_notification()
        
        pygame.display.flip()

    def draw_notification(self):
        """Draw any active notification"""
        if self.notification and time.time() < self.notification['end_time']:
            notif_bg = pygame.Surface((self.board_width, 40))
            notif_bg.set_alpha(200)
            notif_bg.fill((0, 0, 0))
            
            self.screen.blit(notif_bg, (0, HEADER_HEIGHT))
            
            notif_text = self.header_font.render(self.notification['text'], True, self.notification['color'])
            notif_rect = notif_text.get_rect(center=(self.board_width // 2, HEADER_HEIGHT + 20))
            self.screen.blit(notif_text, notif_rect)

    def draw_header(self):
        """Draw the game header"""
        header_rect = pygame.Rect(0, 0, self.window_width, HEADER_HEIGHT)
        pygame.draw.rect(self.screen, (50, 50, 50), header_rect)
        
        title = self.header_font.render("AI-Enhanced Minesweeper", True, (255, 255, 255))
        title_rect = title.get_rect(midleft=(20, HEADER_HEIGHT // 2))
        self.screen.blit(title, title_rect)
        
        if self.game.first_click_made:
            time_text = f"Time: {self.game.get_game_duration():.1f}s"
            
            remaining_mines = max(0, self.game.mines - self.count_flagged_mines())
            mines_text = f"Mines: {remaining_mines} / {self.game.mines}"
            mine_color = (255, 255, 255)
            if remaining_mines > self.game.mines * 0.7:
                mine_color = (255, 150, 150)  # Reddish
            time_surface = self.header_font.render(time_text, True, (255, 255, 255))
            mines_surface = self.header_font.render(mines_text, True, mine_color)
            
            density = (self.game.mines / (self.game.width * self.game.height)) * 100
            density_text = f"Density: {density:.1f}%"
            density_surface = self.header_font.render(density_text, True, (255, 255, 255))
            
            self.screen.blit(time_surface, (self.board_width - 180, 10))
            self.screen.blit(mines_surface, (self.board_width - 180, 35))
            self.screen.blit(density_surface, (self.board_width - 400, HEADER_HEIGHT // 2 - 10))
        else:
            mines_text = f"Mines: {self.game.mines}"
            mines_surface = self.header_font.render(mines_text, True, (255, 255, 255))
            self.screen.blit(mines_surface, (self.board_width - 180, HEADER_HEIGHT // 2))

    def count_flagged_mines(self):
        """Count flagged mines"""
        return sum(1 for i in range(self.game.height) for j in range(self.game.width) 
                   if self.game.tile_states[i][j] == TileState.FLAGGED)

    def draw_board(self):
        """Draw the game board"""
        for i in range(self.game.height):
            for j in range(self.game.width):
                x = j * TILE_SIZE
                y = HEADER_HEIGHT + i * TILE_SIZE
                
                if self.game.tile_states[i][j] == TileState.HIDDEN:
                    self.screen.blit(self.images['hidden'], (x, y))
                    if self.game.hint_tile == (i, j):
                        self.screen.blit(self.images['hint'], (x, y))
                elif self.game.tile_states[i][j] == TileState.FLAGGED:
                    self.screen.blit(self.images['flag'], (x, y))
                elif self.game.tile_states[i][j] == TileState.REVEALED:
                    if self.game.board[i][j] == -1:
                        self.screen.blit(self.images['mine'], (x, y))
                    else:
                        color = (220, 220, 220)
                        tile_surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
                        tile_surface.fill(color)
                        
                        pygame.draw.rect(tile_surface, (180, 180, 180), 
                                        (0, 0, TILE_SIZE, TILE_SIZE), 1)
                        
                        if self.game.board[i][j] > 0:
                            num_color = COLORS.get(self.game.board[i][j], (0, 0, 0))
                            number = self.tile_font.render(
                                str(self.game.board[i][j]), True, num_color)
                            number_rect = number.get_rect(
                                center=(TILE_SIZE // 2, TILE_SIZE // 2))
                            tile_surface.blit(number, number_rect)
                        
                        self.screen.blit(tile_surface, (x, y))

    def draw_panel(self):
        """Draw the control panel"""
        panel_rect = pygame.Rect(self.board_width, 0, PANEL_WIDTH, self.window_height)
        pygame.draw.rect(self.screen, (70, 70, 70), panel_rect)
        
        self.draw_button('new_game', "New Game", (100, 200, 100))
        self.draw_button('hint', "Hint", (100, 100, 200))
        self.draw_button('undo', "Undo", (200, 150, 100))
        self.draw_button('redo', "Redo", (200, 150, 100))
        
        mode_text = "Classic Mode" if self.game.mode == GameMode.CLASSIC else "AI Challenge"
        mode_color = (150, 150, 150) if self.game.mode == GameMode.CLASSIC else (200, 100, 100)
        self.draw_button('mode_toggle', mode_text, mode_color)
        
        self.draw_button('size_10x10', "10x10", (150, 150, 200))
        self.draw_button('size_15x15', "15x15", (150, 150, 200))
        self.draw_button('size_20x20', "20x20", (150, 150, 200))
        
        autoplay_color = (100, 255, 100) if self.autoplay_active else (150, 150, 150)
        autoplay_text = "Stop Autoplay" if self.autoplay_active else "Autoplay"
        self.draw_button('autoplay', autoplay_text, autoplay_color)
        
        if self.game.mode == GameMode.AI_CHALLENGE:
            win_rate = self.game.player_stats.get_win_rate() * 100
            stats_text = f"Win Rate: {win_rate:.1f}%"
            stats_surface = self.button_font.render(stats_text, True, (255, 255, 255))
            self.screen.blit(stats_surface, (self.board_width + 20, 450))
            
            games_text = f"Games Played: {self.game.player_stats.total_games}"
            games_surface = self.button_font.render(games_text, True, (255, 255, 255))
            self.screen.blit(games_surface, (self.board_width + 20, 470))
            
            recent_win_rate = self.game.player_stats.get_recent_win_rate()
            difficulty_text = "Difficulty: "
            if recent_win_rate > 0.7:
                difficulty_text += "Hard"
                difficulty_color = (255, 100, 100)  
            elif recent_win_rate < 0.3:
                difficulty_text += "Easy"
                difficulty_color = (100, 255, 100)  
            else:
                difficulty_text += "Medium"
                difficulty_color = (255, 255, 100) 
                
            difficulty_surface = self.button_font.render(difficulty_text, True, difficulty_color)
            self.screen.blit(difficulty_surface, (self.board_width + 20, 490))
            
            # Show consecutive losses if any
            consecutive_losses = 0
            for result in reversed(self.game.player_stats.last_few_results):
                if not result:
                    consecutive_losses += 1
                else:
                    break
                    
            if consecutive_losses > 0:
                losses_text = f"Consecutive Losses: {consecutive_losses}"
                losses_surface = self.button_font.render(losses_text, True, (255, 150, 150))
                self.screen.blit(losses_surface, (self.board_width + 20, 510))

    def draw_button(self, name, text, color):
        """Draw a button with text"""
        pygame.draw.rect(self.screen, color, self.buttons[name])
        
        text_surface = self.button_font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.buttons[name].center)
        self.screen.blit(text_surface, text_rect)

    def draw_game_over(self):
        """Draw game over overlay"""
        if self.game.game_state == GameState.PLAYING:
            return
        
        overlay = pygame.Surface((self.board_width, self.board_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        if self.game.game_state == GameState.WIN:
            message = "You Won!"
            color = (0, 255, 0)
        else:
            message = "Game Over"
            color = (255, 0, 0)
            
        message_surface = self.header_font.render(message, True, color)
        message_rect = message_surface.get_rect(
            center=(self.board_width // 2, self.board_height // 2 - 20))
        self.screen.blit(message_surface, message_rect)
        
        time_text = f"Time: {self.game.get_game_duration():.1f}s"
        time_surface = self.button_font.render(time_text, True, (255, 255, 255))
        time_rect = time_surface.get_rect(
            center=(self.board_width // 2, self.board_height // 2 + 10))
        self.screen.blit(time_surface, time_rect)
        
        prompt = "Press SPACE to restart"
        prompt_surface = self.button_font.render(prompt, True, (255, 255, 255))
        prompt_rect = prompt_surface.get_rect(
            center=(self.board_width // 2, self.board_height // 2 + 40))
        self.screen.blit(prompt_surface, prompt_rect)
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        previous_mines = 0
        
        while self.running:
            if self.show_startup_menu:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                    
                    start_game, game_params = self.startup_menu.handle_event(event)
                    if start_game:
                        self.game = Minesweeper(
                            width=game_params['width'], 
                            height=game_params['height'], 
                            mines=game_params['mines'],
                            mode=game_params['mode']
                        )
                        
                        self.calculate_window_size()
                        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                        self.update_button_positions()
                        
                        self.images = {       
                            'hidden': self._create_tile_surface((200, 200, 200)),
                            'flag': self._create_flag_surface(),
                            'mine': self._create_mine_surface(),
                            'hint': self._create_hint_surface(),
                        }
                        
                        previous_mines = self.game.mines
                        
                        self.show_startup_menu = False
                    
                    if self.show_startup_menu:
                        self.startup_menu.draw()
            else:
                self.handle_events()
                
                keys = pygame.key.get_pressed()
                if self.game.game_state != GameState.PLAYING and keys[pygame.K_SPACE]:
                    self.game.initialize_game()
                    self.autoplay_active = False  
                
                if (self.autoplay_active and 
                    self.game.game_state == GameState.PLAYING and
                    time.time() - self.last_autoplay_move > self.autoplay_speed):
                    
                    self.game.hint_tile = None
                    
                    self.make_ai_move()
                    self.last_autoplay_move = time.time()
                
                self.draw()
                
                if self.game.game_state != GameState.PLAYING:
                    self.draw_game_over()
                    pygame.display.flip()
                
                if (self.game.mode == GameMode.AI_CHALLENGE and 
                    self.game.first_click_made and 
                    self.game.game_state == GameState.PLAYING):
                    
                    current_time = time.time()
                    if current_time - self.last_hint_time > self.hint_cooldown:
                        self.last_hint_time = current_time
                        
                        revealed_count = sum(1 for i in range(self.game.height) for j in range(self.game.width) 
                                            if self.game.tile_states[i][j] == TileState.REVEALED)
                        total_cells = self.game.height * self.game.width
                        
                        if self.game.mines > 0:
                            game_progress = revealed_count / max(1, total_cells - self.game.mines)
                            game_time = self.game.get_game_duration()
                            
                            if game_progress > 0.3 and game_time < 10:
                                self.game.ai.redistribute_mines()
                            
                            elif game_progress < 0.1 and game_time > 30:
                                self.game.ai.redistribute_mines()
                
                if previous_mines != 0 and previous_mines != self.game.mines:
                    if previous_mines < self.game.mines:
                        self.show_notification(f"Difficulty increased: {self.game.mines} mines", (255, 150, 150), 3.0)
                    else:
                        self.show_notification(f"Difficulty decreased: {self.game.mines} mines", (150, 255, 150), 3.0)
                    
                    previous_mines = self.game.mines
            
            clock.tick(30)
        pygame.quit()
    
    def make_ai_move(self):
        """Make a move using AI logic for autoplay feature"""
        hint_pos = self.game.ai.get_hint()
        
        if hint_pos and self.game.tile_states[hint_pos[0]][hint_pos[1]] == TileState.HIDDEN:
            self.game.reveal(hint_pos[0], hint_pos[1])
            return True
        
        unflagged_mines = []
        for i in range(self.game.height):
            for j in range(self.game.width):
                if (self.game.board[i][j] == -1 and 
                    self.game.tile_states[i][j] == TileState.HIDDEN):
                    unflagged_mines.append((i, j))
        
        if unflagged_mines and random.random() < 0.2:
            mine = random.choice(unflagged_mines)
            self.game.toggle_flag(mine[0], mine[1])
            return True
        
        hidden_cells = []
        for i in range(self.game.height):
            for j in range(self.game.width):
                if self.game.tile_states[i][j] == TileState.HIDDEN:
                    hidden_cells.append((i, j))
        
        if hidden_cells:
            pos = random.choice(hidden_cells)
            self.game.reveal(pos[0], pos[1])
            return True
        
        return False

    def _create_tile_surface(self, color):
        """Create a surface for a tile with the given color"""
        surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
        surface.fill(color)
        pygame.draw.rect(surface, (100, 100, 100), (0, 0, TILE_SIZE, TILE_SIZE), 1)
        return surface

    def _create_flag_surface(self):
        """Create a surface for a flagged tile"""
        surface = self._create_tile_surface((200, 200, 200))
        pygame.draw.rect(surface, (255, 0, 0), 
                        (TILE_SIZE // 2 - 2, TILE_SIZE // 4, 4, TILE_SIZE // 2), 0)
        pygame.draw.polygon(surface, (255, 0, 0), 
                           [(TILE_SIZE // 2, TILE_SIZE // 4), 
                            (TILE_SIZE // 2 + TILE_SIZE // 3, TILE_SIZE // 3), 
                            (TILE_SIZE // 2, TILE_SIZE // 2.5)])
        return surface

    def _create_mine_surface(self):
        """Create a surface for a mine tile"""
        surface = self._create_tile_surface((200, 0, 0))
        pygame.draw.circle(surface, (0, 0, 0), 
                         (TILE_SIZE // 2, TILE_SIZE // 2), TILE_SIZE // 3)
        return surface

    def _create_hint_surface(self):
        """Create a semi-transparent hint overlay"""
        surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(surface, (0, 255, 0, 100), (0, 0, TILE_SIZE, TILE_SIZE))
        return surface

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.window_width, self.window_height = event.size
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
                if self.game:
                    self.calculate_window_size()
                    self.update_button_positions()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()

    def handle_mouse_click(self, event):
        """Handle mouse clicks on the game board and buttons"""
        x, y = event.pos
        
        if x < self.board_width and y >= HEADER_HEIGHT:
            board_x = x // TILE_SIZE
            board_y = (y - HEADER_HEIGHT) // TILE_SIZE
            
            if 0 <= board_x < self.game.width and 0 <= board_y < self.game.height:
                if event.button == 1:
                    self.game.reveal(board_y, board_x)
                elif event.button == 3:
                    self.game.toggle_flag(board_y, board_x)
        
        else:
            if self.buttons['new_game'].collidepoint(event.pos):
                self.game.initialize_game()
                self.autoplay_active = False
                
            elif self.buttons['hint'].collidepoint(event.pos):
                self.game.get_hint()
            
            elif self.buttons['undo'].collidepoint(event.pos):
                self.game.undo()
                self.autoplay_active = False
            
            elif self.buttons['redo'].collidepoint(event.pos):
                self.game.redo()
            
            elif self.buttons['mode_toggle'].collidepoint(event.pos):
                new_mode = GameMode.CLASSIC if self.game.mode == GameMode.AI_CHALLENGE else GameMode.AI_CHALLENGE
                self.game.change_mode(new_mode)
            
            elif self.buttons['size_10x10'].collidepoint(event.pos):
                self.game.resize_board(10, 10)
                self.calculate_window_size()
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                self.update_button_positions()
                self.autoplay_active = False
            
            elif self.buttons['size_15x15'].collidepoint(event.pos):
                self.game.resize_board(15, 15)
                self.calculate_window_size()
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                self.update_button_positions()
                self.autoplay_active = False
            
            elif self.buttons['size_20x20'].collidepoint(event.pos):
                self.game.resize_board(20, 20)
                self.calculate_window_size()
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                self.update_button_positions()
                self.autoplay_active = False
                
            elif self.buttons['autoplay'].collidepoint(event.pos):
                self.autoplay_active = not self.autoplay_active
                self.last_autoplay_move = time.time()
                
                if self.autoplay_active:
                    self.show_notification("Autoplay started", (100, 255, 100))
                else:
                    self.show_notification("Autoplay stopped", (255, 150, 150))

    def show_notification(self, text, color=(255, 255, 255), duration=2.0):
        """Show a temporary notification message"""
        self.notification = {
            'text': text,
            'color': color,
            'end_time': time.time() + duration
        }
def main():
    gui = MinesweeperGUI()
    gui.run()

if __name__ == "__main__":
    main()
