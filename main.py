# main.py - Updated with tick system and better click handling
import pygame
import sys
import math
from game import ConquestGame, Player
from config import GameConfig

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GAME_WIDTH = 800
GAME_HEIGHT = 600
UI_WIDTH = 400

# Colors (same as before)
COLORS = {
    'grey': (204, 204, 204),
    'red': (255, 68, 68),
    'green': (68, 255, 68),
    'blue': (68, 68, 255),
    'yellow': (255, 255, 68),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'light_gray': (240, 240, 240),
    'dark_gray': (100, 100, 100),
    'cyan': (0, 255, 255),
    'button_gray': (180, 180, 180),
    'button_hover': (160, 160, 160)
}

# ... (Button and TextInput classes remain the same) ...
class Button:
    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.hovered = False
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                return True
        return False
        
    def draw(self, surface):
        color = COLORS['button_hover'] if self.hovered else COLORS['button_gray']
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, COLORS['black'], self.rect, 2)
        
        text_surface = self.font.render(self.text, True, COLORS['black'])
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

class TextInput:
    def __init__(self, x, y, width, height, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.text = ""
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit():
                self.text += event.unicode
                
    def update(self, dt):
        self.cursor_timer += dt
        if self.cursor_timer >= 500:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
            
    def draw(self, surface):
        color = COLORS['white'] if self.active else COLORS['light_gray']
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, COLORS['black'], self.rect, 2)
        
        if self.text:
            text_surface = self.font.render(self.text, True, COLORS['black'])
            surface.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))
        else:
            placeholder = self.font.render("Troops", True, COLORS['dark_gray'])
            surface.blit(placeholder, (self.rect.x + 5, self.rect.y + 5))
            
        if self.active and self.cursor_visible:
            cursor_x = self.rect.x + 5 + (len(self.text) * 10)
            pygame.draw.line(surface, COLORS['black'], 
                           (cursor_x, self.rect.y + 3), 
                           (cursor_x, self.rect.y + self.rect.height - 3), 2)
    
    def get_value(self):
        try:
            return int(self.text) if self.text else 0
        except ValueError:
            return 0
            
    def set_text(self, text):
        self.text = str(text)
        
    def clear(self):
        self.text = ""

class ConquestGameUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Conquest Game")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 36)
        
        # Game state
        self.game = ConquestGame(GAME_WIDTH, GAME_HEIGHT)
        self.selected_source = None
        self.selected_target = None
        
        # UI elements
        self.troop_input = TextInput(GAME_WIDTH + 20, 300, 100, 30, self.font)
        self.send_button = Button(GAME_WIDTH + 130, 300, 100, 30, "Send!", self.font)
        self.reset_button = Button(GAME_WIDTH + 20, 400, 100, 30, "Reset", self.font)
        
    def handle_node_click(self, pos):
        node = self.game.get_node_at_position(pos[0], pos[1])
        
        if node is None:
            return
            
        if not self.selected_source:
            # Select source node - must be blue and have troops
            if (self.game.node_owners[node] == Player.BLUE and 
                self.game.node_troops[node] > 0):
                self.selected_source = node
                self.troop_input.set_text(str(self.game.node_troops[node]))
                print(f"Selected source: Node {node} with {self.game.node_troops[node]} troops")
        else:
            # Select target node
            valid_targets = self.game.get_valid_targets(self.selected_source)
            print(f"Valid targets from {self.selected_source}: {valid_targets}")
            
            if node in valid_targets:
                self.selected_target = node
                print(f"Selected target: Node {node}")
            elif node == self.selected_source:
                # Deselect
                print("Deselecting source")
                self.clear_selection()
            else:
                print(f"Invalid target: Node {node} not connected to {self.selected_source}")
            
    def clear_selection(self):
        self.selected_source = None
        self.selected_target = None
        self.troop_input.clear()
        
    def execute_move(self):
        if not self.selected_source or not self.selected_target:
            print("Cannot execute move: missing source or target")
            return
            
        troops = self.troop_input.get_value()
        max_troops = self.game.node_troops[self.selected_source]
        
        if troops <= 0 or troops > max_troops:
            print(f"Invalid troop count: {troops} (max: {max_troops})")
            return
            
        success = self.game.make_move(
            Player.BLUE, 
            self.selected_source, 
            self.selected_target, 
            troops
        )
        
        if success:
            print(f"Move successful: {troops} troops from {self.selected_source} to {self.selected_target}")
            self.clear_selection()
        else:
            print("Move failed!")
                
    def draw_game_board(self):
        # ... (same drawing code as before) ...
        game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        game_surface.fill(COLORS['white'])
        
        # Draw edges
        for edge in self.game.graph.edges():
            source_pos = self.game.node_positions[edge[0]]
            target_pos = self.game.node_positions[edge[1]]
            pygame.draw.line(
                game_surface,
                COLORS['dark_gray'],
                (source_pos['x'], source_pos['y']),
                (target_pos['x'], target_pos['y']),
                2
            )
        
        # Draw nodes
        for node in self.game.graph.nodes():
            pos = self.game.node_positions[node]
            owner = self.game.node_owners[node]
            troops = self.game.node_troops[node]
            
            radius = max(15, math.sqrt(troops) * 2)
            color = COLORS[owner.value]
            
            # Highlight selected nodes and valid targets
            border_color = COLORS['black']
            border_width = 2
            
            if node == self.selected_source:
                border_color = COLORS['black']
                border_width = 5
            elif (self.selected_source and 
                  node in self.game.get_valid_targets(self.selected_source)):
                border_color = COLORS['cyan']
                border_width = 4
                
            # Draw node
            pygame.draw.circle(
                game_surface, 
                color, 
                (int(pos['x']), int(pos['y'])), 
                int(radius)
            )
            pygame.draw.circle(
                game_surface, 
                border_color, 
                (int(pos['x']), int(pos['y'])), 
                int(radius), 
                border_width
            )
            
            # Draw troop count
            text = self.font.render(str(troops), True, COLORS['black'])
            text_rect = text.get_rect(center=(pos['x'], pos['y']))
            game_surface.blit(text, text_rect)
        
        # Draw attack animations
        for attack_data in self.game.get_ongoing_attacks():
            source_pos = self.game.node_positions[attack_data['source']]
            target_pos = self.game.node_positions[attack_data['target']]
            progress = attack_data['progress']
            
            num_balls = max(1, attack_data['troops'] // 10)
            for i in range(num_balls):
                offset = (i / max(1, num_balls - 1)) * 0.2 if num_balls > 1 else 0
                ball_progress = max(0, min(1, progress - offset))
                
                ball_x = source_pos['x'] + (target_pos['x'] - source_pos['x']) * ball_progress
                ball_y = source_pos['y'] + (target_pos['y'] - source_pos['y']) * ball_progress
                
                color = COLORS[attack_data['player']]
                pygame.draw.circle(game_surface, color, (int(ball_x), int(ball_y)), 4)
                pygame.draw.circle(game_surface, COLORS['black'], (int(ball_x), int(ball_y)), 4, 1)
        
        self.screen.blit(game_surface, (0, 0))
        
    def draw_ui_panel(self):
        # ... (same UI drawing code, but add tick display) ...
        ui_rect = pygame.Rect(GAME_WIDTH, 0, UI_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLORS['light_gray'], ui_rect)
        pygame.draw.line(self.screen, COLORS['black'], (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        y_offset = 20
        
        # Title and tick info
        title = self.large_font.render("Conquest Game", True, COLORS['black'])
        self.screen.blit(title, (GAME_WIDTH + 20, y_offset))
        y_offset += 30
        
        tick_info = self.small_font.render(f"Tick: {self.game.current_tick}", True, COLORS['black'])
        self.screen.blit(tick_info, (GAME_WIDTH + 20, y_offset))
        y_offset += 30
        
        # Scores
        scores_title = self.font.render("Scores:", True, COLORS['black'])
        self.screen.blit(scores_title, (GAME_WIDTH + 20, y_offset))
        y_offset += 30
        
        scores = self.game.get_scores()
        total_nodes = len(self.game.graph.nodes())
        for player, score in scores.items():
            color = COLORS[player.value]
            text = self.font.render(f"{player.value}: {score}/{total_nodes} nodes", True, color)
            self.screen.blit(text, (GAME_WIDTH + 30, y_offset))
            y_offset += 25
        
        y_offset += 20
        
        # Selection info
        if self.selected_source:
            source_info = self.font.render(
                f"Source: Node {self.selected_source}", True, COLORS['black']
            )
            self.screen.blit(source_info, (GAME_WIDTH + 20, y_offset))
            y_offset += 25
            
            troops_info = self.small_font.render(
                f"Available troops: {self.game.node_troops[self.selected_source]}", 
                True, COLORS['black']
            )
            self.screen.blit(troops_info, (GAME_WIDTH + 20, y_offset))
            y_offset += 20
            
            if self.selected_target:
                target_node = self.game.node_owners[self.selected_target]
                target_troops = self.game.node_troops[self.selected_target]
                action = "Transfer" if target_node == Player.BLUE else "Attack"
                
                target_info = self.font.render(
                    f"{action} Node {self.selected_target}", True, COLORS['black']
                )
                self.screen.blit(target_info, (GAME_WIDTH + 20, y_offset))
                y_offset += 25
                
                if action == "Attack":
                    needed = target_troops + 1
                    advice = self.small_font.render(
                        f"Defender: {target_troops} troops", True, COLORS['black']
                    )
                    self.screen.blit(advice, (GAME_WIDTH + 20, y_offset))
                    y_offset += 20
                    
                    advice2 = self.small_font.render(
                        f"Need {needed}+ to conquer", True, COLORS['black']
                    )
                    self.screen.blit(advice2, (GAME_WIDTH + 20, y_offset))
                    y_offset += 20
        else:
            instruction = self.font.render(
                "Click a blue node to select", True, COLORS['black']
            )
            self.screen.blit(instruction, (GAME_WIDTH + 20, y_offset))
            y_offset += 25
        
        # Rules and config info
        y_offset += 10
        config_title = self.small_font.render("Game Settings:", True, COLORS['black'])
        self.screen.blit(config_title, (GAME_WIDTH + 20, y_offset))
        y_offset += 20
        
        config_info = [
            f"Tick Speed: {GameConfig.GAME_TICK_SPEED}/sec",
            f"Income: +{GameConfig.INCOME_PER_TICK} every {GameConfig.INCOME_INTERVAL} ticks",
            f"Bot Moves: Every {GameConfig.BOT_MOVE_INTERVAL} ticks",
            "Combat: Need MORE troops to conquer"
        ]
        
        for info in config_info:
            text = self.small_font.render(info, True, COLORS['black'])
            self.screen.blit(text, (GAME_WIDTH + 25, y_offset))
            y_offset += 15
        
        # Draw UI elements
        self.troop_input.draw(self.screen)
        self.send_button.draw(self.screen)
        self.reset_button.draw(self.screen)
        
        # Ongoing attacks (similar to before, adjusted for tick system)
        y_offset = 500
        attacks_title = self.font.render("Ongoing Actions:", True, COLORS['black'])
        self.screen.blit(attacks_title, (GAME_WIDTH + 20, y_offset))
        y_offset += 30
        
        ongoing = self.game.get_ongoing_attacks()
        if ongoing:
            for attack in ongoing[:6]:
                player_color = COLORS[attack['player']]
                action = "Transfer" if attack['is_transfer'] else "Attack"
                progress = int(attack['progress'] * 100)
                
                text = self.small_font.render(
                    f"{attack['player']} {action} {attack['target']}: {progress}%",
                    True, player_color
                )
                self.screen.blit(text, (GAME_WIDTH + 30, y_offset))
                y_offset += 18
        else:
            no_attacks = self.small_font.render("No ongoing actions", True, COLORS['black'])
            self.screen.blit(no_attacks, (GAME_WIDTH + 30, y_offset))
        
        # Game over message
        if self.game.game_over:
            winner = self.game.get_winner()
            if winner:
                game_over_text = self.large_font.render(
                    f"{winner.value.upper()} WINS!", True, COLORS[winner.value]
                )
                self.screen.blit(game_over_text, (GAME_WIDTH + 20, 700))
        
    def run(self):
        running = True
        
        while running:
            dt = self.clock.tick(GameConfig.GAME_TICK_SPEED)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        if event.pos[0] < GAME_WIDTH:  # Click in game area
                            self.handle_node_click(event.pos)
                            
                # Handle UI events
                self.troop_input.handle_event(event)
                
                if self.send_button.handle_event(event):
                    self.execute_move()
                    
                if self.reset_button.handle_event(event):
                    self.game.reset_game()
                    self.clear_selection()
            
            # Update UI elements
            self.troop_input.update(dt)
            
            # Update game state (now tick-based)
            self.game.update()
            
            # Draw everything
            self.screen.fill(COLORS['white'])
            self.draw_game_board()
            self.draw_ui_panel()
            
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game_ui = ConquestGameUI()
    game_ui.run()