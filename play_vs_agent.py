# play_vs_agent.py
import pygame
import sys
import math
import numpy as np # For observations and actions
from stable_baselines3 import PPO # To load the model

from game import ConquestGame, Player # Your existing game logic
from config import GameConfig       # Your existing game config

# --- Configuration for Agent ---
MODEL_PATH = "training_logs/ppo_conquest_blue_20250527-173435/models/ppo_conquest_ep_chkpt_ep5500.zip"
AGENT_CONTROLLED_PLAYER = Player.YELLOW
HUMAN_PLAYER = Player.BLUE # As per original main.py UI logic
# --- End Agent Configuration ---

# Initialize Pygame
pygame.init()
pygame.font.init() # Ensure font module is initialized

# Constants (from your main.py)
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GAME_WIDTH = 800
GAME_HEIGHT = 600
UI_WIDTH = 400

COLORS = {
    'grey': (204, 204, 204), 'red': (255, 68, 68), 'green': (68, 255, 68),
    'blue': (68, 68, 255), 'yellow': (255, 255, 68), 'white': (255, 255, 255),
    'black': (0, 0, 0), 'light_gray': (240, 240, 240), 'dark_gray': (100, 100, 100),
    'cyan': (0, 255, 255), 'button_gray': (180, 180, 180), 'button_hover': (160, 160, 160)
}

# --- UI Classes (Copied from your main.py) ---
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
                
    def update(self, dt): # dt is delta time in milliseconds
        self.cursor_timer += dt
        if self.cursor_timer >= 500: # Blink every 500ms
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
            
    def draw(self, surface):
        color = COLORS['white'] if self.active else COLORS['light_gray']
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, COLORS['black'], self.rect, 2)
        
        text_to_render = self.text
        text_color = COLORS['black']
        if not self.text and not self.active:
            text_to_render = "Troops"
            text_color = COLORS['dark_gray']
        
        text_surface = self.font.render(text_to_render, True, text_color)
        surface.blit(text_surface, (self.rect.x + 5, self.rect.y + (self.rect.height - text_surface.get_height()) // 2)) # Center text vertically
            
        if self.active and self.cursor_visible:
            # Calculate cursor position more accurately based on rendered text width
            text_width_so_far = self.font.size(self.text)[0]
            cursor_x = self.rect.x + 5 + text_width_so_far
            pygame.draw.line(surface, COLORS['black'], 
                           (cursor_x, self.rect.y + 5), 
                           (cursor_x, self.rect.y + self.rect.height - 5), 2)
    
    def get_value(self):
        try: return int(self.text) if self.text else 0
        except ValueError: return 0
            
    def set_text(self, text): self.text = str(text)
    def clear(self): self.text = ""
# --- End UI Classes ---


class ConquestGameVsAgentUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f"Conquest Game: Human ({HUMAN_PLAYER.value}) vs Agent ({AGENT_CONTROLLED_PLAYER.value})")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 36)
        
        # --- Agent Setup ---
        self.agent_player = AGENT_CONTROLLED_PLAYER
        try:
            self.agent_model = PPO.load(MODEL_PATH)
            print(f"Successfully loaded agent model from: {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading agent model from {MODEL_PATH}: {e}")
            print("Agent will not be active. Game will proceed with default AI for YELLOW if any.")
            self.agent_model = None
        
        # Game state - Pass agent_player to ConquestGame
        # This tells the game engine not to use its internal AI for this player
        self.game = ConquestGame(GAME_WIDTH, GAME_HEIGHT, agent_player=self.agent_player)
        
        self.selected_source = None
        self.selected_target = None
        
        # UI elements
        self.troop_input = TextInput(GAME_WIDTH + 20, 300, 100, 30, self.font)
        self.send_button = Button(GAME_WIDTH + 130, 300, 100, 30, "Send!", self.font)
        self.reset_button = Button(GAME_WIDTH + 20, 400, 100, 30, "Reset", self.font)

        # For constructing agent observation
        self.player_int_map = Player.get_int_mapping()
        self.active_players_list = Player.get_active_players()
        self.NUM_TROOP_CHOICES = 4 # From ConquestEnv (0-25%, 1-50%, 2-75%, 3-100%)
        self.NO_OP_ACTION_IDX = self.NUM_TROOP_CHOICES # Action index for NO-OP

    def _get_observation_for_agent(self) -> dict:
        """ Constructs the observation dictionary for the PPO agent. """
        padded_state = self.game.get_padded_state_for_agent() # From game.py

        agent_id_onehot = np.zeros(len(self.player_int_map), dtype=np.int8)
        agent_id_onehot[self.player_int_map[self.agent_player]] = 1

        tick_norm = np.array([min(1.0, self.game.current_tick / GameConfig.MAX_EPISODE_TICKS)], dtype=np.float32)

        player_troop_counts = self.game.get_player_troop_counts()
        player_node_counts = self.game.get_scores()
        total_troops = max(1, self.game.get_total_troops_on_map())
        num_actual_nodes_in_game = len(self.game.graph.nodes())
        total_map_nodes = max(1, num_actual_nodes_in_game)

        troop_ratios = np.zeros(len(self.active_players_list), dtype=np.float32)
        node_ratios = np.zeros(len(self.active_players_list), dtype=np.float32)

        for i, p_enum in enumerate(self.active_players_list):
            troop_ratios[i] = player_troop_counts.get(p_enum, 0) / total_troops
            node_ratios[i] = player_node_counts.get(p_enum, 0) / total_map_nodes
            
        num_actual_nodes_norm_val = padded_state["num_actual_nodes"][0] / GameConfig.MAX_NODES

        obs = {
            "node_owners": padded_state["node_owners"],
            "node_troops": padded_state["node_troops"],
            "adjacency_matrix": padded_state["adjacency_matrix"],
            "agent_player_id_onehot": agent_id_onehot,
            "current_tick_norm": tick_norm,
            "player_troop_ratios": troop_ratios,
            "player_node_ratios": node_ratios,
            "num_actual_nodes_norm": np.array([num_actual_nodes_norm_val], dtype=np.float32)
        }
        return obs

    def _execute_agent_action(self, agent_action: np.ndarray):
        """ Decodes and executes the agent's action. """
        source_idx, target_idx, troop_choice_idx = agent_action
        
        if troop_choice_idx == self.NO_OP_ACTION_IDX:
            print(f"Agent ({self.agent_player.value}) chose NO-OP.")
            return

        num_actual_nodes_in_game = len(self.game.graph.nodes())

        # Basic validation (agent should learn this, but good for robustness here)
        if not (0 <= source_idx < num_actual_nodes_in_game and 0 <= target_idx < num_actual_nodes_in_game):
            print(f"Agent ({self.agent_player.value}) chose invalid node index. Source: {source_idx}, Target: {target_idx}. Skipping.")
            return
        if self.game.node_owners.get(source_idx) != self.agent_player:
            print(f"Agent ({self.agent_player.value}) chose source node {source_idx} it doesn't own. Skipping.")
            return
        if source_idx == target_idx:
            print(f"Agent ({self.agent_player.value}) chose same source and target {source_idx}. Skipping.")
            return
        if target_idx not in self.game.graph.neighbors(source_idx):
            print(f"Agent ({self.agent_player.value}) chose non-neighbor target {target_idx} from {source_idx}. Skipping.")
            return

        available_troops = self.game.node_troops.get(source_idx, 0)
        if available_troops <= 0:
            print(f"Agent ({self.agent_player.value}) chose source {source_idx} with no troops. Skipping.")
            return

        percentages = [0.25, 0.50, 0.75, 1.00] # For troop_choice_idx 0-3
        chosen_percentage = percentages[troop_choice_idx]
        
        is_transfer_action = self.game.node_owners.get(target_idx) == self.agent_player
        max_sendable = available_troops
        if not is_transfer_action and available_troops > 1:
            max_sendable = available_troops - 1
        
        troops_to_send = math.floor(max_sendable * chosen_percentage)
        troops_to_send = max(1, troops_to_send) if max_sendable > 0 else 0

        if troops_to_send > 0 and troops_to_send <= available_troops:
            print(f"Agent ({self.agent_player.value}) action: {troops_to_send} troops from {source_idx} to {target_idx}")
            self.game.make_move(self.agent_player, source_idx, target_idx, troops_to_send)
        else:
            print(f"Agent ({self.agent_player.value}) tried to send {troops_to_send} from {source_idx} (available: {available_troops}, max_sendable: {max_sendable}). Invalid. Skipping.")


    def handle_node_click(self, pos): # Human player (BLUE) logic
        node = self.game.get_node_at_position(pos[0], pos[1])
        if node is None: return
            
        if not self.selected_source:
            if (self.game.node_owners.get(node) == HUMAN_PLAYER and 
                self.game.node_troops.get(node, 0) > 0):
                self.selected_source = node
                self.troop_input.set_text(str(self.game.node_troops.get(node,0)))
        else:
            # Use the general get_valid_targets from game.py
            valid_targets = self.game.get_valid_targets(self.selected_source, HUMAN_PLAYER)
            if node in valid_targets:
                self.selected_target = node
            elif node == self.selected_source:
                self.clear_selection()
            # else: invalid target, do nothing or provide feedback
            
    def clear_selection(self): # Human player
        self.selected_source = None
        self.selected_target = None
        self.troop_input.clear()
        
    def execute_human_move(self): # Human player
        if not self.selected_source or not self.selected_target: return
        troops = self.troop_input.get_value()
        max_troops = self.game.node_troops.get(self.selected_source, 0)
        if troops <= 0 or troops > max_troops: return
            
        if self.game.make_move(HUMAN_PLAYER, self.selected_source, self.selected_target, troops):
            self.clear_selection()
                
    # --- Drawing methods (draw_game_board, draw_ui_panel) are identical to your main.py ---
    # --- For brevity, I'll assume they are here. If not, copy them from your main.py ---
    def draw_game_board(self):
        game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        game_surface.fill(COLORS['white'])
        
        if not self.game.graph: return # Handle case where graph might be empty briefly
        
        for edge in self.game.graph.edges():
            pos_s = self.game.node_positions.get(edge[0])
            pos_t = self.game.node_positions.get(edge[1])
            if pos_s and pos_t:
                pygame.draw.line(game_surface, COLORS['dark_gray'],
                                 (pos_s['x'], pos_s['y']), (pos_t['x'], pos_t['y']), 2)
        
        for node_id in self.game.graph.nodes():
            pos = self.game.node_positions.get(node_id)
            if not pos: continue
            owner = self.game.node_owners.get(node_id, Player.GREY)
            troops = self.game.node_troops.get(node_id, 0)
            
            radius = max(15, math.sqrt(max(0, troops)) * 2) # Ensure troops is not negative for sqrt
            color_key = owner.value if owner else Player.GREY.value
            node_color = COLORS.get(color_key, COLORS['black'])
            
            border_color = COLORS['black']
            border_width = 2
            
            if node_id == self.selected_source and self.game.node_owners.get(node_id) == HUMAN_PLAYER:
                border_color = COLORS['black'] # Keep human selection distinct
                border_width = 5
            elif self.selected_source and \
                 self.game.node_owners.get(self.selected_source) == HUMAN_PLAYER and \
                 node_id in self.game.get_valid_targets(self.selected_source, HUMAN_PLAYER):
                border_color = COLORS['cyan']
                border_width = 4
                
            pygame.draw.circle(game_surface, node_color, (int(pos['x']), int(pos['y'])), int(radius))
            pygame.draw.circle(game_surface, border_color, (int(pos['x']), int(pos['y'])), int(radius), border_width)
            
            text_surface = self.font.render(str(troops), True, COLORS['black'])
            text_rect = text_surface.get_rect(center=(pos['x'], pos['y']))
            game_surface.blit(text_surface, text_rect)
        
        for attack_data in self.game.get_ongoing_attacks():
            source_pos = self.game.node_positions.get(attack_data['source'])
            target_pos = self.game.node_positions.get(attack_data['target'])
            if not source_pos or not target_pos: continue
            
            progress = attack_data['progress']
            ball_x = source_pos['x'] + (target_pos['x'] - source_pos['x']) * progress
            ball_y = source_pos['y'] + (target_pos['y'] - source_pos['y']) * progress
            
            player_color_val = attack_data['player']
            ball_color = COLORS.get(player_color_val, COLORS['black'])
            pygame.draw.circle(game_surface, ball_color, (int(ball_x), int(ball_y)), 4)
            pygame.draw.circle(game_surface, COLORS['black'], (int(ball_x), int(ball_y)), 4, 1)
        
        self.screen.blit(game_surface, (0, 0))
        
    def draw_ui_panel(self):
        ui_rect = pygame.Rect(GAME_WIDTH, 0, UI_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLORS['light_gray'], ui_rect)
        pygame.draw.line(self.screen, COLORS['black'], (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        y_offset = 20
        title = self.large_font.render(f"Human ({HUMAN_PLAYER.value}) vs AI ({AGENT_CONTROLLED_PLAYER.value})", True, COLORS['black'])
        self.screen.blit(title, (GAME_WIDTH + 20, y_offset)); y_offset += 40
        
        tick_info = self.small_font.render(f"Tick: {self.game.current_tick}", True, COLORS['black'])
        self.screen.blit(tick_info, (GAME_WIDTH + 20, y_offset)); y_offset += 30
        
        scores_title = self.font.render("Node Scores:", True, COLORS['black'])
        self.screen.blit(scores_title, (GAME_WIDTH + 20, y_offset)); y_offset += 30
        
        scores = self.game.get_scores()
        total_map_nodes = len(self.game.graph.nodes()) if self.game.graph else 0
        for player_enum_val, score in scores.items(): # player_enum_val is Player enum
            color = COLORS.get(player_enum_val.value, COLORS['black'])
            text = self.font.render(f"{player_enum_val.value}: {score}/{total_map_nodes}", True, color)
            self.screen.blit(text, (GAME_WIDTH + 30, y_offset)); y_offset += 20
        y_offset += 10
        
        if self.selected_source and self.game.node_owners.get(self.selected_source) == HUMAN_PLAYER:
            source_owner = self.game.node_owners.get(self.selected_source)
            source_troops = self.game.node_troops.get(self.selected_source,0)
            source_info = self.font.render(f"Source: Node {self.selected_source} ({source_owner.value}, Troops: {source_troops})", True, COLORS['black'])
            self.screen.blit(source_info, (GAME_WIDTH + 20, y_offset)); y_offset += 25
            
            if self.selected_target:
                target_owner = self.game.node_owners.get(self.selected_target)
                target_troops = self.game.node_troops.get(self.selected_target,0)
                action = "Transfer to" if target_owner == HUMAN_PLAYER else "Attack"
                target_info_text = f"{action} Node {self.selected_target} ({target_owner.value}, Troops: {target_troops})"
                target_info = self.font.render(target_info_text, True, COLORS['black'])
                self.screen.blit(target_info, (GAME_WIDTH + 20, y_offset)); y_offset += 25
        else:
            instruction = self.font.render(f"Click a {HUMAN_PLAYER.value} node", True, COLORS['black'])
            self.screen.blit(instruction, (GAME_WIDTH + 20, y_offset)); y_offset += 25

        # Position UI elements lower if selection info is present
        input_y_pos = GAME_HEIGHT // 2 + 50
        self.troop_input.rect.y = input_y_pos
        self.send_button.rect.y = input_y_pos
        self.reset_button.rect.y = input_y_pos + 40

        self.troop_input.draw(self.screen)
        self.send_button.draw(self.screen)
        self.reset_button.draw(self.screen)
        
        y_offset = input_y_pos + 80 # Adjust for ongoing attacks display
        attacks_title = self.font.render("Ongoing Actions:", True, COLORS['black'])
        self.screen.blit(attacks_title, (GAME_WIDTH + 20, y_offset)); y_offset += 30
        
        ongoing = self.game.get_ongoing_attacks()
        if ongoing:
            for attack in ongoing[:6]: # Display up to 6
                player_color = COLORS.get(attack['player'], COLORS['black'])
                action_text = "Transfer" if attack['is_transfer'] else "Attack"
                progress = int(attack['progress'] * 100)
                text_str = f"{attack['player']} {action_text} {attack['target']}: {progress}%"
                text = self.small_font.render(text_str, True, player_color)
                self.screen.blit(text, (GAME_WIDTH + 30, y_offset)); y_offset += 18
        else:
            no_attacks = self.small_font.render("No ongoing actions", True, COLORS['black'])
            self.screen.blit(no_attacks, (GAME_WIDTH + 30, y_offset)); y_offset += 18
        
        if self.game.game_over:
            winner = self.game.get_winner()
            if winner:
                win_text = f"{winner.value.upper()} WINS!"
                win_color = COLORS.get(winner.value, COLORS['black'])
                game_over_s = self.large_font.render(win_text, True, win_color)
                text_rect = game_over_s.get_rect(center=(GAME_WIDTH + UI_WIDTH // 2, WINDOW_HEIGHT - 50))
                self.screen.blit(game_over_s, text_rect)
    # --- End Drawing Methods ---

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(GameConfig.GAME_TICK_SPEED) # Pygame ticks for rendering speed
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if event.pos[0] < GAME_WIDTH:
                            self.handle_node_click(event.pos) # Human player node selection
                
                self.troop_input.handle_event(event)
                if self.send_button.handle_event(event):
                    self.execute_human_move() # Human player sends troops
                if self.reset_button.handle_event(event):
                    self.game.reset_game()
                    self.clear_selection()
            
            self.troop_input.update(dt)
            
            # --- Game Logic Update ---
            # This is where the game ticks forward.
            # We need to inject the agent's action *before* or *as part of* the bot move phase.
            
            # Check if it's time for bot/agent moves based on game ticks
            # The game.update() method handles its own internal ticking for income and attack processing.
            # The agent's move should align with when other bots would move.

            if not self.game.game_over:
                # Agent's turn logic (integrated with bot move timing)
                if self.game.current_tick - self.game.last_bot_move_tick >= GameConfig.BOT_MOVE_INTERVAL:
                    # --- Agent's Turn ---
                    if self.agent_model and self.agent_player and \
                       any(owner == self.agent_player for owner in self.game.node_owners.values()): # If agent is in game
                        
                        obs = self._get_observation_for_agent()
                        # Use deterministic=True for consistent agent behavior during play
                        agent_action, _states = self.agent_model.predict(obs, deterministic=True)
                        self._execute_agent_action(agent_action)
                    
                    # --- Other Bots' Turns (handled by game.update() if agent_player is set) ---
                    # The game.update() will call its internal bot logic for non-agent players
                    # because we initialized ConquestGame with agent_player=self.agent_player.
                    # So, we don't need to explicitly call bot_random_move or bot_green_strategic here
                    # for RED and GREEN if they are not the agent_player.
                    # The game.update() below will handle their turns when it processes bot moves.
                    pass # The game.update() will handle other bots if agent_player is set

            # Update the main game state (processes attacks, income, and non-agent bot moves)
            self.game.update() # This advances the game by one tick and handles bot moves

            # --- Drawing ---
            self.screen.fill(COLORS['white'])
            self.draw_game_board()
            self.draw_ui_panel()
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    ui = ConquestGameVsAgentUI()
    ui.run()