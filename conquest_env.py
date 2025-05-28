# conquest_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math # For troop calculation
from typing import Dict, List, Tuple, Optional, Any

# Imports from your project structure
from config import GameConfig
from game import ConquestGame, Player # Assuming Attack is not directly used by env interface
from bots import BotStrategy, RandomBot, AggressiveBot # For type hinting or defaults if needed

# Pygame for rendering (optional import for headless mode)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    # print("Pygame not found. Human rendering mode will not be available.")


class ConquestEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', None], # None for headless
        'render_fps': 10 # For human mode visualization
    }

    def __init__(self, 
                 agent_player_id: Player = Player.BLUE, 
                 opponent_bot_config: Optional[Dict[Player, Any]] = None, # Config for non-agent players
                 render_mode: Optional[str] = None):
        super().__init__()

        self.agent_player = agent_player_id
        
        # Prepare the full bot_config for the ConquestGame instance
        # Start with the provided opponent config (which should not include the agent_player_id)
        full_bot_config_for_game = opponent_bot_config.copy() if opponent_bot_config is not None else {}
        
        # Mark the agent player as "rl_agent" so ConquestGame's internal update loop skips its turn.
        full_bot_config_for_game[self.agent_player] = "rl_agent" 
        
        # Ensure any other active players not specified in opponent_bot_config get a default AI
        # (ConquestGame's __init__ already does this if a player is missing from its bot_config)
        # So, we just need to ensure the agent is marked.
        
        # Initialize ConquestGame with the constructed bot configuration
        self.game = ConquestGame(bot_config=full_bot_config_for_game) 
        
        self.player_int_map = Player.get_int_mapping()
        self.int_player_map = {v: k for k,v in self.player_int_map.items()} # For debugging or rendering
        self.active_players_list = Player.get_active_players() # For consistent indexing in obs

        # Action space: (source_node_idx, target_node_idx, troop_choice_idx)
        # troop_choice_idx: 0=25%, 1=50%, 2=75%, 3=100% (of sendable troops)
        # Last index is for NO-OP
        self.NUM_TROOP_CHOICES = 4 
        self.NO_OP_ACTION_IDX = self.NUM_TROOP_CHOICES 
        
        self.action_space = spaces.MultiDiscrete([
            GameConfig.MAX_NODES,         # Source node index
            GameConfig.MAX_NODES,         # Target node index
            self.NUM_TROOP_CHOICES + 1    # Troop choice (0-3 for percentages, last for NO_OP)
        ])

        # Observation space (using Dict)
        _player_ids_len = len(self.player_int_map) # Includes Grey
        _active_player_ids_len = len(self.active_players_list) # Non-Grey players

        self.observation_space = spaces.Dict({
            "node_owners": spaces.Box(low=0, high=_player_ids_len - 1, shape=(GameConfig.MAX_NODES,), dtype=np.int32),
            "node_troops": spaces.Box(low=0, high=np.inf, shape=(GameConfig.MAX_NODES,), dtype=np.float32),
            "adjacency_matrix": spaces.Box(low=0, high=1, shape=(GameConfig.MAX_NODES, GameConfig.MAX_NODES), dtype=np.int8),
            "agent_player_id_onehot": spaces.Box(low=0, high=1, shape=(_player_ids_len,), dtype=np.int8),
            "current_tick_norm": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "player_troop_ratios": spaces.Box(low=0, high=1.0, shape=(_active_player_ids_len,), dtype=np.float32),
            "player_node_ratios": spaces.Box(low=0, high=1.0, shape=(_active_player_ids_len,), dtype=np.float32),
            "num_actual_nodes_norm": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.large_font = None
        self.ui_colors = {} # For rendering

        if self.render_mode == "human" and not PYGAME_AVAILABLE:
            print("Warning: Human render mode selected, but Pygame is not available. Falling back to no rendering.")
            self.render_mode = None # Fallback
        
        self.current_agent_steps = 0 # Tracks steps within an episode for this env instance


    def _get_obs(self) -> Dict[str, Any]:
        padded_state = self.game.get_padded_state_for_agent() # From game.py

        agent_id_onehot = np.zeros(len(self.player_int_map), dtype=np.int8)
        if self.agent_player in self.player_int_map: # Ensure agent_player is valid
            agent_id_onehot[self.player_int_map[self.agent_player]] = 1

        tick_norm = np.array([min(1.0, self.game.current_tick / GameConfig.MAX_EPISODE_TICKS)], dtype=np.float32)

        player_troop_counts = self.game.get_player_troop_counts()
        player_node_counts = self.game.get_scores() # get_scores returns node counts
        total_troops = max(1, self.game.get_total_troops_on_map())
        
        num_actual_nodes_in_game = len(self.game.graph.nodes()) if self.game.graph else 0
        total_map_nodes = max(1, num_actual_nodes_in_game)

        troop_ratios = np.zeros(len(self.active_players_list), dtype=np.float32)
        node_ratios = np.zeros(len(self.active_players_list), dtype=np.float32)

        for i, p_enum in enumerate(self.active_players_list):
            troop_ratios[i] = player_troop_counts.get(p_enum, 0) / total_troops
            node_ratios[i] = player_node_counts.get(p_enum, 0) / total_map_nodes
            
        num_actual_nodes_norm_val = padded_state["num_actual_nodes"][0] / GameConfig.MAX_NODES \
                                    if GameConfig.MAX_NODES > 0 else 0.0

        return {
            "node_owners": padded_state["node_owners"],
            "node_troops": padded_state["node_troops"],
            "adjacency_matrix": padded_state["adjacency_matrix"],
            "agent_player_id_onehot": agent_id_onehot,
            "current_tick_norm": tick_norm,
            "player_troop_ratios": troop_ratios,
            "player_node_ratios": node_ratios,
            "num_actual_nodes_norm": np.array([num_actual_nodes_norm_val], dtype=np.float32)
        }

    def _get_info(self) -> Dict[str, Any]:
        # This info is returned at the end of each step
        return {
            "current_tick": self.game.current_tick,
            "agent_player_enum": self.agent_player, # For debugging or external use
            "agent_troops": self.game.get_player_troop_counts().get(self.agent_player, 0),
            "agent_nodes": self.game.get_scores().get(self.agent_player, 0),
            "winner": self.game.get_winner(), # Crucial for callbacks
            "num_actual_nodes": len(self.game.graph.nodes()) if self.game.graph else 0
            # Add any other info useful for callbacks or debugging
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed) # Important for reproducibility via seeding
        if seed is not None:
            random.seed(seed) # Seed Python's random for game setup
            np.random.seed(seed) # Seed NumPy if used directly for game setup randomness

        self.game.reset_game() # Resets the underlying game engine
        self.current_agent_steps = 0
        
        observation = self._get_obs()
        info = self._get_info() # Get initial info

        if self.render_mode == "human":
            self._init_render_if_needed() # Ensure pygame is ready
            self._render_frame()
            
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.current_agent_steps += 1
        
        source_idx, target_idx, troop_choice_idx = action
        reward = 0.0
        terminated = False # Game ended based on win/loss condition
        truncated = False  # Episode ended due to time limit or step limit
        action_penalty = 0.0
        was_invalid_action_attempt = False # For info dict

        num_actual_nodes_in_game = len(self.game.graph.nodes()) if self.game.graph else 0

        if troop_choice_idx == self.NO_OP_ACTION_IDX:
            pass # Valid No-Op
        elif not (0 <= source_idx < num_actual_nodes_in_game and 0 <= target_idx < num_actual_nodes_in_game):
            action_penalty -= 1.0; was_invalid_action_attempt = True
        elif self.game.node_owners.get(source_idx) != self.agent_player:
            action_penalty -= 0.5; was_invalid_action_attempt = True
        elif source_idx == target_idx: # Cannot target self
            action_penalty -= 0.5; was_invalid_action_attempt = True
        elif target_idx not in self.game.graph.neighbors(source_idx): # Check for valid neighbor
            action_penalty -= 0.5; was_invalid_action_attempt = True
        else:
            available_troops = self.game.node_troops.get(source_idx, 0)
            if available_troops > 0:
                percentages = [0.25, 0.50, 0.75, 1.00]
                chosen_percentage = percentages[troop_choice_idx]
                
                is_transfer_action = self.game.node_owners.get(target_idx) == self.agent_player
                max_sendable = available_troops
                if not is_transfer_action and available_troops > 1: # Attacking, try to leave one
                    max_sendable = available_troops - 1
                
                troops_to_send = math.floor(max_sendable * chosen_percentage)
                troops_to_send = max(1, troops_to_send) if max_sendable > 0 else 0

                if troops_to_send > 0 and troops_to_send <= available_troops :
                    if not self.game.make_move(self.agent_player, source_idx, target_idx, troops_to_send):
                        action_penalty -= 0.1; was_invalid_action_attempt = True # make_move failed
                else: 
                     action_penalty -= 0.2; was_invalid_action_attempt = True # Bad troop amount logic
            else: 
                action_penalty -= 0.5; was_invalid_action_attempt = True # No troops at source
        
        reward += action_penalty

        prev_agent_troops = self.game.get_player_troop_counts().get(self.agent_player, 0)
        prev_agent_nodes = self.game.get_scores().get(self.agent_player, 0)

        game_ended_during_sim = False
        for _ in range(GameConfig.AGENT_STEP_INTERVAL_TICKS):
            if self.game.update(): # game.update() returns True if game ended that tick
                game_ended_during_sim = True
                break
        
        terminated = self.game.game_over or game_ended_during_sim

        current_agent_troops = self.game.get_player_troop_counts().get(self.agent_player, 0)
        current_agent_nodes = self.game.get_scores().get(self.agent_player, 0)

        reward += (current_agent_troops - prev_agent_troops) * 0.05
        reward += (current_agent_nodes - prev_agent_nodes) * 0.5
        
        winner = self.game.get_winner()
        if terminated:
            if winner == self.agent_player: reward += 50.0
            elif winner is not None: reward -= 50.0

        if self.current_agent_steps >= GameConfig.MAX_AGENT_STEPS_PER_EPISODE: truncated = True
        if self.game.current_tick >= GameConfig.MAX_EPISODE_TICKS: truncated = True
        
        if truncated and not terminated: reward -= 5.0 

        if not terminated and (current_agent_troops == 0 and current_agent_nodes == 0):
            terminated = True; reward -= 25.0

        observation = self._get_obs()
        info = self._get_info() # Get fresh info after game updates
        info["was_invalid_action_attempt"] = was_invalid_action_attempt # Add this for potential callback use

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info

    def _init_render_if_needed(self):
        if self.render_mode == "human" and PYGAME_AVAILABLE and self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window_surface = pygame.display.set_mode((1200, 800))
            pygame.display.set_caption(f"Conquest Gym Env - Agent: {self.agent_player.value}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.large_font = pygame.font.Font(None, 36)
            self.ui_colors = { # Define colors here if not globally accessible
                'grey': (204,204,204), 'red': (255,68,68), 'green': (68,255,68),
                'blue': (68,68,255), 'yellow': (255,255,68), 'white': (255,255,255),
                'black': (0,0,0), 'dark_gray': (100,100,100), 'cyan': (0,255,255),
                'light_gray': (240,240,240)
            }
        elif self.render_mode == "rgb_array" and PYGAME_AVAILABLE and self.font is None:
            # For rgb_array, we still need fonts if text is rendered on the canvas
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.large_font = pygame.font.Font(None, 36)
            self.ui_colors = {
                'grey': (204,204,204), 'red': (255,68,68), 'green': (68,255,68),
                'blue': (68,68,255), 'yellow': (255,255,68), 'white': (255,255,255),
                'black': (0,0,0), 'dark_gray': (100,100,100), 'cyan': (0,255,255),
                'light_gray': (240,240,240)
            }


    def _render_game_board(self, surface: pygame.Surface):
        if not hasattr(self, 'ui_colors') or not self.ui_colors: self._init_render_if_needed()
        game_area_width = 800 # Consistent with UI assumptions
        surface.fill(self.ui_colors.get('white', (255,255,255)))

        if not self.game.graph: return
        
        for edge in self.game.graph.edges():
            pos_s = self.game.node_positions.get(edge[0])
            pos_t = self.game.node_positions.get(edge[1])
            if pos_s and pos_t:
                pygame.draw.line(surface, self.ui_colors.get('dark_gray', (100,100,100)),
                                 (pos_s['x'], pos_s['y']), (pos_t['x'], pos_t['y']), 2)
        
        for node_id in self.game.graph.nodes():
            pos = self.game.node_positions.get(node_id)
            if not pos: continue
            owner = self.game.node_owners.get(node_id, Player.GREY)
            troops = self.game.node_troops.get(node_id, 0)
            
            radius = max(15, math.sqrt(max(0, troops)) * 2)
            color_key = owner.value if owner else Player.GREY.value
            node_color = self.ui_colors.get(color_key, self.ui_colors.get('black',(0,0,0)))
            
            border_color = self.ui_colors.get('cyan', (0,255,255)) if owner == self.agent_player else self.ui_colors.get('black',(0,0,0))
            border_width = 3 if owner == self.agent_player else 2
                
            pygame.draw.circle(surface, node_color, (int(pos['x']), int(pos['y'])), int(radius))
            pygame.draw.circle(surface, border_color, (int(pos['x']), int(pos['y'])), int(radius), border_width)
            
            if self.font:
                text_surface = self.font.render(str(troops), True, self.ui_colors.get('black',(0,0,0)))
                text_rect = text_surface.get_rect(center=(pos['x'], pos['y']))
                surface.blit(text_surface, text_rect)
        
        for attack_data in self.game.get_ongoing_attacks():
            source_pos = self.game.node_positions.get(attack_data['source'])
            target_pos = self.game.node_positions.get(attack_data['target'])
            if not source_pos or not target_pos: continue

            progress = attack_data['progress']
            ball_x = source_pos['x'] + (target_pos['x'] - source_pos['x']) * progress
            ball_y = source_pos['y'] + (target_pos['y'] - source_pos['y']) * progress
            
            player_color_val = attack_data['player'] # This is player.value (string)
            ball_color = self.ui_colors.get(player_color_val, self.ui_colors.get('black',(0,0,0)))
            pygame.draw.circle(surface, ball_color, (int(ball_x), int(ball_y)), 4)
            pygame.draw.circle(surface, self.ui_colors.get('black',(0,0,0)), (int(ball_x), int(ball_y)), 4, 1)

    def _render_ui_panel(self, surface: pygame.Surface):
        if not hasattr(self, 'ui_colors') or not self.ui_colors or not self.font: self._init_render_if_needed()
        if not self.font: return # Cannot render UI without font

        ui_start_x = 800; ui_width = 400; window_height = 800
        
        ui_rect = pygame.Rect(ui_start_x, 0, ui_width, window_height)
        pygame.draw.rect(surface, self.ui_colors.get('light_gray',(240,240,240)), ui_rect)
        pygame.draw.line(surface, self.ui_colors.get('black',(0,0,0)), (ui_start_x, 0), (ui_start_x, window_height), 2)
        
        y_offset = 20
        title = self.large_font.render(f"Agent: {self.agent_player.value}", True, self.ui_colors.get('black',(0,0,0)))
        surface.blit(title, (ui_start_x + 20, y_offset)); y_offset += 40
        
        tick_info = self.font.render(f"Tick: {self.game.current_tick} / {GameConfig.MAX_EPISODE_TICKS}", True, self.ui_colors.get('black',(0,0,0)))
        surface.blit(tick_info, (ui_start_x + 20, y_offset)); y_offset += 25
        step_info = self.font.render(f"Agent Step: {self.current_agent_steps} / {GameConfig.MAX_AGENT_STEPS_PER_EPISODE}", True, self.ui_colors.get('black',(0,0,0)))
        surface.blit(step_info, (ui_start_x + 20, y_offset)); y_offset += 30

        scores_title = self.font.render("Node Scores:", True, self.ui_colors.get('black',(0,0,0)))
        surface.blit(scores_title, (ui_start_x + 20, y_offset)); y_offset += 30
        
        scores = self.game.get_scores()
        total_map_nodes = len(self.game.graph.nodes()) if self.game.graph else 0
        for p_enum, score in scores.items():
            color = self.ui_colors.get(p_enum.value, self.ui_colors.get('black',(0,0,0)))
            text = self.font.render(f"{p_enum.value}: {score}/{total_map_nodes}", True, color)
            surface.blit(text, (ui_start_x + 30, y_offset)); y_offset += 20
        y_offset += 10

        troops_title = self.font.render("Troop Counts:", True, self.ui_colors.get('black',(0,0,0)))
        surface.blit(troops_title, (ui_start_x + 20, y_offset)); y_offset += 30
        troop_counts = self.game.get_player_troop_counts()
        for p_enum, troops in troop_counts.items():
            color = self.ui_colors.get(p_enum.value, self.ui_colors.get('black',(0,0,0)))
            text = self.font.render(f"{p_enum.value}: {troops}", True, color)
            surface.blit(text, (ui_start_x + 30, y_offset)); y_offset += 20
        y_offset += 20

        if self.game.game_over:
            winner = self.game.get_winner()
            win_text = f"{winner.value.upper()} WINS!" if winner else "GAME OVER - NO WINNER"
            win_color = self.ui_colors.get(winner.value, self.ui_colors.get('black',(0,0,0))) if winner else self.ui_colors.get('black',(0,0,0))
            game_over_msg = self.large_font.render(win_text, True, win_color)
            surface.blit(game_over_msg, (ui_start_x + 20, y_offset + 50))

    def _render_frame(self):
        if self.render_mode not in ["human", "rgb_array"] or not PYGAME_AVAILABLE:
            return None
        
        self._init_render_if_needed() # Ensure Pygame elements like fonts are ready
        if self.render_mode == "human" and self.window_surface is None: return None # Init failed
        if self.font is None: return None # Cannot render UI without font

        canvas_width = 1200; canvas_height = 800
        canvas = pygame.Surface((canvas_width, canvas_height))
        canvas.fill(self.ui_colors.get('white', (255,255,255)))

        game_board_surface = canvas.subsurface(pygame.Rect(0, 0, 800, 800)) # Game area
        self._render_game_board(game_board_surface)

        ui_panel_surface = canvas.subsurface(pygame.Rect(800, 0, 400, 800)) # UI area
        self._render_ui_panel(ui_panel_surface)

        if self.render_mode == "human":
            self.window_surface.blit(canvas, (0,0))
            pygame.event.pump() # Process event queue
            pygame.display.flip()
            if self.clock: self.clock.tick(self.metadata['render_fps'])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))
        return None

    def render(self):
        return self._render_frame()

    def close(self):
        if self.window_surface is not None and PYGAME_AVAILABLE:
            pygame.display.quit()
            pygame.font.quit() # Quit font module
            pygame.quit() # Quit Pygame itself
            self.window_surface = None
            self.clock = None
            self.font = None; self.small_font = None; self.large_font = None
            self.ui_colors = {}