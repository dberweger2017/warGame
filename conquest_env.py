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

    def __init__(self, agent_player_id: Player = Player.BLUE, render_mode: Optional[str] = None):
        super().__init__()

        self.agent_player = agent_player_id
        # Game instance from game.py, configured for the agent
        self.game = ConquestGame(agent_player=self.agent_player) 
        
        self.player_int_map = Player.get_int_mapping()
        self.int_player_map = {v: k for k,v in self.player_int_map.items()}
        self.active_players_list = Player.get_active_players() # For consistent indexing in obs

        # Action space: (source_node_idx, target_node_idx, troop_choice_idx)
        # troop_choice_idx: 0=25%, 1=50%, 2=75%, 3=100% (of sendable troops)
        # 4=No-Op (or pass turn)
        self.NUM_TROOP_CHOICES = 4 
        self.NO_OP_ACTION_IDX = self.NUM_TROOP_CHOICES # An additional action for NO-OP
        
        self.action_space = spaces.MultiDiscrete([
            GameConfig.MAX_NODES,         # Source node index
            GameConfig.MAX_NODES,         # Target node index
            self.NUM_TROOP_CHOICES + 1    # Troop choice (0-3 for percentages, 4 for NO_OP)
        ])

        # Observation space (using Dict)
        _player_ids_len = len(self.player_int_map)
        _active_player_ids_len = len(self.active_players_list)

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
            self.render_mode = None
        
        self.current_agent_steps = 0


    def _get_obs(self) -> Dict[str, Any]:
        padded_state = self.game.get_padded_state_for_agent()

        agent_id_onehot = np.zeros(len(self.player_int_map), dtype=np.int8)
        agent_id_onehot[self.player_int_map[self.agent_player]] = 1

        tick_norm = np.array([min(1.0, self.game.current_tick / GameConfig.MAX_EPISODE_TICKS)], dtype=np.float32)

        player_troop_counts = self.game.get_player_troop_counts()
        player_node_counts = self.game.get_scores() # get_scores returns node counts
        total_troops = max(1, self.game.get_total_troops_on_map())
        
        # Use actual number of nodes from the game for node ratio calculation
        num_actual_nodes_in_game = len(self.game.graph.nodes())
        total_map_nodes = max(1, num_actual_nodes_in_game)


        troop_ratios = np.zeros(len(self.active_players_list), dtype=np.float32)
        node_ratios = np.zeros(len(self.active_players_list), dtype=np.float32)

        for i, p_enum in enumerate(self.active_players_list):
            troop_ratios[i] = player_troop_counts.get(p_enum, 0) / total_troops
            node_ratios[i] = player_node_counts.get(p_enum, 0) / total_map_nodes
            
        num_actual_nodes_norm_val = padded_state["num_actual_nodes"][0] / GameConfig.MAX_NODES

        return {
            "node_owners": padded_state["node_owners"],
            "node_troops": padded_state["node_troops"], # Consider normalizing troops
            "adjacency_matrix": padded_state["adjacency_matrix"],
            "agent_player_id_onehot": agent_id_onehot,
            "current_tick_norm": tick_norm,
            "player_troop_ratios": troop_ratios,
            "player_node_ratios": node_ratios,
            "num_actual_nodes_norm": np.array([num_actual_nodes_norm_val], dtype=np.float32)
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_tick": self.game.current_tick,
            "agent_player_enum": self.agent_player,
            "agent_troops": self.game.get_player_troop_counts().get(self.agent_player, 0),
            "agent_nodes": self.game.get_scores().get(self.agent_player, 0),
            "winner": self.game.get_winner(),
            "num_actual_nodes": len(self.game.graph.nodes())
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed) # For any np randomness if used directly

        self.game.reset_game() # Resets the underlying game engine
        self.current_agent_steps = 0
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._init_render_if_needed()
            self._render_frame()
            
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.current_agent_steps += 1
        
        source_idx, target_idx, troop_choice_idx = action
        reward = 0.0
        terminated = False # Game ended based on win/loss condition
        truncated = False  # Episode ended due to time limit or step limit

        # --- 1. Interpret and Validate Agent's Action ---
        action_penalty = 0.0
        num_actual_nodes_in_game = len(self.game.graph.nodes())

        if troop_choice_idx == self.NO_OP_ACTION_IDX:
            # Valid No-Op, no action taken by agent this step
            pass # reward += 0.001 # Tiny reward for valid no-op if desired
        elif not (0 <= source_idx < num_actual_nodes_in_game and 0 <= target_idx < num_actual_nodes_in_game):
            action_penalty -= 1.0 # Invalid node index
        elif self.game.node_owners.get(source_idx) != self.agent_player:
            action_penalty -= 0.5 # Source not owned by agent
        elif source_idx == target_idx:
            action_penalty -= 0.5 # Source and target are same
        elif target_idx not in self.game.graph.neighbors(source_idx):
            action_penalty -= 0.5 # Target not a neighbor
        else:
            available_troops = self.game.node_troops.get(source_idx, 0)
            if available_troops > 0:
                percentages = [0.25, 0.50, 0.75, 1.00] # For troop_choice_idx 0-3
                chosen_percentage = percentages[troop_choice_idx]
                
                # Determine max sendable (leave 1 if attacking, unless only 1 exists or it's a transfer)
                is_transfer_action = self.game.node_owners.get(target_idx) == self.agent_player
                max_sendable = available_troops
                if not is_transfer_action and available_troops > 1: # Attacking
                    max_sendable = available_troops -1 # Try to leave one troop
                
                troops_to_send = math.floor(max_sendable * chosen_percentage)
                troops_to_send = max(1, troops_to_send) if max_sendable > 0 else 0


                if troops_to_send > 0 and troops_to_send <= available_troops :
                    if not self.game.make_move(self.agent_player, source_idx, target_idx, troops_to_send):
                        action_penalty -= 0.1 # make_move failed for other internal reasons
                    # else: reward += 0.01 # Small reward for valid successful move initiation
                else: # Not enough troops for this action or zero troops to send
                     action_penalty -= 0.2 
            else: # No troops at source
                action_penalty -= 0.5
        
        reward += action_penalty

        # --- 2. Store pre-simulation state for reward calculation ---
        prev_agent_troops = self.game.get_player_troop_counts().get(self.agent_player, 0)
        prev_agent_nodes = self.game.get_scores().get(self.agent_player, 0)

        # --- 3. Advance Game State by AGENT_STEP_INTERVAL_TICKS ---
        game_ended_during_sim = False
        for _ in range(GameConfig.AGENT_STEP_INTERVAL_TICKS):
            if self.game.update(): # game.update() returns True if game ended that tick
                game_ended_during_sim = True
                break
        
        terminated = self.game.game_over or game_ended_during_sim

        # --- 4. Calculate Reward based on state change ---
        current_agent_troops = self.game.get_player_troop_counts().get(self.agent_player, 0)
        current_agent_nodes = self.game.get_scores().get(self.agent_player, 0)

        reward += (current_agent_troops - prev_agent_troops) * 0.05  # Scaler for troop changes
        reward += (current_agent_nodes - prev_agent_nodes) * 0.5    # Scaler for node changes
        
        winner = self.game.get_winner()
        if terminated:
            if winner == self.agent_player:
                reward += 50.0
            elif winner is not None: # Someone else won
                reward -= 50.0
            # else: No specific reward for draw if game ends without winner (e.g. mutual elimination)

        # --- 5. Check for Truncation ---
        if self.current_agent_steps >= GameConfig.MAX_AGENT_STEPS_PER_EPISODE:
            truncated = True
        if self.game.current_tick >= GameConfig.MAX_EPISODE_TICKS:
            truncated = True
            if not terminated: # Penalty for running out of time without winning
                 reward -= 5.0 

        # If agent has no troops or nodes, it's effectively out.
        if not terminated and (current_agent_troops == 0 and current_agent_nodes == 0):
            terminated = True # Agent eliminated
            reward -= 25.0 # Penalty for being eliminated

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info

    def _init_render_if_needed(self):
        if self.render_mode == "human" and PYGAME_AVAILABLE and self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window_surface = pygame.display.set_mode((1200, 800)) # Match main.py window
            pygame.display.set_caption(f"Conquest Gym Env - Agent: {self.agent_player.value}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.large_font = pygame.font.Font(None, 36)
            # Colors from your main.py for consistency
            self.ui_colors = {
                'grey': (204, 204, 204), 'red': (255, 68, 68), 'green': (68, 255, 68),
                'blue': (68, 68, 255), 'yellow': (255, 255, 68), 'white': (255, 255, 255),
                'black': (0, 0, 0), 'dark_gray': (100, 100, 100), 'cyan': (0, 255, 255),
                'light_gray': (240, 240, 240)
            }

    def _render_game_board(self, surface: pygame.Surface):
        """Renders the game board area, adapted from your main.py's draw_game_board."""
        game_area_width = 800 # As in your main.py
        surface.fill(self.ui_colors['white']) # Fill game area

        # Draw edges
        for edge in self.game.graph.edges():
            source_pos = self.game.node_positions.get(edge[0])
            target_pos = self.game.node_positions.get(edge[1])
            if source_pos and target_pos:
                pygame.draw.line(surface, self.ui_colors['dark_gray'],
                                 (source_pos['x'], source_pos['y']),
                                 (target_pos['x'], target_pos['y']), 2)
        # Draw nodes
        for node_id in self.game.graph.nodes():
            pos = self.game.node_positions.get(node_id)
            if not pos: continue
            owner = self.game.node_owners.get(node_id, Player.GREY)
            troops = self.game.node_troops.get(node_id, 0)
            
            radius = max(15, math.sqrt(troops) * 2)
            color = self.ui_colors.get(owner.value, self.ui_colors['black'])
            
            border_color = self.ui_colors['black']
            border_width = 2
            if owner == self.agent_player: # Highlight agent nodes
                border_color = self.ui_colors['cyan']
                border_width = 3
                
            pygame.draw.circle(surface, color, (int(pos['x']), int(pos['y'])), int(radius))
            pygame.draw.circle(surface, border_color, (int(pos['x']), int(pos['y'])), int(radius), border_width)
            
            text = self.font.render(str(troops), True, self.ui_colors['black'])
            text_rect = text.get_rect(center=(pos['x'], pos['y']))
            surface.blit(text, text_rect)
        
        # Draw attack animations
        for attack_data in self.game.get_ongoing_attacks(): # Uses the method from game.py
            source_pos = self.game.node_positions.get(attack_data['source'])
            target_pos = self.game.node_positions.get(attack_data['target'])
            if not source_pos or not target_pos: continue

            progress = attack_data['progress']
            ball_x = source_pos['x'] + (target_pos['x'] - source_pos['x']) * progress
            ball_y = source_pos['y'] + (target_pos['y'] - source_pos['y']) * progress
            
            player_color_val = attack_data['player'] # This is player.value (string)
            ball_color = self.ui_colors.get(player_color_val, self.ui_colors['black'])
            pygame.draw.circle(surface, ball_color, (int(ball_x), int(ball_y)), 4)
            pygame.draw.circle(surface, self.ui_colors['black'], (int(ball_x), int(ball_y)), 4, 1)

    def _render_ui_panel(self, surface: pygame.Surface):
        """Renders a simplified UI panel, adapted from your main.py."""
        ui_start_x = 800 # As in your main.py
        ui_width = 400
        window_height = 800
        
        ui_rect = pygame.Rect(ui_start_x, 0, ui_width, window_height)
        pygame.draw.rect(surface, self.ui_colors['light_gray'], ui_rect)
        pygame.draw.line(surface, self.ui_colors['black'], (ui_start_x, 0), (ui_start_x, window_height), 2)
        
        y_offset = 20
        title = self.large_font.render(f"Agent: {self.agent_player.value}", True, self.ui_colors['black'])
        surface.blit(title, (ui_start_x + 20, y_offset))
        y_offset += 40
        
        tick_info = self.font.render(f"Tick: {self.game.current_tick} / {GameConfig.MAX_EPISODE_TICKS}", True, self.ui_colors['black'])
        surface.blit(tick_info, (ui_start_x + 20, y_offset)); y_offset += 25
        step_info = self.font.render(f"Agent Step: {self.current_agent_steps} / {GameConfig.MAX_AGENT_STEPS_PER_EPISODE}", True, self.ui_colors['black'])
        surface.blit(step_info, (ui_start_x + 20, y_offset)); y_offset += 30

        scores_title = self.font.render("Node Scores:", True, self.ui_colors['black'])
        surface.blit(scores_title, (ui_start_x + 20, y_offset)); y_offset += 30
        
        scores = self.game.get_scores()
        total_map_nodes = len(self.game.graph.nodes())
        for p_enum, score in scores.items():
            color = self.ui_colors.get(p_enum.value, self.ui_colors['black'])
            text = self.font.render(f"{p_enum.value}: {score}/{total_map_nodes}", True, color)
            surface.blit(text, (ui_start_x + 30, y_offset)); y_offset += 20
        y_offset += 10

        troops_title = self.font.render("Troop Counts:", True, self.ui_colors['black'])
        surface.blit(troops_title, (ui_start_x + 20, y_offset)); y_offset += 30
        troop_counts = self.game.get_player_troop_counts()
        for p_enum, troops in troop_counts.items():
            color = self.ui_colors.get(p_enum.value, self.ui_colors['black'])
            text = self.font.render(f"{p_enum.value}: {troops}", True, color)
            surface.blit(text, (ui_start_x + 30, y_offset)); y_offset += 20
        y_offset += 20

        if self.game.game_over:
            winner = self.game.get_winner()
            win_text = f"{winner.value.upper()} WINS!" if winner else "GAME OVER - NO WINNER"
            win_color = self.ui_colors.get(winner.value, self.ui_colors['black']) if winner else self.ui_colors['black']
            game_over_msg = self.large_font.render(win_text, True, win_color)
            surface.blit(game_over_msg, (ui_start_x + 20, y_offset + 50))


    def _render_frame(self):
        if not PYGAME_AVAILABLE: return None
        self._init_render_if_needed() # Ensure Pygame is initialized

        if self.render_mode == "human" and self.window_surface is None:
            # Should have been initialized by _init_render_if_needed
            return None 
        
        # Create a canvas for the entire window
        canvas_width = 1200 # Game (800) + UI (400)
        canvas_height = 800
        canvas = pygame.Surface((canvas_width, canvas_height))
        canvas.fill(self.ui_colors['white']) # Default background for areas not drawn

        # Game board area
        game_board_surface = canvas.subsurface(pygame.Rect(0, 0, 800, 800))
        self._render_game_board(game_board_surface)

        # UI Panel area
        ui_panel_surface = canvas.subsurface(pygame.Rect(800, 0, 400, 800))
        self._render_ui_panel(ui_panel_surface) # Pass the main canvas for UI drawing

        if self.render_mode == "human":
            self.window_surface.blit(canvas, (0,0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))


    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()
        return None # No rendering for other modes or if Pygame not available

    def close(self):
        if self.window_surface is not None and PYGAME_AVAILABLE:
            pygame.display.quit()
            pygame.font.quit()
            pygame.quit()
            self.window_surface = None
            self.clock = None
            self.font = None
            self.small_font = None
            self.large_font = None

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    print(f"Pygame Available: {PYGAME_AVAILABLE}")
    
    # Test with human rendering if available, otherwise headless
    # test_render_mode = "human" if PYGAME_AVAILABLE else None
    test_render_mode = "human" # Try human mode
    # test_render_mode = None # For headless speed test

    if test_render_mode == "human" and not PYGAME_AVAILABLE:
        print("Cannot run human render test, Pygame not found. Set test_render_mode=None for headless.")
        exit()

    env = ConquestEnv(agent_player_id=Player.BLUE, render_mode=test_render_mode)

    # Optional: Check with SB3 checker
    # from stable_baselines3.common.env_checker import check_env
    # try:
    #     check_env(env, warn=True) # warn=True to see warnings, skip_render_check=False
    #     print("Environment check passed!")
    # except Exception as e:
    #     print(f"Environment check failed: {e}")


    for episode in range(2): # Run a few episodes
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset(seed=42 + episode)
        # print("Initial Observation (keys):", list(obs.keys()))
        # print("Initial Info:", info)
        
        terminated = False
        truncated = False
        total_reward_acc = 0.0
        
        for step_num in range(GameConfig.MAX_AGENT_STEPS_PER_EPISODE + 5): # Run for max steps + a few
            action = env.action_space.sample() # Take a random action
            # print(f"\nStep {step_num + 1}, Action: {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # print(f"Reward: {reward:.2f}")
            # print(f"Terminated: {terminated}, Truncated: {truncated}")
            # print(f"Info (Tick {info['current_tick']}): Agent Troops: {info['agent_troops']}, Agent Nodes: {info['agent_nodes']}")
            total_reward_acc += reward

            if terminated or truncated:
                print(f"\nEpisode finished after {step_num + 1} agent steps (Game Tick: {info['current_tick']}).")
                print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
                print(f"Total reward: {total_reward_acc:.2f}")
                winner_player = info.get('winner')
                print(f"Winner: {winner_player.value if winner_player else 'None'}")
                break
        if not (terminated or truncated):
             print(f"Episode reached step limit without termination/truncation. Total reward: {total_reward_acc:.2f}")


    env.close()
    print("\nEnvironment test finished.")