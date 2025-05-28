# game.py - Updated for modular bots and Gym Environment compatibility
import random
import networkx as nx
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any 
import numpy as np # For padded state

from config import GameConfig
# Import bot strategies - this assumes bots.py is in the same directory or accessible in PYTHONPATH
from bots import BotStrategy, RandomBot, AggressiveBot 

class Player(Enum):
    GREY = "grey"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"

    @classmethod
    def get_int_mapping(cls) -> Dict['Player', int]:
        return {player: i for i, player in enumerate(cls)}

    @classmethod
    def from_int(cls, value: int) -> Optional['Player']:
        mapping = {i: player for i, player in enumerate(cls)}
        return mapping.get(value)

    @classmethod
    def get_active_players(cls) -> List['Player']: # Non-Grey players
        return [p for p in cls if p != Player.GREY]

class Attack:
    def __init__(self, player: Player, source: int, target: int, troops: int, 
                 completion_tick: int, start_tick: int, is_transfer: bool = False):
        self.player = player
        self.source = source
        self.target = target
        self.troops = troops
        self.completion_tick = completion_tick
        self.start_tick = start_tick
        self.is_transfer = is_transfer

class ConquestGame:
    def __init__(self, width: int = 800, height: int = 600, 
                 bot_config: Optional[Dict[Player, Any]] = None):
        self.width = width
        self.height = height
        
        self.bot_config = bot_config if bot_config is not None else {}
        # Default unassigned active players to RandomBot if not in bot_config
        for p_enum in Player.get_active_players():
            if p_enum not in self.bot_config:
                # print(f"Player {p_enum.value} not in bot_config, defaulting to RandomBot.")
                self.bot_config[p_enum] = RandomBot()

        self.current_tick = 0
        self.last_income_tick = 0
        self.last_bot_move_tick = 0
        
        self.ongoing_attacks: List[Attack] = []
        self.node_positions: Dict[int, Dict[str, float]] = {}
        self.graph: nx.Graph = nx.Graph() # Initialize graph
        self.node_owners: Dict[int, Player] = {}
        self.node_troops: Dict[int, int] = {}
        self.game_over = False
        
        self.player_int_map = Player.get_int_mapping() # Store for convenience
        self.reset_game()
    
    def reset_game(self):
        num_nodes = random.randint(GameConfig.MIN_NODES, GameConfig.MAX_NODES)
        num_nodes = min(num_nodes, GameConfig.MAX_NODES) # Ensure not over max
        
        k_neighbors = min(num_nodes - 1 if num_nodes > 1 else 0, 6) # k < n for watts_strogatz
        if num_nodes <= k_neighbors and num_nodes > 0 : k_neighbors = num_nodes -1 # Ensure k < n
        elif num_nodes == 0: k_neighbors = 0

        if num_nodes > 0 :
            self.graph = nx.connected_watts_strogatz_graph(num_nodes, k_neighbors, 0.4)
        else:
            self.graph = nx.Graph() # Empty graph if num_nodes is 0

        self._calculate_positions()
        
        self.node_owners = {node: Player.GREY for node in self.graph.nodes()}
        self.node_troops = {node: random.randint(GameConfig.NEUTRAL_MIN_TROOPS, GameConfig.NEUTRAL_MAX_TROOPS) 
                           for node in self.graph.nodes()}
        
        active_players = Player.get_active_players()
        start_nodes_potential = list(self.graph.nodes())
        random.shuffle(start_nodes_potential)
        
        for i, player_enum in enumerate(active_players):
            if i < len(start_nodes_potential):
                node = start_nodes_potential[i]
                self.node_owners[node] = player_enum
                self.node_troops[node] = GameConfig.STARTING_TROOPS
            else:
                # Not enough nodes for all players, some might not start with a node
                # print(f"Warning: Not enough start nodes for player {player_enum.value}")
                break 
        
        self.game_over = False
        self.ongoing_attacks = []
        self.current_tick = 0
        self.last_income_tick = 0
        self.last_bot_move_tick = 0

    def _calculate_positions(self):
        num_nodes = len(self.graph.nodes())
        if num_nodes == 0: 
            self.node_positions = {}
            return
        radius = min(self.width, self.height) * 0.3
        center_x, center_y = self.width // 2, self.height // 2
        
        for i, node in enumerate(self.graph.nodes()):
            angle = (i / num_nodes) * 2 * math.pi
            self.node_positions[node] = {
                'x': center_x + radius * math.cos(angle),
                'y': center_y + radius * math.sin(angle)
            }
    
    def _get_distance(self, node1: int, node2: int) -> float:
        if node1 not in self.node_positions or node2 not in self.node_positions:
            return GameConfig.TRAVEL_SPEED * GameConfig.MAX_EPISODE_TICKS # A very large distance
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)
    
    def _calculate_travel_time(self, source: int, target: int) -> int:
        distance = self._get_distance(source, target)
        return max(1, int(distance / GameConfig.TRAVEL_SPEED))
    
    def _calculate_future_troops(self, node: int, ticks_in_future: int) -> int:
        current_troops = self.node_troops.get(node, 0)
        if self.node_owners.get(node) == Player.GREY:
            return current_troops
        
        ticks_since_last_income = self.current_tick - self.last_income_tick
        remaining_ticks_to_next_income = GameConfig.INCOME_INTERVAL - ticks_since_last_income
        
        future_troops = current_troops
        if ticks_in_future >= remaining_ticks_to_next_income:
            remaining_ticks_after_first_income = ticks_in_future - remaining_ticks_to_next_income
            income_cycles = 1 + (remaining_ticks_after_first_income // GameConfig.INCOME_INTERVAL)
            future_troops += (income_cycles * GameConfig.INCOME_PER_TICK)
        return future_troops

    def get_node_at_position(self, x: float, y: float) -> Optional[int]:
        for node_id in self.graph.nodes(): # Iterate over actual node IDs
            pos = self.node_positions.get(node_id)
            if not pos: continue
            troops = self.node_troops.get(node_id, 0)
            # Ensure troops is not negative for sqrt, though it shouldn't be.
            radius = max(20, math.sqrt(max(0, troops)) * 2.5) 
            if math.sqrt((x - pos['x'])**2 + (y - pos['y'])**2) <= radius:
                return node_id
        return None
    
    def get_valid_moves(self, player: Player) -> List[Tuple[int, int, bool]]:
        moves = []
        player_nodes = [n for n, owner in self.node_owners.items() if owner == player]
        for source in player_nodes:
            if self.node_troops.get(source, 0) > 0 and source in self.graph: # Check source in graph
                for target in self.graph.neighbors(source):
                    if target != source:
                        is_transfer = self.node_owners.get(target) == player
                        moves.append((source, target, is_transfer))
        return moves
    
    def get_valid_targets(self, source_node_id: int, for_player: Player) -> List[int]:
        if source_node_id is None or source_node_id not in self.graph:
            return []
        if self.node_owners.get(source_node_id) != for_player:
            return []
        valid_targets = []
        for target in self.graph.neighbors(source_node_id): # No need to check source_node_id in graph again
            if target != source_node_id:
                valid_targets.append(target)
        return valid_targets

    def make_move(self, player: Player, source: int, target: int, troops: int) -> bool:
        if not (source in self.graph and target in self.graph): return False
        if self.node_owners.get(source) != player: return False
        if target not in list(self.graph.neighbors(source)): return False
        if not (1 <= troops <= self.node_troops.get(source, 0)): return False
        
        travel_ticks = self._calculate_travel_time(source, target)
        completion_tick = self.current_tick + travel_ticks
        is_transfer = self.node_owners.get(target) == player
        
        self.node_troops[source] -= troops
        attack = Attack(player, source, target, troops, completion_tick, self.current_tick, is_transfer)
        self.ongoing_attacks.append(attack)
        return True
    
    def _process_attacks(self):
        completed_indices = []
        for i, attack in enumerate(self.ongoing_attacks):
            if self.current_tick >= attack.completion_tick:
                completed_indices.append(i)
                if attack.target not in self.graph: continue

                target_owner = self.node_owners.get(attack.target)
                target_troops = self.node_troops.get(attack.target, 0)

                if attack.is_transfer:
                    if target_owner == attack.player:
                        self.node_troops[attack.target] = target_troops + attack.troops
                else: # Combat
                    if target_owner == attack.player: 
                        self.node_troops[attack.target] = target_troops + attack.troops
                    elif attack.troops > target_troops:
                        self.node_owners[attack.target] = attack.player
                        self.node_troops[attack.target] = attack.troops - target_troops
                    else:
                        self.node_troops[attack.target] = max(0, target_troops - attack.troops)
        
        for i in sorted(completed_indices, reverse=True):
            del self.ongoing_attacks[i]
    
    def update(self) -> bool: # Returns True if game ended this tick
        if self.game_over:
            return True

        self.current_tick += 1
        self._process_attacks()
        
        if self.current_tick - self.last_income_tick >= GameConfig.INCOME_INTERVAL:
            self.add_income()
            self.last_income_tick = self.current_tick
        
        if self.current_tick - self.last_bot_move_tick >= GameConfig.BOT_MOVE_INTERVAL:
            if self.get_winner(): # Check before bots move
                self.game_over = True
                return True

            for player_enum in Player.get_active_players():
                if self.game_over: break

                if not any(owner == player_enum for owner in self.node_owners.values()):
                    continue # Player has no nodes, skip

                player_controller = self.bot_config.get(player_enum)

                if isinstance(player_controller, BotStrategy):
                    action = player_controller.get_action(self, player_enum)
                    if action:
                        source, target, troops_to_send = action
                        # print(f"Game: Bot {player_enum.value} ({type(player_controller).__name__}) action: {troops_to_send} from {source} to {target}")
                        self.make_move(player_enum, source, target, troops_to_send)
                elif player_controller == "human" or player_controller == "rl_agent":
                    # Externally controlled, game engine does nothing for this player here
                    pass 
                # else: player_controller is None or an unknown string, player does nothing

            self.last_bot_move_tick = self.current_tick
        
        if self.get_winner(): # Check after bot moves
            self.game_over = True
        
        return self.game_over
    
    def add_income(self):
        for node, owner in self.node_owners.items():
            if owner != Player.GREY:
                self.node_troops[node] = self.node_troops.get(node,0) + GameConfig.INCOME_PER_TICK
    
    def get_scores(self) -> Dict[Player, int]: # Node counts
        scores = {player: 0 for player in Player.get_active_players()}
        for owner in self.node_owners.values():
            if owner != Player.GREY:
                scores[owner] = scores.get(owner, 0) + 1
        return scores
    
    def get_player_troop_counts(self) -> Dict[Player, int]:
        player_troops = {player: 0 for player in Player.get_active_players()}
        for node, owner in self.node_owners.items():
            if owner != Player.GREY:
                player_troops[owner] = player_troops.get(owner, 0) + self.node_troops.get(node,0)
        return player_troops

    def get_total_troops_on_map(self) -> int:
        return sum(self.node_troops.get(n,0) for n in self.graph.nodes())

    def get_winner(self) -> Optional[Player]:
        player_troop_counts = self.get_player_troop_counts()
        total_troops_on_map = self.get_total_troops_on_map()

        if total_troops_on_map == 0: return None

        for player, troops in player_troop_counts.items():
            if (troops / total_troops_on_map) > 0.95:
                return player

        active_players_with_troops = [p for p, t in player_troop_counts.items() if t > 0]
        if len(active_players_with_troops) == 1:
            return active_players_with_troops[0]
        
        scores = self.get_scores() # Node counts
        players_with_nodes = [p for p, count in scores.items() if count > 0]
        if len(players_with_nodes) == 1:
            sole_player = players_with_nodes[0]
            # If only one player has nodes, and they also have all troops (or are the only one with troops)
            if player_troop_counts.get(sole_player, 0) == total_troops_on_map :
                return sole_player
        return None
    
    def get_ongoing_attacks(self) -> List[Dict[str, Any]]:
        result = []
        for attack in self.ongoing_attacks:
            total_travel_time = attack.completion_tick - attack.start_tick
            progress = 1.0
            if total_travel_time > 0:
                elapsed_time = self.current_tick - attack.start_tick
                progress = min(1.0, max(0.0, elapsed_time / total_travel_time))
            
            result.append({
                'player': attack.player.value, 'source': attack.source, 'target': attack.target,
                'troops': attack.troops, 'completion_tick': attack.completion_tick,
                'start_tick': attack.start_tick, 'is_transfer': attack.is_transfer,
                'progress': progress
            })
        return result

    def get_padded_state_for_agent(self) -> Dict[str, np.ndarray]:
        """ Provides a fixed-size, padded representation of the game state for an RL agent. """
        num_map_nodes = len(self.graph.nodes())
        grey_player_int_id = self.player_int_map[Player.GREY]

        obs_node_owners = np.full(GameConfig.MAX_NODES, grey_player_int_id, dtype=np.int32)
        obs_node_troops = np.zeros(GameConfig.MAX_NODES, dtype=np.float32)
        obs_adj_matrix = np.zeros((GameConfig.MAX_NODES, GameConfig.MAX_NODES), dtype=np.int8)
        
        # Assuming node IDs from Watts-Strogatz are 0..N-1.
        # If node IDs can be arbitrary and sparse, a mapping would be needed.
        # For now, direct indexing is used, assuming node_id < MAX_NODES.
        for node_id in self.graph.nodes():
            if node_id < GameConfig.MAX_NODES: # Ensure we don't write out of bounds
                 obs_node_owners[node_id] = self.player_int_map.get(self.node_owners.get(node_id, Player.GREY), grey_player_int_id)
                 obs_node_troops[node_id] = float(self.node_troops.get(node_id, 0))

        for u, v in self.graph.edges():
            if u < GameConfig.MAX_NODES and v < GameConfig.MAX_NODES: # Ensure within bounds
                obs_adj_matrix[u, v] = 1
                obs_adj_matrix[v, u] = 1
        
        return {
            "node_owners": obs_node_owners,
            "node_troops": obs_node_troops,
            "adjacency_matrix": obs_adj_matrix,
            "num_actual_nodes": np.array([num_map_nodes], dtype=np.float32)
        }