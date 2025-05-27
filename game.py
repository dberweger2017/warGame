# game.py - Updated for Gym Environment
import random
import networkx as nx
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any # Added Any
import numpy as np # Added for padded state

# Assuming GameConfig is in a separate file as per user request
from config import GameConfig # Import GameConfig

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
    # ... (Attack class remains the same as your provided version) ...
    def __init__(self, player, source, target, troops, completion_tick, start_tick, is_transfer=False):
        self.player = player
        self.source = source
        self.target = target
        self.troops = troops
        self.completion_tick = completion_tick
        self.start_tick = start_tick
        self.is_transfer = is_transfer

class ConquestGame:
    def __init__(self, width=800, height=600, agent_player: Optional[Player] = None): # Added agent_player
        self.width = width
        self.height = height
        
        # Define bots, excluding the agent if specified
        self.agent_player = agent_player
        self.bots = [p for p in [Player.RED, Player.GREEN, Player.YELLOW] if p != self.agent_player]
        # self.human_player = Player.BLUE # This might be the agent or another bot

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
        # Ensure num_nodes doesn't exceed MAX_NODES for consistency
        num_nodes = min(num_nodes, GameConfig.MAX_NODES)
        
        # Ensure k < n for watts_strogatz_graph
        k_neighbors = min(num_nodes -1 if num_nodes > 1 else 0, 6)
        self.graph = nx.connected_watts_strogatz_graph(num_nodes, k_neighbors, 0.4)
        
        self._calculate_positions()
        
        self.node_owners = {node: Player.GREY for node in self.graph.nodes()}
        self.node_troops = {node: random.randint(GameConfig.NEUTRAL_MIN_TROOPS, GameConfig.NEUTRAL_MAX_TROOPS) 
                           for node in self.graph.nodes()}
        
        # Assign starting nodes to all active players (Red, Green, Blue, Yellow)
        active_players = Player.get_active_players()
        
        # Ensure there are enough nodes for starting players
        if num_nodes < len(active_players):
            # This case should ideally be handled by setting MIN_NODES appropriately
            # For now, we'll assign to as many as possible
            print(f"Warning: Not enough nodes ({num_nodes}) for all active players ({len(active_players)}). Some may not get a starting node.")
        
        start_nodes_potential = list(self.graph.nodes())
        random.shuffle(start_nodes_potential)
        
        for i, player in enumerate(active_players):
            if i < len(start_nodes_potential):
                node = start_nodes_potential[i]
                self.node_owners[node] = player
                self.node_troops[node] = GameConfig.STARTING_TROOPS
            else:
                break # No more nodes to assign
        
        self.game_over = False
        self.ongoing_attacks = []
        self.current_tick = 0
        self.last_income_tick = 0
        self.last_bot_move_tick = 0
        # print(f"Game reset. Agent: {self.agent_player}. Bots: {[b.value for b in self.bots]}. Nodes: {len(self.graph.nodes())}")

    # ... (_calculate_positions, _get_distance, _calculate_travel_time, _calculate_future_troops unchanged) ...
    def _calculate_positions(self):
        num_nodes = len(self.graph.nodes())
        if num_nodes == 0: return # Avoid division by zero if graph is empty
        radius = min(self.width, self.height) * 0.3
        center_x, center_y = self.width // 2, self.height // 2
        
        for i, node in enumerate(self.graph.nodes()):
            angle = (i / num_nodes) * 2 * math.pi
            self.node_positions[node] = {
                'x': center_x + radius * math.cos(angle),
                'y': center_y + radius * math.sin(angle)
            }
    
    def _get_distance(self, node1, node2):
        # Ensure nodes exist in positions, otherwise return a large distance
        if node1 not in self.node_positions or node2 not in self.node_positions:
            # print(f"Warning: Node position not found for {node1} or {node2} in _get_distance.")
            return GameConfig.TRAVEL_SPEED * GameConfig.MAX_EPISODE_TICKS # A very large distance
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)
    
    def _calculate_travel_time(self, source, target):
        """Calculate travel time in ticks"""
        distance = self._get_distance(source, target)
        return max(1, int(distance / GameConfig.TRAVEL_SPEED))
    
    def _calculate_future_troops(self, node, ticks_in_future):
        """Calculate how many troops a node will have after given ticks"""
        current_troops = self.node_troops.get(node, 0) # Use .get for safety
        
        if self.node_owners.get(node) == Player.GREY: # Use .get for safety
            return current_troops
        
        ticks_since_last_income = self.current_tick - self.last_income_tick
        remaining_ticks_to_next_income = GameConfig.INCOME_INTERVAL - ticks_since_last_income
        
        if ticks_in_future >= remaining_ticks_to_next_income:
            remaining_ticks = ticks_in_future - remaining_ticks_to_next_income
            income_cycles = 1 + (remaining_ticks // GameConfig.INCOME_INTERVAL)
            future_troops = current_troops + (income_cycles * GameConfig.INCOME_PER_TICK)
        else:
            future_troops = current_troops
        return future_troops

    def get_node_at_position(self, x, y): # Unchanged
        for node_id in self.graph.nodes():
            pos = self.node_positions.get(node_id)
            if not pos: continue
            # Ensure node_troops has entry for node_id, default to 0
            troops = self.node_troops.get(node_id, 0)
            radius = max(20, math.sqrt(troops) * 2.5)
            if math.sqrt((x - pos['x'])**2 + (y - pos['y'])**2) <= radius:
                return node_id
        return None
    
    def get_valid_moves(self, player: Player) -> List[Tuple[int, int, bool]]: # Unchanged
        moves = []
        player_nodes = [n for n, owner in self.node_owners.items() if owner == player]
        
        for source in player_nodes:
            if self.node_troops.get(source, 0) > 0:
                # Ensure source is in graph before calling neighbors
                if source in self.graph:
                    for target in self.graph.neighbors(source):
                        if target != source:
                            is_transfer = self.node_owners.get(target) == player
                            moves.append((source, target, is_transfer))
        return moves
    
    def get_valid_targets(self, source_node_id: int, for_player: Player) -> List[int]:
        """Get valid target nodes for a given source for a specific player."""
        if source_node_id is None or source_node_id not in self.graph:
            return []
        
        if self.node_owners.get(source_node_id) != for_player:
            return []
            
        valid_targets = []
        if source_node_id in self.graph: # Check if source_node_id is in graph
            for target in self.graph.neighbors(source_node_id):
                if target != source_node_id:
                    valid_targets.append(target)
        return valid_targets

    def make_move(self, player: Player, source: int, target: int, troops: int) -> bool: # Mostly unchanged
        # Validate the move
        if not (source in self.graph and target in self.graph):
            # print(f"Move failed: Source {source} or Target {target} not in graph.")
            return False
        if self.node_owners.get(source) != player:
            # print(f"Move failed: Player {player.value} does not own source {source} (Owner: {self.node_owners.get(source)})")
            return False
        if target not in list(self.graph.neighbors(source)):
            # print(f"Move failed: Target {target} not neighbor of {source}.")
            return False
        if troops < 1 or troops > self.node_troops.get(source, 0):
            # print(f"Move failed: Invalid troops {troops} for source {source} (has {self.node_troops.get(source,0)}).")
            return False
        
        travel_ticks = self._calculate_travel_time(source, target)
        completion_tick = self.current_tick + travel_ticks
        is_transfer = self.node_owners.get(target) == player
        
        self.node_troops[source] -= troops
        attack = Attack(player, source, target, troops, completion_tick, self.current_tick, is_transfer)
        self.ongoing_attacks.append(attack)
        # print(f"Move initiated by {player.value}: {troops} from {source} to {target}. Arrives tick {completion_tick}.")
        return True
    
    def _process_attacks(self): # Mostly unchanged, ensure .get for safety
        completed_indices = []
        for i, attack in enumerate(self.ongoing_attacks):
            if self.current_tick >= attack.completion_tick:
                completed_indices.append(i)
                if attack.target not in self.graph: continue # Target node might have been removed (edge case)

                target_owner = self.node_owners.get(attack.target)
                target_troops = self.node_troops.get(attack.target, 0)

                if attack.is_transfer:
                    if target_owner == attack.player: # Still owner
                        self.node_troops[attack.target] = target_troops + attack.troops
                    # else: troops are lost if target changed owner
                else: # Combat
                    if target_owner == attack.player: # Target became friendly (e.g. captured by another of our attacks)
                        self.node_troops[attack.target] = target_troops + attack.troops
                    elif attack.troops > target_troops: # Attacker wins
                        # old_owner_val = target_owner.value if target_owner else "N/A"
                        self.node_owners[attack.target] = attack.player
                        self.node_troops[attack.target] = attack.troops - target_troops
                        # print(f"Node {attack.target} captured by {attack.player.value} from {old_owner_val}")
                    else: # Defender wins or holds
                        self.node_troops[attack.target] = max(0, target_troops - attack.troops)
        
        for i in sorted(completed_indices, reverse=True):
            del self.ongoing_attacks[i]

    def bot_green_strategic(self): # Unchanged, but will only be called if Green is not agent
        player = Player.GREEN
        # This check is now implicitly handled by self.bots list in update()
        # if player == self.agent_player: return

        valid_moves = self.get_valid_moves(player)
        if not valid_moves: return
        
        conquerable_moves = []
        for source, target, is_transfer in valid_moves:
            if not is_transfer and self.node_owners.get(target) != player:
                travel_ticks = self._calculate_travel_time(source, target)
                future_defending_troops = self._calculate_future_troops(target, travel_ticks)
                troops_needed = future_defending_troops + 1
                if self.node_troops.get(source,0) >= troops_needed:
                    conquerable_moves.append((source, target, troops_needed))
        
        if conquerable_moves:
            # Simple: attack one random conquerable target
            # For more deterministic behavior, could sort and pick best
            source, target, troops_needed = random.choice(conquerable_moves)
            if self.node_troops.get(source,0) >= troops_needed: # Re-check
                 self.make_move(player, source, target, troops_needed)
    
    def bot_random_move(self, bot_player: Player): # Unchanged, but only for non-agent bots
        # if bot_player == self.agent_player: return

        valid_moves = self.get_valid_moves(bot_player)
        if not valid_moves: return
        
        source, target, is_transfer = random.choice(valid_moves)
        max_troops = self.node_troops.get(source,0)
        if max_troops == 0: return
            
        troops_to_send = 0
        if is_transfer:
            if max_troops > 0: troops_to_send = random.randint(1, max_troops)
        else: # Attacking
            if max_troops > 1: troops_to_send = random.randint(1, max_troops -1) # Try to leave one
            elif max_troops == 1: troops_to_send = 1
        
        if troops_to_send > 0 : self.make_move(bot_player, source, target, troops_to_send)
    
    def update(self) -> bool: # Returns True if game ended this tick
        """Update game state by one tick. Manages income, attack processing, and non-agent bot moves."""
        if self.game_over:
            return True

        self.current_tick += 1
        self._process_attacks()
        
        if self.current_tick - self.last_income_tick >= GameConfig.INCOME_INTERVAL:
            self.add_income()
            self.last_income_tick = self.current_tick
        
        # Non-agent Bot moves at intervals
        if self.current_tick - self.last_bot_move_tick >= GameConfig.BOT_MOVE_INTERVAL:
            # Check for winner before bots move, to prevent moves after game ends
            if self.get_winner():
                self.game_over = True
                return True

            for bot_player in self.bots: # self.bots now excludes agent_player
                if not self.game_over:
                    if any(owner == bot_player for owner in self.node_owners.values()): # If bot still in game
                        if bot_player == Player.GREEN: # Assuming Green has specific logic
                            self.bot_green_strategic()
                        else: # Other bots are random
                            self.bot_random_move(bot_player)
            self.last_bot_move_tick = self.current_tick
        
        if self.get_winner():
            self.game_over = True
        
        return self.game_over
    
    def add_income(self): # Unchanged
        for node, owner in self.node_owners.items():
            if owner != Player.GREY:
                self.node_troops[node] = self.node_troops.get(node,0) + GameConfig.INCOME_PER_TICK
    
    def get_scores(self) -> Dict[Player, int]: # Node counts, unchanged
        scores = {player: 0 for player in Player.get_active_players()}
        for owner in self.node_owners.values():
            if owner != Player.GREY:
                scores[owner] = scores.get(owner, 0) + 1
        return scores
    
    def get_player_troop_counts(self) -> Dict[Player, int]: # Unchanged
        player_troops = {player: 0 for player in Player.get_active_players()}
        for node, owner in self.node_owners.items():
            if owner != Player.GREY:
                player_troops[owner] = player_troops.get(owner, 0) + self.node_troops.get(node,0)
        return player_troops

    def get_total_troops_on_map(self) -> int: # Unchanged
        return sum(self.node_troops.get(n,0) for n in self.graph.nodes())

    def get_winner(self) -> Optional[Player]: # Unchanged (using troop majority)
        player_troop_counts = self.get_player_troop_counts()
        total_troops_on_map = self.get_total_troops_on_map()

        if total_troops_on_map == 0: return None

        for player, troops in player_troop_counts.items():
            if (troops / total_troops_on_map) > 0.95:
                return player

        active_players_with_troops = [p for p, t in player_troop_counts.items() if t > 0]
        if len(active_players_with_troops) == 1:
            return active_players_with_troops[0]
        
        scores = self.get_scores()
        players_with_nodes = [p for p, count in scores.items() if count > 0]
        if len(players_with_nodes) == 1:
            sole_player = players_with_nodes[0]
            if player_troop_counts.get(sole_player, 0) == total_troops_on_map:
                return sole_player
        return None
    
    def get_ongoing_attacks(self): # Unchanged
        result = []
        for attack in self.ongoing_attacks:
            total_travel_time = attack.completion_tick - attack.start_tick
            if total_travel_time <= 0: progress = 1.0
            else:
                elapsed_time = self.current_tick - attack.start_tick
                progress = min(1.0, max(0.0, elapsed_time / total_travel_time))
            
            result.append({
                'player': attack.player.value, 'source': attack.source, 'target': attack.target,
                'troops': attack.troops, 'completion_tick': attack.completion_tick,
                'start_tick': attack.start_tick, 'is_transfer': attack.is_transfer,
                'progress': progress
            })
        return result

    # --- New method for Gym Environment Observation ---
    def get_padded_state_for_agent(self) -> Dict[str, np.ndarray]:
        """
        Provides a fixed-size, padded representation of the game state.
        """
        num_map_nodes = len(self.graph.nodes())

        # Node owners (integer IDs)
        # Initialize with Grey player's int ID
        grey_player_int_id = self.player_int_map[Player.GREY]
        obs_node_owners = np.full(GameConfig.MAX_NODES, grey_player_int_id, dtype=np.int32)
        for i, node_id in enumerate(self.graph.nodes()): # Iterate in graph's node order
            if i < GameConfig.MAX_NODES: # Should always be true if num_map_nodes <= MAX_NODES
                 obs_node_owners[node_id] = self.player_int_map[self.node_owners.get(node_id, Player.GREY)]


        # Node troops (float for normalization later if needed, or int)
        obs_node_troops = np.zeros(GameConfig.MAX_NODES, dtype=np.float32)
        for i, node_id in enumerate(self.graph.nodes()):
            if i < GameConfig.MAX_NODES:
                obs_node_troops[node_id] = float(self.node_troops.get(node_id, 0))
        
        # Adjacency matrix (binary)
        obs_adj_matrix = np.zeros((GameConfig.MAX_NODES, GameConfig.MAX_NODES), dtype=np.int8)
        # Map graph node IDs to 0..MAX_NODES-1 indices if they are not already
        # For simplicity, assuming node IDs from Watts-Strogatz are 0..N-1
        for u, v in self.graph.edges():
            if u < GameConfig.MAX_NODES and v < GameConfig.MAX_NODES:
                obs_adj_matrix[u, v] = 1
                obs_adj_matrix[v, u] = 1
        
        return {
            "node_owners": obs_node_owners,
            "node_troops": obs_node_troops,
            "adjacency_matrix": obs_adj_matrix,
            "num_actual_nodes": np.array([num_map_nodes], dtype=np.float32) # Store actual num nodes
        }