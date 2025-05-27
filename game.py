import random
import networkx as nx
import time
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional

class Player(Enum):
    GREY = "grey"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"

class Attack:
    def __init__(self, player, source, target, troops, completion_time):
        self.player = player
        self.source = source
        self.target = target
        self.troops = troops
        self.completion_time = completion_time

class ConquestGame:
    def __init__(self):
        self.bots = [Player.RED, Player.GREEN, Player.YELLOW]
        self.human_player = Player.BLUE
        self.last_income_time = time.time()
        self.last_bot_move_time = time.time()
        self.ongoing_attacks = []
        self.node_positions = {}  # Store positions for distance calculation
        self.reset_game()
    
    def reset_game(self):
        # Generate random connected graph with more connections
        num_nodes = random.randint(15, 25)
        self.graph = nx.connected_watts_strogatz_graph(num_nodes, 6, 0.4)  # More connections
        
        # Calculate positions for distance
        self._calculate_positions()
        
        # Initialize node states
        self.node_owners = {node: Player.GREY for node in self.graph.nodes()}
        self.node_troops = {node: random.randint(0, 50) for node in self.graph.nodes()}
        
        # Assign starting nodes to players
        start_nodes = random.sample(list(self.graph.nodes()), 4)
        players = [Player.RED, Player.GREEN, Player.BLUE, Player.YELLOW]
        
        for node, player in zip(start_nodes, players):
            self.node_owners[node] = player
            self.node_troops[node] = 10
        
        self.game_over = False
        self.ongoing_attacks = []
        self.last_income_time = time.time()
        self.last_bot_move_time = time.time()
    
    def _calculate_positions(self):
        """Calculate fixed positions for distance calculations"""
        num_nodes = len(self.graph.nodes())
        radius = 300
        for i, node in enumerate(self.graph.nodes()):
            angle = (i / num_nodes) * 2 * math.pi
            self.node_positions[node] = {
                'x': 400 + radius * math.cos(angle),
                'y': 300 + radius * math.sin(angle)
            }
    
    def _get_distance(self, node1, node2):
        """Calculate Euclidean distance between nodes"""
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)
    
    def get_valid_moves(self, player: Player) -> List[Tuple[int, int]]:
        moves = []
        player_nodes = [n for n, owner in self.node_owners.items() if owner == player]
        
        for source in player_nodes:
            if self.node_troops[source] > 1:
                for target in self.graph.neighbors(source):
                    if self.node_owners[target] != player:
                        moves.append((source, target))
        return moves
    
    def make_move(self, player: Player, source: int, target: int, troops: int) -> bool:
        """Start an attack (with travel time)"""
        if (source, target) not in self.get_valid_moves(player):
            return False
        if troops < 1 or troops >= self.node_troops[source]:
            return False
        
        # Calculate travel time based on distance (speed: 100 pixels per second)
        distance = self._get_distance(source, target)
        travel_time = distance / 100  # seconds
        completion_time = time.time() + travel_time
        
        # Remove troops from source immediately
        self.node_troops[source] -= troops
        
        # Add to ongoing attacks
        attack = Attack(player, source, target, troops, completion_time)
        self.ongoing_attacks.append(attack)
        
        return True
    
    def _process_attacks(self):
        """Process completed attacks"""
        current_time = time.time()
        completed = []
        
        for i, attack in enumerate(self.ongoing_attacks):
            if current_time >= attack.completion_time:
                # Execute combat
                defending_troops = self.node_troops[attack.target]
                if attack.troops > defending_troops:
                    # Attacker wins
                    self.node_owners[attack.target] = attack.player
                    self.node_troops[attack.target] = attack.troops - defending_troops
                # If defender wins, troops are just lost
                completed.append(i)
        
        # Remove completed attacks (reverse order to maintain indices)
        for i in reversed(completed):
            del self.ongoing_attacks[i]
    
    def bot_random_move(self, bot: Player):
        valid_moves = self.get_valid_moves(bot)
        if not valid_moves:
            return
        
        source, target = random.choice(valid_moves)
        max_troops = self.node_troops[source] - 1
        troops = random.randint(1, max_troops)
        
        if self.make_move(bot, source, target, troops):
            distance = self._get_distance(source, target)
            travel_time = distance / 100
            print(f"{bot.value} attacks {target} from {source} with {troops} troops (ETA: {travel_time:.1f}s)")
    
    def update(self):
        current_time = time.time()
        
        # Process ongoing attacks
        self._process_attacks()
        
        # Add income every 5 seconds
        if current_time - self.last_income_time >= 5:
            self.add_income()
            self.last_income_time = current_time
            print("Income added!")
        
        # Bot moves every 10 seconds
        if current_time - self.last_bot_move_time >= 10:
            for bot in self.bots:
                if not self.game_over:
                    self.bot_random_move(bot)
            self.last_bot_move_time = current_time
        
        # Check win condition
        winner = self.get_winner()
        if winner:
            self.game_over = True
            print(f"Game Over! {winner.value} wins!")
    
    def add_income(self):
        for node, owner in self.node_owners.items():
            if owner != Player.GREY:
                self.node_troops[node] += 5
    
    def get_scores(self) -> Dict[Player, int]:
        scores = {player: 0 for player in Player if player != Player.GREY}
        for owner in self.node_owners.values():
            if owner != Player.GREY:
                scores[owner] += 1
        return scores
    
    def get_winner(self) -> Optional[Player]:
        scores = self.get_scores()
        if sum(scores.values()) == len(self.graph.nodes()):
            return max(scores, key=scores.get)
        return None
    
    def get_ongoing_attacks(self):
        """Get ongoing attacks for frontend visualization"""
        return [{
            'player': attack.player.value,
            'source': attack.source,
            'target': attack.target,
            'troops': attack.troops,
            'completion_time': attack.completion_time,
            'progress': min(1.0, (time.time() - (attack.completion_time - self._get_distance(attack.source, attack.target) / 100)) / (self._get_distance(attack.source, attack.target) / 100))
        } for attack in self.ongoing_attacks]