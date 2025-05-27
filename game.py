# game.py - Updated with new combat rules and strategic green bot
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
    def __init__(self, player, source, target, troops, completion_time, is_transfer=False):
        self.player = player
        self.source = source
        self.target = target
        self.troops = troops
        self.completion_time = completion_time
        self.is_transfer = is_transfer

class ConquestGame:
    def __init__(self):
        self.bots = [Player.RED, Player.GREEN, Player.YELLOW]
        self.human_player = Player.BLUE
        self.last_income_time = time.time()
        self.last_bot_move_time = time.time()
        self.ongoing_attacks = []
        self.node_positions = {}
        self.reset_game()
    
    def reset_game(self):
        num_nodes = random.randint(15, 25)
        self.graph = nx.connected_watts_strogatz_graph(num_nodes, 6, 0.4)
        
        self._calculate_positions()
        
        self.node_owners = {node: Player.GREY for node in self.graph.nodes()}
        self.node_troops = {node: random.randint(0, 50) for node in self.graph.nodes()}
        
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
        num_nodes = len(self.graph.nodes())
        radius = 300
        for i, node in enumerate(self.graph.nodes()):
            angle = (i / num_nodes) * 2 * math.pi
            self.node_positions[node] = {
                'x': 400 + radius * math.cos(angle),
                'y': 300 + radius * math.sin(angle)
            }
    
    def _get_distance(self, node1, node2):
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)
    
    def get_valid_moves(self, player: Player) -> List[Tuple[int, int, bool]]:
        moves = []
        player_nodes = [n for n, owner in self.node_owners.items() if owner == player]
        
        for source in player_nodes:
            if self.node_troops[source] > 0:
                for target in self.graph.neighbors(source):
                    if target != source:
                        is_transfer = self.node_owners[target] == player
                        moves.append((source, target, is_transfer))
        return moves
    
    def make_move(self, player: Player, source: int, target: int, troops: int) -> bool:
        valid_moves = self.get_valid_moves(player)
        move_found = False
        is_transfer = False
        
        for valid_source, valid_target, valid_is_transfer in valid_moves:
            if valid_source == source and valid_target == target:
                move_found = True
                is_transfer = valid_is_transfer
                break
        
        if not move_found:
            return False
        
        if troops < 1 or troops > self.node_troops[source]:
            return False
        
        distance = self._get_distance(source, target)
        travel_time = distance / 100
        completion_time = time.time() + travel_time
        
        self.node_troops[source] -= troops
        
        attack = Attack(player, source, target, troops, completion_time, is_transfer)
        self.ongoing_attacks.append(attack)
        
        return True
    
    def _process_attacks(self):
        current_time = time.time()
        completed = []
        
        for i, attack in enumerate(self.ongoing_attacks):
            if current_time >= attack.completion_time:
                if attack.is_transfer:
                    # Friendly transfer
                    self.node_troops[attack.target] += attack.troops
                else:
                    # Combat - NEW RULES
                    defending_troops = self.node_troops[attack.target]
                    if attack.troops > defending_troops:
                        # Attacker wins and conquers
                        self.node_owners[attack.target] = attack.player
                        self.node_troops[attack.target] = attack.troops - defending_troops
                    else:
                        # Defender keeps the node (even if troops == defending_troops)
                        self.node_troops[attack.target] = max(0, defending_troops - attack.troops)
                
                completed.append(i)
        
        for i in reversed(completed):
            del self.ongoing_attacks[i]
    
    def bot_green_strategic(self):
        """Green bot: Only attacks when it can guarantee conquest"""
        valid_moves = self.get_valid_moves(Player.GREEN)
        if not valid_moves:
            return
        
        # Find conquerable targets
        conquerable_moves = []
        for source, target, is_transfer in valid_moves:
            if not is_transfer:  # Only consider attacks, not transfers
                defending_troops = self.node_troops[target]
                available_troops = self.node_troops[source]
                troops_needed = defending_troops + 1  # Need to exceed defending troops
                
                if available_troops >= troops_needed:
                    conquerable_moves.append((source, target, troops_needed))
        
        if conquerable_moves:
            # Attack ALL conquerable targets with minimum required troops
            for source, target, troops_needed in conquerable_moves:
                if self.node_troops[source] >= troops_needed:  # Double-check we still have troops
                    self.make_move(Player.GREEN, source, target, troops_needed)
                    print(f"GREEN strategic attack: {target} from {source} with {troops_needed} troops (defender had {self.node_troops[target]} before income)")
    
    def bot_random_move(self, bot: Player):
        """Random strategy for RED and YELLOW"""
        valid_moves = self.get_valid_moves(bot)
        if not valid_moves:
            return
        
        source, target, is_transfer = random.choice(valid_moves)
        max_troops = self.node_troops[source]
        
        if max_troops == 0:
            return
            
        if not is_transfer and max_troops > 1:
            troops = random.randint(1, max_troops - 1)
        else:
            troops = random.randint(1, max_troops)
        
        if self.make_move(bot, source, target, troops):
            action = "transfers to" if is_transfer else "attacks"
            distance = self._get_distance(source, target)
            travel_time = distance / 100
            print(f"{bot.value} {action} {target} from {source} with {troops} troops (ETA: {travel_time:.1f}s)")
    
    def update(self):
        current_time = time.time()
        
        self._process_attacks()
        
        if current_time - self.last_income_time >= 5:
            self.add_income()
            self.last_income_time = current_time
            print("Income added!")
        
        if current_time - self.last_bot_move_time >= 10:
            for bot in self.bots:
                if not self.game_over:
                    if bot == Player.GREEN:
                        self.bot_green_strategic()
                    else:
                        self.bot_random_move(bot)
            self.last_bot_move_time = current_time
        
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
        return [{
            'player': attack.player.value,
            'source': attack.source,
            'target': attack.target,
            'troops': attack.troops,
            'completion_time': attack.completion_time,
            'is_transfer': attack.is_transfer,
            'progress': min(1.0, (time.time() - (attack.completion_time - self._get_distance(attack.source, attack.target) / 100)) / (self._get_distance(attack.source, attack.target) / 100))
        } for attack in self.ongoing_attacks]