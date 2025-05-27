# game.py - Updated Green bot with travel time calculation
import random
import networkx as nx
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional
from config import GameConfig

class Player(Enum):
    GREY = "grey"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"

class Attack:
    def __init__(self, player, source, target, troops, completion_tick, start_tick, is_transfer=False):
        self.player = player
        self.source = source
        self.target = target
        self.troops = troops
        self.completion_tick = completion_tick
        self.start_tick = start_tick
        self.is_transfer = is_transfer

class ConquestGame:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.bots = [Player.RED, Player.GREEN, Player.YELLOW]
        self.human_player = Player.BLUE
        
        # Tick-based timing
        self.current_tick = 0
        self.last_income_tick = 0
        self.last_bot_move_tick = 0
        
        self.ongoing_attacks = []
        self.node_positions = {}
        self.reset_game()
    
    def reset_game(self):
        num_nodes = random.randint(GameConfig.MIN_NODES, GameConfig.MAX_NODES)
        self.graph = nx.connected_watts_strogatz_graph(num_nodes, 6, 0.4)
        
        self._calculate_positions()
        
        self.node_owners = {node: Player.GREY for node in self.graph.nodes()}
        self.node_troops = {node: random.randint(GameConfig.NEUTRAL_MIN_TROOPS, GameConfig.NEUTRAL_MAX_TROOPS) 
                           for node in self.graph.nodes()}
        
        start_nodes = random.sample(list(self.graph.nodes()), 4)
        players = [Player.RED, Player.GREEN, Player.BLUE, Player.YELLOW]
        
        for node, player in zip(start_nodes, players):
            self.node_owners[node] = player
            self.node_troops[node] = GameConfig.STARTING_TROOPS
        
        self.game_over = False
        self.ongoing_attacks = []
        self.current_tick = 0
        self.last_income_tick = 0
        self.last_bot_move_tick = 0
    
    def _calculate_positions(self):
        num_nodes = len(self.graph.nodes())
        radius = min(self.width, self.height) * 0.3
        center_x, center_y = self.width // 2, self.height // 2
        
        for i, node in enumerate(self.graph.nodes()):
            angle = (i / num_nodes) * 2 * math.pi
            self.node_positions[node] = {
                'x': center_x + radius * math.cos(angle),
                'y': center_y + radius * math.sin(angle)
            }
    
    def _get_distance(self, node1, node2):
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)
    
    def _calculate_travel_time(self, source, target):
        """Calculate travel time in ticks"""
        distance = self._get_distance(source, target)
        return max(1, int(distance / GameConfig.TRAVEL_SPEED))
    
    def _calculate_future_troops(self, node, ticks_in_future):
        """Calculate how many troops a node will have after given ticks"""
        current_troops = self.node_troops[node]
        
        # Only player-owned nodes get income, not grey nodes
        if self.node_owners[node] == Player.GREY:
            return current_troops
        
        # Calculate how many income cycles will occur
        ticks_since_last_income = self.current_tick - self.last_income_tick
        remaining_ticks_to_next_income = GameConfig.INCOME_INTERVAL - ticks_since_last_income
        
        if ticks_in_future >= remaining_ticks_to_next_income:
            # At least one income cycle will occur
            remaining_ticks = ticks_in_future - remaining_ticks_to_next_income
            income_cycles = 1 + (remaining_ticks // GameConfig.INCOME_INTERVAL)
            future_troops = current_troops + (income_cycles * GameConfig.INCOME_PER_TICK)
        else:
            # No income cycles will occur
            future_troops = current_troops
            
        return future_troops
    
    def get_node_at_position(self, x, y):
        """Find node at given screen coordinates"""
        for node in self.graph.nodes():
            pos = self.node_positions[node]
            radius = max(20, math.sqrt(self.node_troops[node]) * 2.5) # Adjusted for better clickability
            if math.sqrt((x - pos['x'])**2 + (y - pos['y'])**2) <= radius:
                return node
        return None
    
    def get_valid_moves(self, player: Player) -> List[Tuple[int, int, bool]]:
        moves = []
        player_nodes = [n for n, owner in self.node_owners.items() if owner == player]
        
        for source in player_nodes:
            if self.node_troops[source] > 0: # Ensure source has troops
                for target in self.graph.neighbors(source):
                    if target != source:
                        is_transfer = self.node_owners[target] == player
                        moves.append((source, target, is_transfer))
        return moves
    
    def get_valid_targets(self, source):
        """Get valid target nodes for a given source"""
        if source is None:
            return []
        
        # Check if source is owned by blue player (or any player for general use)
        # For UI, it's specific to BLUE, but method could be more general
        if self.node_owners[source] != self.human_player: # Assuming human_player is BLUE
            return []
            
        valid_targets = []
        for target in self.graph.neighbors(source):
            if target != source:
                valid_targets.append(target)
        
        return valid_targets
    
    def make_move(self, player: Player, source: int, target: int, troops: int) -> bool:
        # Validate the move
        if source not in self.graph.nodes() or target not in self.graph.nodes():
            print(f"Error: Invalid node ID. Source: {source}, Target: {target}")
            return False
            
        if self.node_owners[source] != player:
            print(f"Error: Player {player.value} does not own source node {source}")
            return False
            
        if target not in list(self.graph.neighbors(source)):
            print(f"Error: Target node {target} is not a neighbor of source node {source}")
            return False
        
        if troops < 1:
            print(f"Error: Cannot send zero or negative troops ({troops})")
            return False
        if troops > self.node_troops[source]:
            print(f"Error: Not enough troops at source {source}. Has {self.node_troops[source]}, tried to send {troops}")
            return False
        
        # Calculate travel time in ticks
        travel_ticks = self._calculate_travel_time(source, target)
        completion_tick = self.current_tick + travel_ticks
        
        # Determine if it's a transfer or attack
        is_transfer = self.node_owners[target] == player
        
        self.node_troops[source] -= troops
        
        attack = Attack(player, source, target, troops, completion_tick, self.current_tick, is_transfer)
        self.ongoing_attacks.append(attack)
        
        # print(f"Attack created: {player.value} sending {troops} troops from {source} to {target}, will arrive at tick {completion_tick} (travel time: {travel_ticks} ticks)")
        
        return True
    
    def _process_attacks(self):
        completed_indices = [] # Use indices to avoid issues when modifying list during iteration
        
        for i, attack in enumerate(self.ongoing_attacks):
            if self.current_tick >= attack.completion_tick:
                # Ensure target node still exists (it should, but good practice)
                if attack.target not in self.graph.nodes():
                    completed_indices.append(i)
                    continue

                if attack.is_transfer:
                    # Friendly transfer
                    # Check if target node is still owned by the same player
                    if self.node_owners[attack.target] == attack.player:
                        self.node_troops[attack.target] += attack.troops
                        # print(f"Transfer completed: {attack.troops} troops transferred to node {attack.target} by {attack.player.value}")
                    else:
                        # Target node was captured by another player before transfer arrived
                        # These troops are lost or could be handled differently (e.g., reinforce new owner if allied)
                        # For now, let's assume they are lost if ownership changed.
                        # print(f"Transfer to {attack.target} by {attack.player.value} failed: Node ownership changed.")
                        pass # Troops are effectively lost
                else:
                    # Combat - attacker needs MORE troops than defender
                    defending_troops = self.node_troops[attack.target]
                    
                    # Check if the target is still owned by an enemy (or neutral)
                    if self.node_owners[attack.target] == attack.player:
                        # Target was already captured by the attacker or an ally before this attack arrived
                        # This becomes a reinforcement
                        self.node_troops[attack.target] += attack.troops
                        # print(f"Reinforcement: {attack.player.value} added {attack.troops} to already owned node {attack.target}")
                    elif attack.troops > defending_troops:
                        # Attacker wins and conquers
                        old_owner = self.node_owners[attack.target]
                        self.node_owners[attack.target] = attack.player
                        self.node_troops[attack.target] = attack.troops - defending_troops
                        # print(f"Attack successful: {attack.player.value} conquered node {attack.target} from {old_owner.value} (sent {attack.troops} vs {defending_troops})")
                    else:
                        # Defender keeps the node (or it remains neutral)
                        self.node_troops[attack.target] = max(0, defending_troops - attack.troops)
                        # print(f"Attack failed: {attack.player.value} lost {attack.troops} troops attacking node {attack.target} (defender had {defending_troops}, now has {self.node_troops[attack.target]})")
                
                completed_indices.append(i)
        
        # Remove completed attacks by index, in reverse order
        for i in sorted(completed_indices, reverse=True):
            del self.ongoing_attacks[i]
    
    def bot_green_strategic(self):
        """Green bot: Only attacks when it can guarantee conquest, accounting for travel time"""
        valid_moves = self.get_valid_moves(Player.GREEN)
        if not valid_moves:
            return
        
        conquerable_moves = []
        for source, target, is_transfer in valid_moves:
            if not is_transfer:  # Only consider attacks, not transfers
                if self.node_owners[target] == Player.GREEN: # Don't attack own nodes
                    continue

                travel_ticks = self._calculate_travel_time(source, target)
                future_defending_troops = self._calculate_future_troops(target, travel_ticks)
                troops_needed = future_defending_troops + 1
                available_troops = self.node_troops[source]
                
                if available_troops >= troops_needed:
                    conquerable_moves.append((source, target, troops_needed, travel_ticks, future_defending_troops))
                    # print(f"GREEN analysis: Node {target} (owner: {self.node_owners[target].value}) has {self.node_troops[target]} now, will have {future_defending_troops} (travel: {travel_ticks} ticks). Need {troops_needed}.")
        
        if conquerable_moves:
            # Prioritize moves (e.g., weakest target, closest, etc. - for now, just take them)
            # Sort by some criteria if needed, e.g., attack weakest first
            # conquerable_moves.sort(key=lambda x: x[4]) # Sort by future_defending_troops

            for source, target, troops_needed, _, future_defending_troops in conquerable_moves:
                # Re-check troops at source, as previous moves might have used them
                if self.node_troops[source] >= troops_needed:
                    self.make_move(Player.GREEN, source, target, troops_needed)
                    # print(f"GREEN strategic attack: Sending {troops_needed} from {source} to {target} (target will have ~{future_defending_troops})")
    
    def bot_random_move(self, bot: Player):
        valid_moves = self.get_valid_moves(bot)
        if not valid_moves:
            return
        
        source, target, is_transfer = random.choice(valid_moves)
        max_troops_at_source = self.node_troops[source]
        
        if max_troops_at_source <= 1 and not is_transfer : # Must leave 1 troop if attacking, unless it's the only troop
             if max_troops_at_source == 1: # Can send 1 troop if it's the only one
                 troops_to_send = 1
             else: # Not enough troops to attack and leave one behind
                 return
        elif max_troops_at_source == 0: # No troops to send
            return
        else:
            if is_transfer:
                # Can transfer all troops
                troops_to_send = random.randint(1, max_troops_at_source)
            else:
                # If attacking, prefer to leave at least one troop, but can send all if it's a strong attack
                # For random bot, let's allow sending n-1 or all troops
                if max_troops_at_source > 1:
                    troops_to_send = random.randint(1, max_troops_at_source -1 if max_troops_at_source > 5 else max_troops_at_source) # Be a bit more conservative or aggressive
                else: # max_troops_at_source is 1
                    troops_to_send = 1


        if troops_to_send > 0:
            self.make_move(bot, source, target, troops_to_send)
    
    def update(self):
        """Update game state by one tick"""
        if self.game_over:
            return

        self.current_tick += 1
        
        self._process_attacks()
        
        if self.current_tick - self.last_income_tick >= GameConfig.INCOME_INTERVAL:
            self.add_income()
            self.last_income_tick = self.current_tick
        
        if self.current_tick - self.last_bot_move_tick >= GameConfig.BOT_MOVE_INTERVAL:
            # Check for winner before bots move, to prevent moves after game ends
            winner_check = self.get_winner()
            if winner_check:
                self.game_over = True
                # print(f"Game over! Winner: {winner_check.value}")
                return # Stop further updates if game is over

            for bot in self.bots:
                if not self.game_over: # Double check, in case one bot wins and another tries to move
                    # Ensure bot still has nodes
                    if any(self.node_owners[n] == bot for n in self.graph.nodes()):
                        if bot == Player.GREEN:
                            self.bot_green_strategic()
                        else:
                            self.bot_random_move(bot)
                    # else:
                        # print(f"Bot {bot.value} has no nodes, skipping turn.")
            self.last_bot_move_tick = self.current_tick
        
        # Check for winner after all updates for the tick
        if not self.game_over: # Only check if not already over
            winner = self.get_winner()
            if winner:
                self.game_over = True
                # print(f"Game over! Winner: {winner.value} at tick {self.current_tick}")
    
    def add_income(self):
        """Add income only to player-owned nodes (not grey)"""
        added_income_this_tick = False
        for node, owner in self.node_owners.items():
            if owner != Player.GREY:
                self.node_troops[node] += GameConfig.INCOME_PER_TICK
                added_income_this_tick = True
        
        # if added_income_this_tick:
            # print(f"Income added at tick {self.current_tick}: +{GameConfig.INCOME_PER_TICK} troops to all player nodes")
    
    def get_scores(self) -> Dict[Player, int]:
        """Returns node count for each player."""
        scores = {player: 0 for player in Player if player != Player.GREY}
        for owner in self.node_owners.values():
            if owner != Player.GREY:
                scores[owner] += 1
        return scores

    def get_player_troop_counts(self) -> Dict[Player, int]:
        """Calculates total troops for each active player."""
        player_troops = {player: 0 for player in Player if player != Player.GREY}
        for node, owner in self.node_owners.items():
            if owner != Player.GREY:
                player_troops[owner] += self.node_troops[node]
        return player_troops

    def get_total_troops_on_map(self) -> int:
        """Calculates the sum of all troops on the map."""
        return sum(self.node_troops.values())

    def get_winner(self) -> Optional[Player]:
        """
        Determines the winner based on troop majority.
        A player wins if they own more than 95% of the total troops on the map.
        Also checks if only one player remains with troops.
        """
        player_troop_counts = self.get_player_troop_counts()
        total_troops_on_map = self.get_total_troops_on_map()

        if total_troops_on_map == 0:
            # No troops left, or game just started weirdly. No winner by this rule.
            # Could also mean all players were eliminated.
            active_players_with_nodes = {owner for owner in self.node_owners.values() if owner != Player.GREY}
            if len(active_players_with_nodes) == 1:
                 # If only one player owns any nodes, and total troops are 0 (meaning they also have 0)
                 # This is a stalemate or a win for the last one standing if others were eliminated.
                 # Let's assume if only one player has nodes, they win, even with 0 troops if others also have 0.
                 # This scenario is unlikely if income exists.
                 # For now, let's stick to troop majority. If total_troops_on_map is 0, no one has >95%.
                return None


        # Check for troop majority win
        for player, troops in player_troop_counts.items():
            if total_troops_on_map > 0 and (troops / total_troops_on_map) > 0.95:
                # print(f"Player {player.value} has {troops}/{total_troops_on_map} troops ({ (troops / total_troops_on_map)*100:.2f}%) - WINS!")
                return player

        # Check if only one player has any troops left
        active_players_with_troops = [p for p, t in player_troop_counts.items() if t > 0]
        if len(active_players_with_troops) == 1:
            # If only one player has troops, they are the winner, regardless of the 95% rule,
            # assuming other players are eliminated (have 0 troops).
            # print(f"Player {active_players_with_troops[0].value} is the only one with troops - WINS!")
            return active_players_with_troops[0]
        
        # Check if only one player owns any nodes (alternative win if all others are eliminated)
        # This is a fallback if troop-based win isn't met but others are wiped out.
        scores = self.get_scores() # Node counts
        players_with_nodes = [p for p, count in scores.items() if count > 0]
        if len(players_with_nodes) == 1 and total_troops_on_map > 0 : # Ensure the remaining player has nodes
             # And check if this player also has all the troops (or is the only one with troops)
            if player_troop_counts.get(players_with_nodes[0], 0) == total_troops_on_map :
                # print(f"Player {players_with_nodes[0].value} owns all nodes with troops - WINS!")
                return players_with_nodes[0]


        return None
    
    def get_ongoing_attacks(self):
        """Get ongoing attacks with correct progress calculation"""
        result = []
        
        for attack in self.ongoing_attacks:
            total_travel_time = attack.completion_tick - attack.start_tick
            # Ensure total_travel_time is not zero to prevent division by zero
            if total_travel_time <= 0: # Should be at least 1 due to max(1, ...) in travel time calc
                progress = 1.0 # If travel time is instant or past, it's complete
            else:
                elapsed_time = self.current_tick - attack.start_tick
                progress = min(1.0, max(0.0, elapsed_time / total_travel_time))
            
            result.append({
                'player': attack.player.value,
                'source': attack.source,
                'target': attack.target,
                'troops': attack.troops,
                'completion_tick': attack.completion_tick,
                'start_tick': attack.start_tick,
                'is_transfer': attack.is_transfer,
                'progress': progress
            })
        
        return result