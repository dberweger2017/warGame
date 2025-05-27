import random
import networkx as nx
import time
from enum import Enum
from typing import Dict, List, Tuple, Optional

class Player(Enum):
    GREY = "grey"  # unconquered
    RED = "red"    # bot
    GREEN = "green"  # bot
    BLUE = "blue"    # human player
    YELLOW = "yellow"  # bot

class ConquestGame:
    def __init__(self):
        self.bots = [Player.RED, Player.GREEN, Player.YELLOW]
        self.human_player = Player.BLUE
        self.last_income_time = time.time()
        self.last_bot_move_time = time.time()
        self.reset_game()
    
    def reset_game(self):
        # Generate random connected graph
        num_nodes = random.randint(10, 30)
        self.graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.3)
        
        # Initialize node states
        self.node_owners = {node: Player.GREY for node in self.graph.nodes()}
        self.node_troops = {node: random.randint(0, 50) for node in self.graph.nodes()}
        
        # Assign starting nodes to players
        start_nodes = random.sample(list(self.graph.nodes()), 4)
        players = [Player.RED, Player.GREEN, Player.BLUE, Player.YELLOW]
        
        for node, player in zip(start_nodes, players):
            self.node_owners[node] = player
            self.node_troops[node] = 10  # Starting troops
        
        self.game_over = False
        self.last_income_time = time.time()
        self.last_bot_move_time = time.time()
    
    def get_valid_moves(self, player: Player) -> List[Tuple[int, int]]:
        """Get all valid (source, target) moves for a player"""
        moves = []
        player_nodes = [n for n, owner in self.node_owners.items() if owner == player]
        
        for source in player_nodes:
            if self.node_troops[source] > 1:  # Need at least 1 troop to stay
                for target in self.graph.neighbors(source):
                    if self.node_owners[target] != player:  # Can't attack own nodes
                        moves.append((source, target))
        return moves
    
    def make_move(self, player: Player, source: int, target: int, troops: int) -> bool:
        """Execute a move. Returns True if successful"""
        # Validate move
        if (source, target) not in self.get_valid_moves(player):
            return False
        if troops < 1 or troops >= self.node_troops[source]:
            return False
        
        # Combat resolution
        defending_troops = self.node_troops[target]
        if troops > defending_troops:
            # Attacker wins
            self.node_owners[target] = player
            self.node_troops[target] = troops - defending_troops
            self.node_troops[source] -= troops
        else:
            # Defender wins
            self.node_troops[source] -= troops
        
        return True
    
    def bot_random_move(self, bot: Player):
        """Make a random move for a bot"""
        valid_moves = self.get_valid_moves(bot)
        if not valid_moves:
            return
        
        source, target = random.choice(valid_moves)
        max_troops = self.node_troops[source] - 1
        troops = random.randint(1, max_troops)
        
        self.make_move(bot, source, target, troops)
        print(f"{bot.value} attacks {target} from {source} with {troops} troops")
    
    def update(self):
        """Call this continuously to handle timing"""
        current_time = time.time()
        
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
        """Add 5 troops to each owned node"""
        for node, owner in self.node_owners.items():
            if owner != Player.GREY:
                self.node_troops[node] += 5
    
    def get_scores(self) -> Dict[Player, int]:
        """Get node count for each player"""
        scores = {player: 0 for player in Player if player != Player.GREY}
        for owner in self.node_owners.values():
            if owner != Player.GREY:
                scores[owner] += 1
        return scores
    
    def get_winner(self) -> Optional[Player]:
        """Check if game is over and return winner"""
        scores = self.get_scores()
        if sum(scores.values()) == len(self.graph.nodes()):
            return max(scores, key=scores.get)
        return None

# Test with game loop
if __name__ == "__main__":
    game = ConquestGame()
    print(f"Game started with {len(game.graph.nodes())} nodes")
    print(f"Your valid moves: {game.get_valid_moves(Player.BLUE)}")
    
    # Simple game loop
    while not game.game_over:
        game.update()
        time.sleep(0.1)  # Small delay