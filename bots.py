# bots.py
import random
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from game import ConquestGame, Player # For type hinting to avoid circular imports

class BotStrategy:
    """Base class for bot strategies."""
    def get_action(self, game: 'ConquestGame', player: 'Player') -> Optional[Tuple[int, int, int]]:
        """
        Determines the action for the bot.
        Returns a tuple (source_node, target_node, troops_to_send) or None for no action.
        The game's make_move method will be called with this action.
        """
        raise NotImplementedError

class RandomBot(BotStrategy):
    """A bot that makes random valid moves."""
    def get_action(self, game: 'ConquestGame', player: 'Player') -> Optional[Tuple[int, int, int]]:
        valid_moves = game.get_valid_moves(player) # This returns (source, target, is_transfer)
        if not valid_moves:
            return None

        source, target, is_transfer = random.choice(valid_moves)
        max_troops_at_source = game.node_troops.get(source, 0)

        if max_troops_at_source == 0:
            return None
        
        troops_to_send = 0
        if is_transfer:
            if max_troops_at_source > 0:
                troops_to_send = random.randint(1, max_troops_at_source)
        else: # Attacking
            if max_troops_at_source > 1:
                # Send between 1 and all but one troop, or all if feeling aggressive/desperate
                # For a simple random bot, let's allow sending up to n-1 or all.
                troops_to_send = random.randint(1, max_troops_at_source -1 if max_troops_at_source > 5 else max_troops_at_source)
            elif max_troops_at_source == 1:
                troops_to_send = 1
        
        if troops_to_send > 0:
            return source, target, troops_to_send
        return None

class AggressiveBot(BotStrategy):
    """
    A bot that only attacks when it can guarantee conquest,
    accounting for travel time and future troop calculations.
    (Formerly bot_green_strategic)
    """
    def get_action(self, game: 'ConquestGame', player: 'Player') -> Optional[Tuple[int, int, int]]:
        valid_player_moves = game.get_valid_moves(player)
        if not valid_player_moves:
            return None
        
        conquerable_targets_info = [] # Store (source, target, troops_needed)

        for source, target, is_transfer in valid_player_moves:
            if not is_transfer and game.node_owners.get(target) != player: # Only consider attacks on non-owned nodes
                travel_ticks = game._calculate_travel_time(source, target)
                future_defending_troops = game._calculate_future_troops(target, travel_ticks)
                troops_needed_to_conquer = future_defending_troops + 1
                
                available_troops_at_source = game.node_troops.get(source, 0)
                
                if available_troops_at_source >= troops_needed_to_conquer:
                    conquerable_targets_info.append({
                        "source": source,
                        "target": target,
                        "troops_to_send": troops_needed_to_conquer,
                        "priority": troops_needed_to_conquer # Lower is better (easier target)
                    })
        
        if not conquerable_targets_info:
            # Fallback: if no guaranteed conquests, maybe make a large transfer to a frontline node?
            # Or simply do nothing if no aggressive moves are possible. For now, do nothing.
            return None

        # Prioritize the "easiest" guaranteed conquest (requires fewest troops)
        # Or could pick the one that leaves the source strongest, etc.
        # For now, let's pick one, e.g., the one requiring the fewest troops.
        # If multiple, could pick randomly among the best.
        conquerable_targets_info.sort(key=lambda x: x["priority"])
        
        # Try to make the best move, but ensure troops are still available
        # (as previous actions in the same turn by other bots are not considered here)
        best_move = conquerable_targets_info[0]
        
        # Re-check troops at source, as this bot might make multiple moves if allowed per "bot turn"
        # However, our current game loop calls each bot once per BOT_MOVE_INTERVAL.
        if game.node_troops.get(best_move["source"], 0) >= best_move["troops_to_send"]:
            # print(f"AggressiveBot ({player.value}): Attacking {best_move['target']} from {best_move['source']} with {best_move['troops_to_send']}")
            return best_move["source"], best_move["target"], best_move["troops_to_send"]
        
        return None

# You can add more bot strategies here later
# class DefensiveBot(BotStrategy): ...
# class ExpansionistBot(BotStrategy): ...
