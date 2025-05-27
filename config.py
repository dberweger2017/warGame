# config.py
class GameConfig:
    # Timing (in game ticks, not real time)
    GAME_TICK_SPEED = 1000  # Ticks per second (higher = faster game)
    INCOME_INTERVAL = 10  # Ticks between income (10 ticks = 1 second at speed 10)
    BOT_MOVE_INTERVAL = 20  # Ticks between bot moves (100 ticks = 10 seconds at speed 10)
    
    # Income
    INCOME_PER_TICK = 1  # Troops added per income interval
    
    # Travel speed
    TRAVEL_SPEED = 5.0  # Distance units per tick (lower = slower travel)
    
    # Game setup
    MIN_NODES = 15
    MAX_NODES = 25
    STARTING_TROOPS = 10
    NEUTRAL_MIN_TROOPS = 0
    NEUTRAL_MAX_TROOPS = 50