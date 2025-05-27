# config.py
class GameConfig:
    # Timing (in game ticks, not real time)
    GAME_TICK_SPEED = 1000  # Ticks per second (for PyGame human visualization)
    INCOME_INTERVAL = 10    # Ticks between income
    BOT_MOVE_INTERVAL = 20  # Ticks between non-agent bot moves

    # --- Add/Update these for the Gym Environment ---
    AGENT_STEP_INTERVAL_TICKS = 20 # How many game ticks per agent env.step()
                                   # Agent acts, then game simulates this many ticks.
    MAX_EPISODE_TICKS = 5000       # Max game ticks before episode is truncated.
    MAX_AGENT_STEPS_PER_EPISODE = MAX_EPISODE_TICKS // AGENT_STEP_INTERVAL_TICKS # Max agent actions
    # --- End Gym Environment additions ---

    # Income
    INCOME_PER_TICK = 1  # Troops added per income interval
    
    # Travel speed
    TRAVEL_SPEED = 5.0  # Distance units per tick
    
    # Game setup
    MIN_NODES = 15
    MAX_NODES = 25 # Crucial for fixed-size observation/action spaces
    STARTING_TROOPS = 10
    NEUTRAL_MIN_TROOPS = 0
    NEUTRAL_MAX_TROOPS = 50