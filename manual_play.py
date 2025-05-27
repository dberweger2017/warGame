# manual_play.py
import numpy as np
import time # For a slight delay if needed

# Imports from your project structure
from config import GameConfig # For MAX_NODES, etc.
from game import Player # To specify which player you are
from conquest_env import ConquestEnv, PYGAME_AVAILABLE # Import your environment

def get_player_action(env: ConquestEnv, num_actual_nodes: int) -> np.ndarray:
    """
    Gets action input from the human player.
    Returns an action array suitable for env.step()
    """
    print("\n--- Your Turn ---")
    print(f"You are Player: {env.agent_player.value}")
    
    # Display agent's nodes and troops
    agent_nodes_info = []
    current_obs = env._get_obs() # Get current observation for node details
    for node_id in range(num_actual_nodes):
        owner_int = current_obs["node_owners"][node_id]
        owner_player = env.int_player_map.get(owner_int)
        if owner_player == env.agent_player:
            troops = current_obs["node_troops"][node_id]
            agent_nodes_info.append(f"Node {node_id} (Troops: {int(troops)})")
    
    if not agent_nodes_info:
        print("You have no nodes. You must pass or will likely get a penalty.")
    else:
        print("Your Nodes:", ", ".join(agent_nodes_info))

    while True:
        try:
            print("\nChoose action type:")
            print("  'm' or 'move' - Make a move (attack/transfer)")
            print("  'n' or 'noop' - No operation (pass turn)")
            print("  'q' or 'quit' - Quit game")
            action_type = input("Enter action type: ").strip().lower()

            if action_type in ['q', 'quit']:
                return None # Signal to quit

            if action_type in ['n', 'noop']:
                # Construct a NO-OP action
                # Action: [source_idx, target_idx, troop_choice_idx]
                # For NO-OP, source and target can be dummy, troop_choice_idx is NO_OP_ACTION_IDX
                return np.array([0, 0, env.NO_OP_ACTION_IDX], dtype=np.int32)

            if action_type in ['m', 'move']:
                source_node = int(input(f"Enter Source Node ID (0 to {num_actual_nodes - 1}): "))
                if not (0 <= source_node < num_actual_nodes):
                    print(f"Invalid source node. Must be between 0 and {num_actual_nodes - 1}.")
                    continue
                if env.game.node_owners.get(source_node) != env.agent_player:
                    print(f"Invalid source: You do not own Node {source_node}.")
                    continue
                if env.game.node_troops.get(source_node, 0) == 0:
                    print(f"Invalid source: Node {source_node} has no troops.")
                    continue

                # Show valid targets for the selected source
                valid_targets = env.game.get_valid_targets(source_node, env.agent_player)
                if not valid_targets:
                    print(f"Node {source_node} has no valid targets (no neighbors to move to). Try another source or 'noop'.")
                    continue
                
                print(f"Valid targets from Node {source_node}: {valid_targets}")
                for t_id in valid_targets:
                    t_owner = env.game.node_owners.get(t_id, Player.GREY)
                    t_troops = env.game.node_troops.get(t_id, 0)
                    action_desc = "Transfer to" if t_owner == env.agent_player else "Attack"
                    print(f"  Target {t_id} ({action_desc} {t_owner.value}, Troops: {t_troops})")


                target_node = int(input(f"Enter Target Node ID (from valid targets): "))
                if target_node not in valid_targets:
                    print(f"Invalid target node. Choose from {valid_targets}.")
                    continue
                
                print("Choose troop percentage to send (of available/sendable):")
                print("  1: 25%")
                print("  2: 50%")
                print("  3: 75%")
                print("  4: 100%")
                troop_choice_input = int(input("Enter troop choice (1-4): "))

                if not (1 <= troop_choice_input <= 4):
                    print("Invalid troop choice. Must be 1-4.")
                    continue
                
                # Convert to 0-indexed troop_choice_idx for the environment
                troop_choice_idx = troop_choice_input - 1 
                
                return np.array([source_node, target_node, troop_choice_idx], dtype=np.int32)
            
            else:
                print("Invalid action type. Please choose 'move', 'noop', or 'quit'.")

        except ValueError:
            print("Invalid input. Please enter numbers where expected.")
        except Exception as e:
            print(f"An error occurred: {e}")

def manual_play():
    if not PYGAME_AVAILABLE:
        print("Pygame is not available. Human rendering for manual play is not possible.")
        print("Please install Pygame: pip install pygame")
        return

    # --- Configuration ---
    human_player_id = Player.BLUE # Or Player.RED, etc.
    # --- End Configuration ---

    print(f"Starting manual play for Player: {human_player_id.value}")
    print(f"Game will run with human rendering.")
    print(f"An agent step (your turn) advances the game by {GameConfig.AGENT_STEP_INTERVAL_TICKS} game ticks.")

    env = ConquestEnv(agent_player_id=human_player_id, render_mode="human")
    
    play_again = True
    while play_again:
        obs, info = env.reset(seed=int(time.time())) # Use a different seed each time
        num_actual_nodes = info.get("num_actual_nodes", GameConfig.MAX_NODES)
        
        terminated = False
        truncated = False
        current_episode_reward = 0.0
        agent_step = 0

        # Initial render is handled by env.reset if render_mode is human
        # env.render() # Or call explicitly if reset doesn't trigger it as expected

        while not (terminated or truncated):
            agent_step += 1
            print(f"\n--- Agent Step {agent_step} (Game Tick: {info.get('current_tick', 0)}) ---")

            action = get_player_action(env, num_actual_nodes)
            
            if action is None: # Player chose to quit
                print("Quitting game.")
                terminated = True # End the current game loop
                play_again = False # Don't ask to play another
                break 

            obs, reward, terminated, truncated, info = env.step(action)
            num_actual_nodes = info.get("num_actual_nodes", GameConfig.MAX_NODES) # Update for next input
            current_episode_reward += reward

            print(f"Action taken: {action}")
            print(f"Reward for this step: {reward:.2f}")
            print(f"Cumulative Reward: {current_episode_reward:.2f}")
            if info.get('winner'):
                print(f"WINNER: {info['winner'].value}")
            
            # Render is called inside env.step() if render_mode is "human"
            # time.sleep(0.1) # Optional small delay

        print("\n--- GAME OVER ---")
        print(f"Final State. Terminated: {terminated}, Truncated: {truncated}")
        final_winner = info.get('winner')
        if final_winner:
            print(f"The winner is: Player {final_winner.value}!")
        else:
            print("The game ended without a clear winner (or was truncated/quit).")
        print(f"Total reward for this episode: {current_episode_reward:.2f}")

        if play_again: # Only ask if not quit
            while True:
                again = input("Play again? (y/n): ").strip().lower()
                if again == 'y':
                    break
                elif again == 'n':
                    play_again = False
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
    
    env.close()
    print("Manual play session ended.")

if __name__ == "__main__":
    manual_play()