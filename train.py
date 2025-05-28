# train_ppo.py
import os
import time
import numpy as np
from collections import deque 
from typing import Dict, Any, List, Tuple, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList 

# Imports from your project
from conquest_env import ConquestEnv # Your environment
from game import Player # Import Player enum
from config import GameConfig
from bots import RandomBot # Import the RandomBot strategy

# --- Configuration ---
AGENT_PLAYER = Player.BLUE  # The player the RL agent will control

# Define bot assignments for opponent players for the environment
# All opponents will be RandomBots in this setup.
OPPONENT_BOT_ASSIGNMENTS = {
    player: RandomBot() 
    for player in Player.get_active_players() 
    if player != AGENT_PLAYER # Assign RandomBot to all non-agent players
}
# Example: If AGENT_PLAYER is BLUE, then RED, GREEN, YELLOW will be RandomBot.

TARGET_EPISODES = 10_000    # Target number of games/episodes to train for
SAVE_MODEL_EVERY_N_EPISODES = 500 # How often to save a model checkpoint

# Estimate total timesteps
ESTIMATED_AVG_STEPS_PER_EPISODE = GameConfig.MAX_AGENT_STEPS_PER_EPISODE // 2 
TOTAL_TIMESTEPS = TARGET_EPISODES * ESTIMATED_AVG_STEPS_PER_EPISODE
# Or set a fixed number: TOTAL_TIMESTEPS = 2_000_000 

# Determine NUM_CPU, ensuring it's at least 1
cpu_count = os.cpu_count()
NUM_CPU = cpu_count if isinstance(cpu_count, int) and cpu_count > 0 else 1
# NUM_CPU = 1 # Uncomment for debugging or if SubprocVecEnv causes issues

# Log and model save paths
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
LOG_DIR_BASE = "training_logs"
MODEL_NAME_SUFFIX = "_vs_RandomBots" # Indicate opponent type
MODEL_NAME = f"ppo_conquest_{AGENT_PLAYER.value}_{TIMESTAMP}{MODEL_NAME_SUFFIX}"
LOG_DIR = os.path.join(LOG_DIR_BASE, MODEL_NAME)
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard_logs")
MODEL_SAVE_DIR = os.path.join(LOG_DIR, "models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# --- End Configuration ---

class DetailedMetricsCallback(BaseCallback):
    """
    Custom callback for logging detailed game metrics and saving model checkpoints.
    """
    def __init__(self,
                 agent_player_for_win_condition: Player,
                 save_every_n_episodes: int,
                 save_path: str,
                 name_prefix: str = "rl_model",
                 rolling_window_size: int = 100,
                 verbose: int = 0):
        super(DetailedMetricsCallback, self).__init__(verbose)
        self.agent_player = agent_player_for_win_condition
        self.save_every_n_episodes = save_every_n_episodes
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.rolling_window_size = rolling_window_size
        
        self.episode_count_total = 0
        self.last_saved_episode_milestone = 0
        
        self.ep_game_lengths = deque(maxlen=rolling_window_size)
        self.ep_wins = deque(maxlen=rolling_window_size)
        self.ep_agent_max_troops = deque(maxlen=rolling_window_size)
        
        self.num_envs = 0 
        self.env_current_episode_agent_max_troops = []

    def _on_training_start(self) -> None:
        self.num_envs = self.training_env.num_envs
        self.env_current_episode_agent_max_troops = [0] * self.num_envs
        if self.verbose > 0:
            print(f"DetailedMetricsCallback: Initialized for {self.num_envs} environments.")
            print(f"Agent Player for win condition: {self.agent_player.value}")
            print(f"Rolling window for metrics: {self.rolling_window_size} episodes.")

    def _on_step(self) -> bool:
        # Track max troops for ongoing episodes
        # Assumes 'agent_troops' is in the info dict from each step
        infos_current_step = self.locals.get("infos", [{} for _ in range(self.num_envs)])
        for i in range(self.num_envs):
            agent_troops_now = infos_current_step[i].get("agent_troops", 0)
            if agent_troops_now > self.env_current_episode_agent_max_troops[i]:
                self.env_current_episode_agent_max_troops[i] = agent_troops_now

        # Check for episode completions
        dones = self.locals.get("dones", [False] * self.num_envs)
        # infos_on_done = self.locals.get("infos", [{} for _ in range(self.num_envs)]) # Redundant if using infos_current_step

        for i in range(self.num_envs):
            if dones[i]:
                self.episode_count_total += 1
                info_at_done = infos_current_step[i] # Info from the step where done is True
                episode_monitor_info = info_at_done.get("episode") # From SB3 Monitor wrapper

                # 1. Game Length
                if episode_monitor_info and "l" in episode_monitor_info:
                    self.ep_game_lengths.append(episode_monitor_info["l"])
                elif "current_tick" in info_at_done : 
                    self.ep_game_lengths.append(info_at_done["current_tick"])

                # 2. Win Rate
                winner = info_at_done.get("winner") 
                is_win = 1 if winner == self.agent_player else 0
                self.ep_wins.append(is_win)

                # 3. Max Troops Achieved by Agent in This Episode
                self.ep_agent_max_troops.append(self.env_current_episode_agent_max_troops[i])
                self.env_current_episode_agent_max_troops[i] = 0 # Reset for the next episode

                # Log to TensorBoard
                if self.logger:
                    self.logger.record("rollout/episodes_completed_total", self.episode_count_total)
                    if len(self.ep_game_lengths) > 0:
                        self.logger.record("custom/avg_game_length_rollout", np.mean(self.ep_game_lengths))
                    if len(self.ep_wins) > 0:
                        self.logger.record("custom/win_rate_rollout", np.mean(self.ep_wins))
                    if len(self.ep_agent_max_troops) > 0:
                        self.logger.record("custom/avg_agent_max_troops_rollout", np.mean(self.ep_agent_max_troops))
                
                # Checkpoint saving
                if self.episode_count_total >= self.last_saved_episode_milestone + self.save_every_n_episodes:
                    model_path = os.path.join(self.save_path, f"{self.name_prefix}_ep{self.episode_count_total}")
                    if self.model is not None:
                        self.model.save(model_path)
                        if self.verbose > 0:
                            print(f"Saved model checkpoint to {model_path} (Total Episodes: {self.episode_count_total})")
                    self.last_saved_episode_milestone = self.episode_count_total
        return True

def make_env_conquest(rank: int, agent_player_id: Player, 
                      opponent_bot_config: Dict[Player, Any], seed: int = 0):
    """ Utility function for multiprocessed ConquestEnv. """
    def _init():
        env = ConquestEnv(
            agent_player_id=agent_player_id,
            opponent_bot_config=opponent_bot_config, 
            render_mode=None # Headless for training
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed) # Seed the process
    return _init

if __name__ == "__main__":
    print(f"--- Conquest Game PPO Training (vs RandomBots) ---")
    print(f"RL Agent Player: {AGENT_PLAYER.value}")
    print(f"Opponent Bot Assignments for Env: { {p.value: type(b).__name__ for p,b in OPPONENT_BOT_ASSIGNMENTS.items()} }")
    print(f"Target Episodes: {TARGET_EPISODES} (approx. {TOTAL_TIMESTEPS} timesteps)")
    print(f"Save model every: {SAVE_MODEL_EVERY_N_EPISODES} episodes")
    print(f"Number of CPUs for parallel envs: {NUM_CPU}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"TensorBoard Logs: {TENSORBOARD_LOG_DIR}")
    print(f"Run 'tensorboard --logdir={TENSORBOARD_LOG_DIR}' to monitor.")
    print(f"------------------------------------")

    initial_seed = int(time.time()) # Use a different seed for each training run for variability
    
    if NUM_CPU > 1:
        vec_env = SubprocVecEnv([make_env_conquest(i, AGENT_PLAYER, OPPONENT_BOT_ASSIGNMENTS, seed=initial_seed) for i in range(NUM_CPU)])
    else:
        vec_env = DummyVecEnv([make_env_conquest(0, AGENT_PLAYER, OPPONENT_BOT_ASSIGNMENTS, seed=initial_seed)])

    # --- Callbacks ---
    detailed_metrics_callback = DetailedMetricsCallback(
        agent_player_for_win_condition=AGENT_PLAYER,
        save_every_n_episodes=SAVE_MODEL_EVERY_N_EPISODES,
        save_path=MODEL_SAVE_DIR,
        name_prefix=f"ppo_conquest_vs_random_ep_chkpt", # Indicate opponent type in checkpoint name
        rolling_window_size=100, 
        verbose=1
    )
    callbacks = CallbackList([detailed_metrics_callback])

    # --- Define the PPO Model ---
    # Common PPO hyperparameters. These often need tuning.
    ppo_n_steps_per_env = 2048 
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        learning_rate=3e-4,     # 0.0003
        n_steps=ppo_n_steps_per_env, 
        batch_size=64,          
        n_epochs=10,            
        gamma=0.99,             
        gae_lambda=0.95,        
        clip_range=0.2,         
        ent_coef=0.005,         # Entropy coefficient for exploration
        vf_coef=0.5,            
        # max_grad_norm=0.5,    # Optional: Gradient clipping
        # policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])) # Example custom network
    )

    print(f"\nModel Architecture:\n{model.policy}\n")

    # --- Train the Agent ---
    try:
        print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True # Requires 'tqdm' (pip install tqdm)
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        print("Saving current model due to error...")
    finally:
        final_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_final.zip")
        model.save(final_model_path)
        print(f"\nTraining finished or stopped. Final model saved to: {final_model_path}")
        
        # Close the vectorized environment
        vec_env.close()

    print(f"Training session ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Optional: Code to test the loaded model (ensure PYGAME_AVAILABLE is checked if rendering) ---
    # from conquest_env import PYGAME_AVAILABLE # Import if not already
    # if os.path.exists(final_model_path):
    #     print("\n--- Testing loaded final model ---")
    #     try:
    #         del model # remove the current model instance if it exists
    #         loaded_model = PPO.load(final_model_path)
    #         print(f"Successfully loaded model from {final_model_path}")

    #         render_mode_test = "human" if PYGAME_AVAILABLE else None
    #         test_env_opponent_config = { p: RandomBot() for p in Player.get_active_players() if p != AGENT_PLAYER }
    #         test_env = ConquestEnv(agent_player_id=AGENT_PLAYER, opponent_bot_config=test_env_opponent_config, render_mode=render_mode_test)
            
    #         for ep in range(3): # Test for a few episodes
    #             obs, _ = test_env.reset()
    #             ep_steps = 0
    #             ep_done = False
    #             while not ep_done:
    #                 ep_steps +=1
    #                 action, _states = loaded_model.predict(obs, deterministic=True)
    #                 obs, reward, terminated, truncated, info = test_env.step(action)
    #                 ep_done = terminated or truncated
    #                 if render_mode_test == "human":
    #                     test_env.render()
    #                     time.sleep(0.03) # Slow down human rendering a bit
                    
    #                 if ep_done:
    #                     print(f"Test Episode {ep+1} finished after {ep_steps} steps. Winner: {info.get('winner')}, Agent Troops: {info.get('agent_troops')}")
    #                     break
    #                 if ep_steps > GameConfig.MAX_AGENT_STEPS_PER_EPISODE + 20: # Safety break
    #                     print(f"Test Episode {ep+1} exceeded max steps. Truncating.")
    #                     break
    #         test_env.close()
    #     except FileNotFoundError:
    #         print(f"Could not find model at {final_model_path} to test.")
    #     except Exception as e:
    #         print(f"Error during model loading or testing: {e}")
    #         import traceback
    #         traceback.print_exc()
    # else:
    #     print(f"Final model {final_model_path} not found for testing.")