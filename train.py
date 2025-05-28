# train_ppo.py
import os
import time
import numpy as np
from collections import deque # For rolling window statistics

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList 
# CheckpointCallback can still be used if you want timestep-based saves too

# Imports from your project
from conquest_env import ConquestEnv # Your environment
from game import Player # To specify agent player
from config import GameConfig # For paths or other configs

# --- Configuration ---
AGENT_PLAYER = Player.BLUE  # Which player the agent will control
TARGET_EPISODES = 10_000    # Target number of games/episodes to train for
SAVE_MODEL_EVERY_N_EPISODES = 500 # How often to save a model checkpoint

# Estimate total timesteps: TARGET_EPISODES * average_steps_per_episode
# GameConfig.MAX_AGENT_STEPS_PER_EPISODE is a good upper bound for an average.
ESTIMATED_AVG_STEPS_PER_EPISODE = GameConfig.MAX_AGENT_STEPS_PER_EPISODE // 2 # A rough guess
# If games end much faster, you might need more episodes or a direct timestep target.
TOTAL_TIMESTEPS = TARGET_EPISODES * ESTIMATED_AVG_STEPS_PER_EPISODE
# If you prefer a fixed number of timesteps, set it directly:
# TOTAL_TIMESTEPS = 5_000_000 # Example: 5 million timesteps

# Determine NUM_CPU, ensuring it's at least 1
cpu_count = os.cpu_count()
NUM_CPU = cpu_count if isinstance(cpu_count, int) and cpu_count > 0 else 1
# NUM_CPU = 1 # Uncomment for debugging or if SubprocVecEnv causes issues

# Log and model save paths
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
LOG_DIR_BASE = "training_logs"
MODEL_NAME = f"ppo_conquest_{AGENT_PLAYER.value}_{TIMESTAMP}"
LOG_DIR = os.path.join(LOG_DIR_BASE, MODEL_NAME)
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard_logs")
MODEL_SAVE_DIR = os.path.join(LOG_DIR, "models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# --- End Configuration ---

class DetailedMetricsCallback(BaseCallback):
    """
    A custom callback to log detailed game metrics to TensorBoard,
    including win rate, average game length, and max troops achieved by the agent.
    It also handles model checkpointing every N episodes.
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
        
        # Initialize tracking variables
        self.episode_count_total = 0 # Total episodes across all envs
        self.last_saved_episode_milestone = 0 # Based on total episodes
        
        # For rolling statistics
        self.ep_game_lengths = deque(maxlen=rolling_window_size)
        self.ep_wins = deque(maxlen=rolling_window_size) # 1 for win, 0 for loss/draw
        self.ep_agent_max_troops = deque(maxlen=rolling_window_size) # Max troops agent had in an episode

        # Per-environment tracking for current episode's max troops
        self.num_envs = 0 
        self.env_current_episode_agent_max_troops = []


    def _on_training_start(self) -> None:
        """ This method is called before the first rollout starts. """
        self.num_envs = self.training_env.num_envs
        self.env_current_episode_agent_max_troops = [0] * self.num_envs
        if self.verbose > 0:
            print(f"DetailedMetricsCallback: Initialized for {self.num_envs} environments.")
            print(f"Agent Player for win condition: {self.agent_player.value}")
            print(f"Rolling window for metrics: {self.rolling_window_size} episodes.")

    def _on_step(self) -> bool:
        # Update max troops for ongoing episodes in each environment
        # This relies on 'agent_troops' being present in the info dict from each step
        infos = self.locals.get("infos", [{} for _ in range(self.num_envs)])
        for i in range(self.num_envs):
            agent_troops_now = infos[i].get("agent_troops", 0)
            if agent_troops_now > self.env_current_episode_agent_max_troops[i]:
                self.env_current_episode_agent_max_troops[i] = agent_troops_now

        # Check for episode completions
        dones = self.locals.get("dones", [False] * self.num_envs)
        for i in range(self.num_envs):
            if dones[i]:
                self.episode_count_total += 1
                # Retrieve the info dict from the step where the episode ended
                # SB3's Monitor wrapper (used by default with VecEnvs) stores episode info
                # in `info['episode']` when an episode is done.
                info = infos[i]
                episode_info = info.get("episode")

                # 1. Game Length
                if episode_info and "l" in episode_info:
                    self.ep_game_lengths.append(episode_info["l"])
                elif "current_tick" in info : # Fallback if Monitor not fully wrapped or custom info
                    self.ep_game_lengths.append(info["current_tick"])


                # 2. Win Rate
                # 'winner' should be in the raw 'info' dict from the environment
                winner = info.get("winner") 
                is_win = 1 if winner == self.agent_player else 0
                self.ep_wins.append(is_win)

                # 3. Max Troops Achieved by Agent in This Episode
                # Use the value tracked per-environment and reset it
                self.ep_agent_max_troops.append(self.env_current_episode_agent_max_troops[i])
                self.env_current_episode_agent_max_troops[i] = 0 # Reset for the next episode in this env

                # Log to TensorBoard
                if self.logger:
                    self.logger.record("rollout/episodes_completed_total", self.episode_count_total)
                    if len(self.ep_game_lengths) > 0:
                        self.logger.record("custom/avg_game_length_rollout", np.mean(self.ep_game_lengths))
                    if len(self.ep_wins) > 0:
                        self.logger.record("custom/win_rate_rollout", np.mean(self.ep_wins))
                    if len(self.ep_agent_max_troops) > 0:
                        self.logger.record("custom/avg_agent_max_troops_rollout", np.mean(self.ep_agent_max_troops))
                
                # Checkpoint saving based on total episodes completed
                if self.episode_count_total >= self.last_saved_episode_milestone + self.save_every_n_episodes:
                    model_path = os.path.join(self.save_path, f"{self.name_prefix}_ep{self.episode_count_total}")
                    if self.model is not None:
                        self.model.save(model_path)
                        if self.verbose > 0:
                            print(f"Saved model checkpoint to {model_path} (Total Episodes: {self.episode_count_total})")
                    self.last_saved_episode_milestone = self.episode_count_total
        return True

def make_env(rank: int, agent_player_id: Player, seed: int = 0):
    """ Utility function for multiprocessed env. """
    def _init():
        env = ConquestEnv(agent_player_id=agent_player_id, render_mode=None)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    print(f"--- Conquest Game PPO Training ---")
    print(f"Agent Player: {AGENT_PLAYER.value}")
    print(f"Target Episodes: {TARGET_EPISODES} (approx. {TOTAL_TIMESTEPS} timesteps)")
    print(f"Save model every: {SAVE_MODEL_EVERY_N_EPISODES} episodes")
    print(f"Number of CPUs for parallel envs: {NUM_CPU}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"TensorBoard Logs: {TENSORBOARD_LOG_DIR}")
    print(f"Run 'tensorboard --logdir={TENSORBOARD_LOG_DIR}' to monitor.")
    print(f"------------------------------------")

    initial_seed = int(time.time())
    if NUM_CPU > 1:
        vec_env = SubprocVecEnv([make_env(i, AGENT_PLAYER, seed=initial_seed) for i in range(NUM_CPU)])
    else:
        vec_env = DummyVecEnv([make_env(0, AGENT_PLAYER, seed=initial_seed)])

    # --- Callbacks ---
    detailed_metrics_callback = DetailedMetricsCallback(
        agent_player_for_win_condition=AGENT_PLAYER,
        save_every_n_episodes=SAVE_MODEL_EVERY_N_EPISODES,
        save_path=MODEL_SAVE_DIR,
        name_prefix=f"ppo_conquest_ep_chkpt",
        rolling_window_size=100, # Log averages over last 100 episodes
        verbose=1
    )
    callbacks = CallbackList([detailed_metrics_callback])

    # --- Define the PPO Model ---
    ppo_n_steps_per_env = 2048 # Default PPO n_steps, adjust if you change it
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        learning_rate=3e-4,
        n_steps=ppo_n_steps_per_env, 
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005, 
        vf_coef=0.5,
        # policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )

    print(f"\nModel Architecture:\n{model.policy}\n")

    # --- Train the Agent ---
    try:
        print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
        # Note: model.learn uses total_timesteps, not episodes.
        # The callback handles episode-based logic.
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True # Requires 'tqdm'
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
        vec_env.close()

    print(f"Training session ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Optional: Test loaded model (ensure PYGAME_AVAILABLE is checked if rendering) ---
    # from conquest_env import PYGAME_AVAILABLE # Import if not already
    # print("\n--- Testing loaded model ---")
    # try:
    #     del model 
    #     loaded_model = PPO.load("training_logs/ppo_conquest_blue_20250527-173435/models/ppo_conquest_ep_chkpt_ep5500.zip")
    #     print(f"Successfully loaded model from {final_model_path}")

    #     # Create a single test environment
    #     render_mode_test = "human" if PYGAME_AVAILABLE else None
    #     test_env = ConquestEnv(agent_player_id=AGENT_PLAYER, render_mode=render_mode_test)
    #     obs, _ = test_env.reset()
        
    #     for _ep in range(3): # Test for a few episodes
    #         ep_steps = 0
    #         for _step in range(GameConfig.MAX_AGENT_STEPS_PER_EPISODE + 10):
    #             ep_steps +=1
    #             action, _states = loaded_model.predict(obs, deterministic=True)
    #             obs, reward, terminated, truncated, info = test_env.step(action)
    #             if render_mode_test == "human":
    #                 test_env.render()
    #                 time.sleep(0.03) 
                
    #             if terminated or truncated:
    #                 print(f"Test Episode {_ep+1} finished after {ep_steps} steps. Winner: {info.get('winner')}, Agent Troops: {info.get('agent_troops')}")
    #                 obs, _ = test_env.reset()
    #                 break
    #     test_env.close()
    # except FileNotFoundError:
    #     print(f"Could not find model at {final_model_path} to test.")
    # except Exception as e:
    #     print(f"Error during model loading or testing: {e}")
    #     import traceback
    #     traceback.print_exc()

