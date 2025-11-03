"""
Training script for Space Invaders DQN Agent
Implements Deep Q-Learning with experience replay and target network
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

from model import create_dqn_model, get_model_summary
from utils import create_environment, plot_training_progress, print_environment_info


class TrainingCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for training monitoring
    """
    
    def __init__(self, save_interval=10000):
        super().__init__()
        self.save_interval = save_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def on_episode_end(self, episode, logs={}):
        """Called at the end of each episode"""
        self.episode_count += 1
        self.episode_rewards.append(logs.get('episode_reward', 0))
        self.episode_lengths.append(logs.get('nb_steps', 0))
        
        # Print progress every 10 episodes
        if self.episode_count % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            avg_length = np.mean(self.episode_lengths[-10:])
            print(f"Episode {self.episode_count}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}")
    
    def on_train_end(self, logs={}):
        """Called at the end of training"""
        print(f"\nTraining completed! Total episodes: {self.episode_count}")
        print(f"Final average reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        
        # Save training plots
        plot_training_progress(
            self.episode_rewards, 
            self.episode_lengths, 
            save_path='training_progress.png'
        )


def create_dqn_agent(env, model, memory_limit=1000000):
    """
    Create DQN agent with specified configuration
    
    Args:
        env: Gym environment
        model: Keras model
        memory_limit (int): Maximum memory size
        
    Returns:
        DQNAgent: Configured DQN agent
    """
    # Memory for experience replay
    memory = SequentialMemory(limit=memory_limit, window_length=4)
    
    # Epsilon-greedy policy with linear annealing
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )
    
    # Create DQN agent
    dqn = DQNAgent(
        model=model,
        nb_actions=env.action_space.n,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=10000,
        policy=policy,
        enable_double_dqn=True,
        enable_dueling_network=True,
        dueling_type='avg'
    )
    
    # Compile agent
    dqn.compile(Adam(learning_rate=1e-4), metrics=['mae'])
    
    return dqn


def train_agent(env, agent, total_steps=1000000, save_weights=True):
    """
    Train the DQN agent
    
    Args:
        env: Gym environment
        agent: DQN agent
        total_steps (int): Total training steps
        save_weights (bool): Whether to save weights
        
    Returns:
        dict: Training history
    """
    print("Starting training...")
    print(f"Total steps: {total_steps:,}")
    print(f"Environment: {env.spec.id}")
    print(f"Action space: {env.action_space.n} actions")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Create callbacks
    callbacks = [TrainingCallback()]
    
    if save_weights:
        # Save weights every 100k steps
        weights_filename = 'dqn_spaceinvaders_weights_{step}.h5f'
        checkpoint_callback = ModelIntervalCheckpoint(
            weights_filename, 
            interval=100000, 
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Log training data
        log_filename = 'dqn_spaceinvaders_log.json'
        file_logger = FileLogger(log_filename, interval=1000)
        callbacks.append(file_logger)
    
    # Train the agent
    history = agent.fit(
        env,
        nb_steps=total_steps,
        visualize=False,
        verbose=1,
        callbacks=callbacks
    )
    
    # Save final weights
    if save_weights:
        final_weights_path = 'dqn_spaceinvaders_weights.h5f'
        agent.save_weights(final_weights_path, overwrite=True)
        print(f"Final weights saved to: {final_weights_path}")
    
    return history


def evaluate_agent(env, agent, nb_episodes=10, visualize=False):
    """
    Evaluate the trained agent
    
    Args:
        env: Gym environment
        agent: DQN agent
        nb_episodes (int): Number of episodes to evaluate
        visualize (bool): Whether to visualize episodes
        
    Returns:
        dict: Evaluation results
    """
    print(f"Evaluating agent for {nb_episodes} episodes...")
    
    # Test the agent
    test_results = agent.test(
        env, 
        nb_episodes=nb_episodes, 
        visualize=visualize, 
        verbose=1
    )
    
    # Calculate statistics
    episode_rewards = test_results.history['episode_reward']
    episode_lengths = test_results.history['nb_steps']
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    print(f"Evaluation Results:")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    
    return results


def main():
    """
    Main training function
    """
    print("=" * 60)
    print("Space Invaders DQN Training")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create environment
    print("Creating environment...")
    env = create_environment()
    print_environment_info(env)
    
    # Create model
    print("\nCreating DQN model...")
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    model = create_dqn_model(input_shape, num_actions)
    get_model_summary(model)
    
    # Create agent
    print("\nCreating DQN agent...")
    agent = create_dqn_agent(env, model)
    
    # Train agent
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    training_history = train_agent(env, agent, total_steps=1000000)
    
    # Evaluate agent
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    
    evaluation_results = evaluate_agent(env, agent, nb_episodes=10)
    
    # Save training summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'total_steps': 1000000,
        'evaluation_episodes': 10,
        'evaluation_results': evaluation_results,
        'model_architecture': {
            'input_shape': input_shape,
            'num_actions': num_actions,
            'total_params': model.count_params()
        }
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to: training_summary.json")
    print("Training completed successfully!")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()


