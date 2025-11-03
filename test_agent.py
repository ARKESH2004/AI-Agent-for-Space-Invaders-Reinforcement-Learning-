"""
Testing script for trained Space Invaders DQN Agent
Loads a trained model and evaluates its performance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from model import create_dqn_model
from utils import create_environment, print_environment_info


def load_trained_agent(weights_path, env):
    """
    Load a trained DQN agent from saved weights
    
    Args:
        weights_path (str): Path to saved weights file
        env: Gym environment
        
    Returns:
        DQNAgent: Loaded DQN agent
    """
    print(f"Loading trained agent from: {weights_path}")
    
    # Create model with same architecture as training
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    model = create_dqn_model(input_shape, num_actions)
    
    # Create agent with same configuration as training
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy(eps=0.05)  # Low epsilon for testing
    
    agent = DQNAgent(
        model=model,
        nb_actions=num_actions,
        memory=memory,
        nb_steps_warmup=0,  # No warmup needed for testing
        target_model_update=10000,
        policy=policy,
        enable_double_dqn=True,
        enable_dueling_network=True,
        dueling_type='avg'
    )
    
    # Compile agent
    agent.compile(Adam(learning_rate=1e-4), metrics=['mae'])
    
    # Load weights
    if os.path.exists(weights_path):
        agent.load_weights(weights_path)
        print("Weights loaded successfully!")
    else:
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    return agent


def test_agent_performance(env, agent, num_episodes=10, render=False):
    """
    Test agent performance over multiple episodes
    
    Args:
        env: Gym environment
        agent: DQN agent
        num_episodes (int): Number of episodes to test
        render (bool): Whether to render episodes
        
    Returns:
        dict: Test results
    """
    print(f"Testing agent for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    episode_frames = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        obs = env.reset()
        total_reward = 0
        step_count = 0
        frames = []
        
        done = False
        while not done:
            # Get action from agent
            action = agent.forward(obs)
            
            # Take step
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Store frame for visualization
            if render and len(frames) < 1000:  # Limit frames to prevent memory issues
                # Convert observation back to displayable format
                frame = obs[:, :, -1]  # Take the most recent frame
                frame = (frame * 255).astype(np.uint8)
                frames.append(frame)
            
            # Break if too many steps (safety)
            if step_count > 10000:
                print("  Episode terminated due to step limit")
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_frames.append(frames)
        
        print(f"  Reward: {total_reward:.2f}, Steps: {step_count}")
    
    # Calculate statistics
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_frames': episode_frames,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    return results


def create_performance_plots(results, save_path=None):
    """
    Create performance visualization plots
    
    Args:
        results (dict): Test results
        save_path (str): Path to save plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(results['episode_rewards'], 'b-', linewidth=2)
    ax1.axhline(y=results['mean_reward'], color='r', linestyle='--', 
                label=f'Mean: {results["mean_reward"]:.2f}')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Episode lengths
    ax2.plot(results['episode_lengths'], 'g-', linewidth=2)
    ax2.axhline(y=results['mean_length'], color='r', linestyle='--',
                label=f'Mean: {results["mean_length"]:.2f}')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True)
    
    # Reward distribution
    ax3.hist(results['episode_rewards'], bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=results['mean_reward'], color='r', linestyle='--',
                label=f'Mean: {results["mean_reward"]:.2f}')
    ax3.set_title('Reward Distribution')
    ax3.set_xlabel('Total Reward')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True)
    
    # Length distribution
    ax4.hist(results['episode_lengths'], bins=10, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=results['mean_length'], color='r', linestyle='--',
                label=f'Mean: {results["mean_length"]:.2f}')
    ax4.set_title('Length Distribution')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved to: {save_path}")
    
    plt.show()


def save_episode_gif(frames, save_path, duration=50):
    """
    Save episode frames as GIF
    
    Args:
        frames (list): List of frames from an episode
        save_path (str): Path to save the GIF
        duration (int): Duration between frames in milliseconds
    """
    if not frames:
        print("No frames to save")
        return
    
    try:
        import imageio
        
        # Convert frames to RGB for GIF
        rgb_frames = []
        for frame in frames:
            if len(frame.shape) == 2:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb_frame = frame
            rgb_frames.append(rgb_frame)
        
        # Save as GIF
        imageio.mimsave(save_path, rgb_frames, duration=duration)
        print(f"Episode GIF saved to: {save_path}")
        
    except ImportError:
        print("imageio not available. Install with: pip install imageio")
    except Exception as e:
        print(f"Error saving GIF: {e}")


def print_test_summary(results):
    """
    Print test results summary
    
    Args:
        results (dict): Test results
    """
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Number of Episodes: {len(results['episode_rewards'])}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min Reward: {results['min_reward']:.2f}")
    print(f"Max Reward: {results['max_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print("=" * 60)


def main():
    """
    Main testing function
    """
    print("=" * 60)
    print("Space Invaders DQN Agent Testing")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check for weights file
    weights_path = 'dqn_spaceinvaders_weights.h5f'
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found: {weights_path}")
        print("Please train the agent first using train.py")
        return
    
    # Create environment
    print("Creating environment...")
    env = create_environment(render_mode='rgb_array')
    print_environment_info(env)
    
    # Load trained agent
    print("\nLoading trained agent...")
    agent = load_trained_agent(weights_path, env)
    
    # Test agent performance
    print("\n" + "=" * 60)
    print("TESTING PHASE")
    print("=" * 60)
    
    # Test without rendering first
    results = test_agent_performance(env, agent, num_episodes=10, render=False)
    print_test_summary(results)
    
    # Create performance plots
    create_performance_plots(results, save_path='test_performance.png')
    
    # Test with rendering for visualization
    print("\nTesting with rendering...")
    render_results = test_agent_performance(env, agent, num_episodes=3, render=True)
    
    # Save best episode as GIF
    if render_results['episode_frames']:
        best_episode_idx = np.argmax(render_results['episode_rewards'])
        best_frames = render_results['episode_frames'][best_episode_idx]
        
        if best_frames:
            gif_path = f'best_episode_{best_episode_idx + 1}.gif'
            save_episode_gif(best_frames, gif_path)
    
    # Save test results
    test_summary = {
        'test_date': datetime.now().isoformat(),
        'weights_file': weights_path,
        'num_episodes': len(results['episode_rewards']),
        'results': {
            'mean_reward': float(results['mean_reward']),
            'std_reward': float(results['std_reward']),
            'min_reward': float(results['min_reward']),
            'max_reward': float(results['max_reward']),
            'mean_length': float(results['mean_length']),
            'std_length': float(results['std_length'])
        }
    }
    
    import json
    with open('test_summary.json', 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    print(f"\nTest summary saved to: test_summary.json")
    print("Testing completed successfully!")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()


