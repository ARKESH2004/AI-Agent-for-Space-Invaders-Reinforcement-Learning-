"""
Simplified Space Invaders DQN Demo
A working demonstration without keras-rl2 compatibility issues
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import random
import time

# Try to import gymnasium, fall back to gym if needed
try:
    import gymnasium as gym
    from gymnasium import spaces
    print("✅ Using Gymnasium")
except ImportError:
    import gym
    from gym import spaces
    print("✅ Using OpenAI Gym")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FramePreprocessor:
    """Handles frame preprocessing for Space Invaders environment"""
    
    def __init__(self, target_size=(84, 84), frame_stack=4):
        self.target_size = target_size
        self.frame_stack = frame_stack
        self.frame_buffer = deque(maxlen=frame_stack)
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = frame
            
        # Resize frame
        resized_frame = cv2.resize(gray_frame, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        return normalized_frame
    
    def stack_frames(self, frame):
        """Stack frames for temporal information"""
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        
        # If buffer is not full, pad with zeros
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.append(np.zeros_like(processed_frame))
        
        stacked_frames = np.stack(self.frame_buffer, axis=-1)
        return stacked_frames
    
    def reset(self):
        """Reset the frame buffer"""
        self.frame_buffer.clear()


class SpaceInvadersWrapper(gym.Wrapper):
    """Custom wrapper for Space Invaders environment"""
    
    def __init__(self, env, frame_skip=4, frame_stack=4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frame_preprocessor = FramePreprocessor(frame_stack=frame_stack)
        
        # Modify action space to only include relevant actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: stacked frames
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(84, 84, frame_stack), 
            dtype=np.float32
        )
    
    def step(self, action):
        """Execute action and return preprocessed observation"""
        total_reward = 0
        done = False
        
        # Skip frames
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        # Preprocess and stack frames
        processed_obs = self.frame_preprocessor.stack_frames(obs)
        
        return processed_obs, total_reward, done, info
    
    def reset(self, **kwargs):
        """Reset environment and return initial observation"""
        obs = self.env.reset(**kwargs)
        self.frame_preprocessor.reset()
        processed_obs = self.frame_preprocessor.stack_frames(obs)
        return processed_obs


def create_dqn_model(input_shape, num_actions, learning_rate=1e-4):
    """Create a Deep Q-Network model for Space Invaders"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First convolutional layer
        layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation='relu',
            padding='valid',
            name='conv1'
        ),
        
        # Second convolutional layer
        layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation='relu',
            padding='valid',
            name='conv2'
        ),
        
        # Third convolutional layer
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
            padding='valid',
            name='conv3'
        ),
        
        # Flatten layer
        layers.Flatten(name='flatten'),
        
        # First dense layer
        layers.Dense(
            units=512,
            activation='relu',
            name='dense1'
        ),
        
        # Output layer (Q-values for each action)
        layers.Dense(
            units=num_actions,
            activation='linear',
            name='q_values'
        )
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_environment(render_mode='rgb_array'):
    """Create and wrap Space Invaders environment"""
    try:
        # Try gymnasium first
        env = gym.make('ALE/SpaceInvaders-v5', render_mode=render_mode)
    except:
        try:
            # Fall back to gym
            env = gym.make('SpaceInvaders-v0')
        except:
            # Create a mock environment for demo
            print("⚠️  Atari environment not available, creating mock environment")
            return create_mock_environment()
    
    # Wrap with custom wrapper
    wrapped_env = SpaceInvadersWrapper(env)
    return wrapped_env


def create_mock_environment():
    """Create a mock environment for demonstration when Atari is not available"""
    class MockEnv:
        def __init__(self):
            self.action_space = spaces.Discrete(6)
            self.observation_space = spaces.Box(low=0, high=1, shape=(84, 84, 4), dtype=np.float32)
            self.frame_preprocessor = FramePreprocessor()
            self.step_count = 0
            
        def reset(self, **kwargs):
            # Generate a random frame
            frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
            self.step_count = 0
            return self.frame_preprocessor.stack_frames(frame)
            
        def step(self, action):
            # Generate random frame
            frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
            self.step_count += 1
            
            # Random reward and done
            reward = random.uniform(-1, 10)
            done = self.step_count > 100 or random.random() < 0.1
            info = {}
            
            processed_obs = self.frame_preprocessor.stack_frames(frame)
            return processed_obs, reward, done, info
            
        def close(self):
            pass
    
    return MockEnv()


def demo_random_agent(env, num_episodes=5):
    """Demo with random actions"""
    print("🎮 Running Random Agent Demo")
    print("=" * 50)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        done = False
        while not done and step_count < 1000:
            # Random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"  Step {step_count}: Reward = {total_reward:.2f}")
        
        episode_rewards.append(total_reward)
        print(f"  Final Reward: {total_reward:.2f}")
        print()
    
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    return episode_rewards


def demo_trained_agent(env, model, num_episodes=3):
    """Demo with trained model (if available)"""
    print("🧠 Running Trained Agent Demo")
    print("=" * 50)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        done = False
        while not done and step_count < 1000:
            # Get Q-values from model
            q_values = model.predict(obs.reshape(1, *obs.shape), verbose=0)
            action = np.argmax(q_values[0])
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"  Step {step_count}: Reward = {total_reward:.2f}, Action = {action}")
        
        episode_rewards.append(total_reward)
        print(f"  Final Reward: {total_reward:.2f}")
        print()
    
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    return episode_rewards


def plot_results(random_rewards, trained_rewards=None):
    """Plot comparison of random vs trained agent"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(random_rewards, 'b-', linewidth=2, label='Random Agent')
    if trained_rewards:
        plt.plot(trained_rewards, 'r-', linewidth=2, label='Trained Agent')
    plt.title('Episode Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(random_rewards, bins=5, alpha=0.7, color='blue', label='Random Agent')
    if trained_rewards:
        plt.hist(trained_rewards, bins=5, alpha=0.7, color='red', label='Trained Agent')
    plt.title('Reward Distribution')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main demo function"""
    print("🚀 SPACE INVADERS DQN DEMO")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create environment
    print("Creating environment...")
    env = create_environment()
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space.shape}")
    print()
    
    # Create model
    print("Creating DQN model...")
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    model = create_dqn_model(input_shape, num_actions)
    print(f"Model created with {model.count_params():,} parameters")
    print()
    
    # Demo 1: Random agent
    random_rewards = demo_random_agent(env, num_episodes=5)
    
    # Demo 2: Trained agent (with untrained model)
    print("Note: Using untrained model for demonstration")
    trained_rewards = demo_trained_agent(env, model, num_episodes=3)
    
    # Plot results
    print("📊 Generating results plot...")
    plot_results(random_rewards, trained_rewards)
    
    print("✅ Demo completed successfully!")
    print("📁 Results saved to: demo_results.png")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()


