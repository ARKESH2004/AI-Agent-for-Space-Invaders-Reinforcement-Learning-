"""
Quick Space Invaders DQN Demo for Presentation
Shows model architecture, training process, and results
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

def create_dqn_model(input_shape, num_actions):
    """Create the DQN model architecture"""
    model = keras.Sequential([
        layers.Input(shape=input_shape, name='input'),
        
        # Convolutional layers
        layers.Conv2D(32, 8, strides=4, activation='relu', padding='valid', name='conv1'),
        layers.Conv2D(64, 4, strides=2, activation='relu', padding='valid', name='conv2'),
        layers.Conv2D(64, 3, strides=1, activation='relu', padding='valid', name='conv3'),
        
        # Dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='dense1'),
        layers.Dense(num_actions, activation='linear', name='output')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def simulate_training_progress():
    """Simulate training progress with realistic curves"""
    episodes = np.arange(0, 1000, 10)
    
    # Simulate learning curve
    base_reward = 20
    learning_curve = base_reward + 50 * (1 - np.exp(-episodes / 200)) + np.random.normal(0, 5, len(episodes))
    learning_curve = np.maximum(learning_curve, 0)  # No negative rewards
    
    # Simulate epsilon decay
    epsilon = 1.0 * np.exp(-episodes / 300)
    epsilon = np.maximum(epsilon, 0.1)
    
    return episodes, learning_curve, epsilon

def plot_training_visualization(episodes, rewards, epsilon):
    """Create comprehensive training visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(episodes, rewards, 'b-', linewidth=2, alpha=0.7)
    ax1.set_title('🎯 Episode Rewards Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(rewards[-100:]), color='r', linestyle='--', 
                label=f'Final Avg: {np.mean(rewards[-100:]):.1f}')
    ax1.legend()
    
    # Epsilon decay
    ax2.plot(episodes, epsilon, 'g-', linewidth=2)
    ax2.set_title('🔍 Exploration Rate (Epsilon)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon Value')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Reward distribution
    ax3.hist(rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=np.mean(rewards), color='r', linestyle='--', 
                label=f'Mean: {np.mean(rewards):.1f}')
    ax3.set_title('📊 Reward Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Total Reward')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Moving average
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax4.plot(episodes[window-1:], moving_avg, 'purple', linewidth=3, label=f'{window}-Episode Moving Avg')
    ax4.plot(episodes, rewards, 'b-', alpha=0.3, label='Raw Rewards')
    ax4.set_title('📈 Learning Progress (Moving Average)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Reward')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_model_architecture():
    """Show the model architecture"""
    print("🧠 DQN Model Architecture")
    print("=" * 50)
    
    # Create model
    input_shape = (84, 84, 4)
    num_actions = 6
    model = create_dqn_model(input_shape, num_actions)
    
    # Print model summary
    model.summary()
    
    # Create architecture diagram
    print("\n📊 Model Architecture Diagram:")
    print("Input: (84, 84, 4) - 4 stacked grayscale frames")
    print("├── Conv2D(32, 8x8, stride=4) + ReLU")
    print("├── Conv2D(64, 4x4, stride=2) + ReLU")
    print("├── Conv2D(64, 3x3, stride=1) + ReLU")
    print("├── Flatten()")
    print("├── Dense(512) + ReLU")
    print("└── Dense(6) - Q-values for each action")
    print(f"\nTotal Parameters: {model.count_params():,}")
    
    return model

def simulate_gameplay():
    """Simulate gameplay with visual representation"""
    print("\n🎮 Simulated Gameplay")
    print("=" * 50)
    
    # Create a simple game state representation
    game_states = []
    actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    
    # Simulate 20 steps of gameplay
    for step in range(20):
        # Simulate game state (simplified)
        score = step * 10 + np.random.randint(0, 20)
        lives = max(1, 3 - step // 7)
        action = np.random.randint(0, 6)
        
        game_states.append({
            'step': step + 1,
            'score': score,
            'lives': lives,
            'action': actions[action],
            'reward': np.random.randint(-1, 15)
        })
        
        print(f"Step {step+1:2d}: Score={score:3d}, Lives={lives}, Action={actions[action]:10s}, Reward={game_states[-1]['reward']:2d}")
        
        if step % 5 == 4:  # Pause every 5 steps
            time.sleep(0.5)
    
    return game_states

def main():
    """Main presentation demo"""
    print("🚀 SPACE INVADERS DQN - PRESENTATION DEMO")
    print("=" * 60)
    
    # 1. Model Architecture
    print("\n1️⃣ MODEL ARCHITECTURE")
    print("-" * 30)
    model = demonstrate_model_architecture()
    
    # 2. Training Simulation
    print("\n2️⃣ TRAINING SIMULATION")
    print("-" * 30)
    episodes, rewards, epsilon = simulate_training_progress()
    print(f"Simulated {len(episodes)} episodes of training")
    print(f"Final average reward: {np.mean(rewards[-100:]):.1f}")
    print(f"Final epsilon: {epsilon[-1]:.3f}")
    
    # 3. Training Visualization
    print("\n3️⃣ TRAINING VISUALIZATION")
    print("-" * 30)
    plot_training_visualization(episodes, rewards, epsilon)
    print("📁 Training visualization saved to: training_visualization.png")
    
    # 4. Gameplay Simulation
    print("\n4️⃣ GAMEPLAY SIMULATION")
    print("-" * 30)
    game_states = simulate_gameplay()
    
    # 5. Performance Summary
    print("\n5️⃣ PERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"🎯 Model Parameters: {model.count_params():,}")
    print(f"📊 Training Episodes: {len(episodes)}")
    print(f"🏆 Best Episode Reward: {np.max(rewards):.1f}")
    print(f"📈 Average Final Reward: {np.mean(rewards[-100:]):.1f}")
    print(f"🎮 Actions Available: 6 (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)")
    print(f"🖼️ Input Shape: (84, 84, 4) - 4 stacked grayscale frames")
    
    print("\n✅ PRESENTATION DEMO COMPLETED!")
    print("📁 Generated files:")
    print("   - training_visualization.png")
    print("   - Model architecture summary above")

if __name__ == "__main__":
    main()


