"""
Main entry point for Space Invaders DQN Project
Provides a command-line interface for training and testing the agent
"""

import argparse
import sys
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from train import main as train_main
from test_agent import main as test_main
from utils import create_environment, print_environment_info


def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🚀 SPACE INVADERS DQN PROJECT 🚀                ║
    ║                                                              ║
    ║         Deep Reinforcement Learning with Keras & TF         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'tensorflow',
        'keras-rl2',
        'gym',
        'gymnasium',
        'opencv-python',
        'numpy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True


def test_environment():
    """Test if the environment can be created and run"""
    try:
        print("Testing environment setup...")
        env = create_environment()
        print_environment_info(env)
        
        # Test a few steps
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}")
            if done:
                break
        
        env.close()
        print("✅ Environment test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def train_mode(args):
    """Run training mode"""
    print("🎯 Starting Training Mode")
    print("=" * 50)
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Run training
    train_main()


def test_mode(args):
    """Run testing mode"""
    print("🧪 Starting Testing Mode")
    print("=" * 50)
    
    # Check if weights file exists
    weights_path = 'dqn_spaceinvaders_weights.h5f'
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file not found: {weights_path}")
        print("Please train the agent first using: python main.py --mode train")
        return
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Run testing
    test_main()


def demo_mode(args):
    """Run demo mode with visualization"""
    print("🎮 Starting Demo Mode")
    print("=" * 50)
    
    # Check if weights file exists
    weights_path = 'dqn_spaceinvaders_weights.h5f'
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file not found: {weights_path}")
        print("Please train the agent first using: python main.py --mode train")
        return
    
    try:
        from test_agent import load_trained_agent, test_agent_performance
        
        # Create environment with rendering
        env = create_environment(render_mode='human')
        
        # Load trained agent
        agent = load_trained_agent(weights_path, env)
        
        # Run demo episodes
        print("Running demo episodes...")
        results = test_agent_performance(env, agent, num_episodes=args.episodes, render=True)
        
        print(f"\nDemo completed! Average reward: {results['mean_reward']:.2f}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Demo mode failed: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Space Invaders DQN - Deep Reinforcement Learning Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train                    # Train the agent
  python main.py --mode test                     # Test the trained agent
  python main.py --mode demo --episodes 3        # Run demo with 3 episodes
  python main.py --mode check                    # Check dependencies and environment
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'demo', 'check'],
        default='check',
        help='Mode to run: train, test, demo, or check dependencies'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes for demo mode (default: 5)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='Skip printing the banner'
    )
    
    args = parser.parse_args()
    
    # Print banner
    if not args.no_banner:
        print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.mode == 'check':
            print("🔍 Checking environment...")
            if test_environment():
                print("\n✅ All checks passed! Ready to train or test.")
            else:
                print("\n❌ Environment check failed!")
                sys.exit(1)
                
        elif args.mode == 'train':
            train_mode(args)
            
        elif args.mode == 'test':
            test_mode(args)
            
        elif args.mode == 'demo':
            demo_mode(args)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


