# 🚀 Space Invaders AI - Deep Reinforcement Learning Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)](https://tensorflow.org)
[![Keras-RL2](https://img.shields.io/badge/Keras--RL2-1.0.5-red)](https://github.com/wau/keras-rl2)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0.26.2-green)](https://gym.openai.com)

This project trains an AI agent to play Space Invaders using **Deep Q-Learning (DQN)** with a Convolutional Neural Network. The agent learns to maximize its score by processing game frames and making optimal decisions through reinforcement learning.

## 🎯 Project Overview

The AI agent uses a **Deep Q-Network (DQN)** implemented with Keras-RL2 to learn optimal strategies for playing Space Invaders. The model processes preprocessed game frames through a CNN architecture and learns to predict Q-values for different actions.

### Key Features

- 🧠 **Deep Q-Network (DQN)** with experience replay and target network
- 🖼️ **CNN Architecture** optimized for Atari game frames
- 🎮 **Frame Preprocessing** (grayscale, resize, normalization, stacking)
- 📊 **Training Visualization** with progress plots and metrics
- 🎯 **Agent Evaluation** with performance analysis
- 🎬 **Demo Mode** with visual gameplay
- 📈 **Comprehensive Logging** and result saving

## 🏗️ Project Structure

```
space_invader_rl/
│
├── main.py                # 🚀 Main entry point with CLI interface
├── model.py               # 🧠 CNN model architecture definition
├── train.py               # 🎯 DQN agent training logic
├── test_agent.py          # 🧪 Agent evaluation and testing
├── utils.py               # 🔧 Frame preprocessing and utilities
├── requirements.txt       # 📦 Project dependencies
└── README.md              # 📖 This documentation
```

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd space_invader_rl

# Install dependencies
pip install -r requirements.txt
```

### 2. Check Setup

Verify everything is working correctly:

```bash
python main.py --mode check
```

### 3. Train the Agent

Start training the DQN agent (this will take several hours):

```bash
python main.py --mode train
```

### 4. Test the Trained Agent

Evaluate the trained agent's performance:

```bash
python main.py --mode test
```

### 5. Run Demo

Watch the agent play with visual rendering:

```bash
python main.py --mode demo --episodes 5
```

## 🧠 Model Architecture

The DQN uses a Convolutional Neural Network with the following architecture:

```
Input: (84, 84, 4) - 4 stacked grayscale frames
│
├── Conv2D(32, 8x8, stride=4) + ReLU
├── Conv2D(64, 4x4, stride=2) + ReLU  
├── Conv2D(64, 3x3, stride=1) + ReLU
├── Flatten()
├── Dense(512) + ReLU
└── Dense(6) - Q-values for each action
```

### Key Components

- **Input**: 4 stacked 84x84 grayscale frames
- **Convolutional Layers**: Extract spatial features from game frames
- **Dense Layers**: Process features and output Q-values
- **Actions**: 6 possible actions (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)

## ⚙️ Training Configuration

### DQN Parameters

- **Memory Size**: 1,000,000 experiences
- **Learning Rate**: 1e-4
- **Epsilon Decay**: Linear from 1.0 to 0.1 over 1M steps
- **Target Network Update**: Every 10,000 steps
- **Warmup Steps**: 50,000
- **Training Steps**: 1,000,000

### Frame Preprocessing

1. **Grayscale Conversion**: Convert RGB to grayscale
2. **Resize**: Scale to 84x84 pixels
3. **Normalization**: Scale pixel values to [0, 1]
4. **Frame Stacking**: Stack 4 consecutive frames for temporal information
5. **Frame Skipping**: Skip 4 frames per action for efficiency

## 📊 Training Process

The training process includes:

1. **Environment Setup**: Create and wrap Space Invaders environment
2. **Model Creation**: Build CNN architecture
3. **Agent Configuration**: Set up DQN with experience replay
4. **Training Loop**: Train for 1M steps with progress monitoring
5. **Weight Saving**: Save model weights every 100k steps
6. **Evaluation**: Test agent performance periodically

### Training Output

- **Progress Logs**: Episode rewards and lengths
- **Weight Checkpoints**: Saved every 100k steps
- **Training Plots**: Reward curves and performance metrics
- **Final Weights**: `dqn_spaceinvaders_weights.h5f`

## 🧪 Testing and Evaluation

### Test Metrics

- **Episode Rewards**: Total score per episode
- **Episode Lengths**: Number of steps per episode
- **Performance Statistics**: Mean, std, min, max values
- **Visualization**: Performance plots and distributions

### Test Output

- **Performance Plots**: Reward and length distributions
- **Test Summary**: JSON file with detailed results
- **Gameplay GIFs**: Visual recordings of best episodes
- **Statistics**: Comprehensive performance analysis

## 🎮 Usage Examples

### Command Line Interface

```bash
# Check dependencies and environment
python main.py --mode check

# Train the agent (default: 1M steps)
python main.py --mode train

# Test the trained agent (default: 10 episodes)
python main.py --mode test

# Run demo with 3 episodes
python main.py --mode demo --episodes 3

# Use custom random seed
python main.py --mode train --seed 123
```

### Programmatic Usage

```python
from utils import create_environment
from model import create_dqn_model
from train import create_dqn_agent, train_agent

# Create environment
env = create_environment()

# Create model and agent
model = create_dqn_model(env.observation_space.shape, env.action_space.n)
agent = create_dqn_agent(env, model)

# Train agent
history = train_agent(env, agent, total_steps=1000000)
```

## 📈 Expected Results

### Training Progress

- **Initial Performance**: Random actions, low scores
- **Learning Phase**: Gradual improvement in scores
- **Convergence**: Stable performance after ~500k steps
- **Final Performance**: Average score of 200-500+ points

### Performance Metrics

- **Training Time**: 6-12 hours (depending on hardware)
- **Memory Usage**: ~2-4 GB RAM
- **Final Epsilon**: 0.1 (10% random actions)
- **Convergence**: Typically after 500k-800k steps

## 🔧 Configuration Options

### Training Parameters

```python
# In train.py
TOTAL_STEPS = 1000000        # Total training steps
MEMORY_LIMIT = 1000000       # Experience replay buffer size
LEARNING_RATE = 1e-4         # Adam optimizer learning rate
EPSILON_MAX = 1.0            # Initial exploration rate
EPSILON_MIN = 0.1            # Final exploration rate
TARGET_UPDATE = 10000        # Target network update frequency
```

### Environment Settings

```python
# In utils.py
FRAME_SIZE = (84, 84)        # Preprocessed frame size
FRAME_STACK = 4              # Number of frames to stack
FRAME_SKIP = 4               # Frames to skip per action
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce batch size or memory limit
   ```python
   memory = SequentialMemory(limit=500000, window_length=4)
   ```

3. **Training Slow**: Use GPU acceleration
   ```python
   # TensorFlow will automatically use GPU if available
   ```

4. **Environment Errors**: Check Atari ROM installation
   ```bash
   pip install gym[atari]
   ```

### Performance Tips

- **GPU Usage**: Training is much faster with GPU
- **Memory Management**: Monitor RAM usage during training
- **Checkpointing**: Save weights regularly to resume training
- **Visualization**: Use demo mode to monitor agent behavior

## 📚 Dependencies

### Core Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.13.0
- **Keras-RL2**: 1.0.5
- **OpenAI Gym**: 0.26.2
- **OpenCV**: 4.10.0.84
- **NumPy**: 1.24.3
- **Matplotlib**: 3.7.2

### Optional Dependencies

- **ImageIO**: For GIF creation
- **TensorBoard**: For training visualization
- **Jupyter**: For interactive development

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:

- Bug reports
- Feature requests
- Code improvements
- Documentation updates

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **OpenAI Gym**: For the Atari environment
- **Keras-RL2**: For the DQN implementation
- **TensorFlow/Keras**: For the deep learning framework
- **DeepMind**: For the original DQN paper

## 📖 References

1. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
2. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
3. [Keras-RL2 Documentation](https://github.com/wau/keras-rl2)
4. [OpenAI Gym Documentation](https://gym.openai.com/)

## 🎯 Future Enhancements

- [ ] Double DQN implementation
- [ ] Prioritized Experience Replay
- [ ] Rainbow DQN features
- [ ] Multi-agent training
- [ ] Real-time visualization
- [ ] Model comparison tools

---

**Happy Learning! 🚀**

*Train your AI agent to become a Space Invaders champion!*


