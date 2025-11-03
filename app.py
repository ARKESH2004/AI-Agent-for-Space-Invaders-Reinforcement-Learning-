"""
Flask Backend API for Space Invaders DQN Frontend
Provides REST API endpoints for training, testing, and demo modes
"""

import os
import json
import base64
import io
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import time

# Import the existing modules
try:
    from utils import create_environment, FramePreprocessor
    from model import create_dqn_model
    from train import create_dqn_agent, train_agent, evaluate_agent
    from test_agent import load_trained_agent, test_agent_performance
except ImportError as e:
    print(f"Warning: Some ML modules not available: {e}")
    # Create mock functions for demo purposes
    def create_environment(*args, **kwargs):
        return None
    def create_dqn_model(*args, **kwargs):
        return None
    def create_dqn_agent(*args, **kwargs):
        return None
    def train_agent(*args, **kwargs):
        return None
    def evaluate_agent(*args, **kwargs):
        return None
    def load_trained_agent(*args, **kwargs):
        return None
    def test_agent_performance(*args, **kwargs):
        return None

app = Flask(__name__)
CORS(app)

# Global variables for training state
training_state = {
    'is_training': False,
    'progress': 0,
    'current_episode': 0,
    'current_reward': 0,
    'episode_rewards': [],
    'episode_lengths': [],
    'training_log': []
}

# Global agent and environment
agent = None
env = None
model = None

def create_app():
    """Create and configure the Flask app"""
    app.config['SECRET_KEY'] = 'space_invaders_dqn_secret_key'
    return app

def frame_to_base64(frame):
    """Convert frame to base64 string for frontend display"""
    if frame is None:
        return None
    
    # Convert frame to RGB if needed
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]  # Remove alpha channel
    
    # Ensure frame is in correct format
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{frame_base64}"

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'training_active': training_state['is_training']
    })

@app.route('/api/setup', methods=['POST'])
def setup_environment():
    """Initialize the environment and model"""
    global env, model, agent
    
    try:
        # Create environment
        env = create_environment(render_mode='rgb_array')
        
        # Create model
        input_shape = env.observation_space.shape
        num_actions = env.action_space.n
        model = create_dqn_model(input_shape, num_actions)
        
        # Create agent
        agent = create_dqn_agent(env, model)
        
        return jsonify({
            'success': True,
            'message': 'Environment and model initialized successfully',
            'environment_info': {
                'action_space': env.action_space.n,
                'observation_shape': list(env.observation_space.shape),
                'model_parameters': model.count_params()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start training the agent"""
    global training_state, agent, env
    
    if training_state['is_training']:
        return jsonify({
            'success': False,
            'error': 'Training is already in progress'
        }), 400
    
    if agent is None or env is None:
        return jsonify({
            'success': False,
            'error': 'Environment and model not initialized'
        }), 400
    
    # Get training parameters from request
    data = request.get_json() or {}
    total_steps = data.get('total_steps', 100000)
    
    # Reset training state
    training_state.update({
        'is_training': True,
        'progress': 0,
        'current_episode': 0,
        'current_reward': 0,
        'episode_rewards': [],
        'episode_lengths': [],
        'training_log': []
    })
    
    # Start training in a separate thread
    def train_worker():
        global training_state
        
        try:
            # Custom training callback to update state
            class TrainingCallback:
                def __init__(self):
                    self.episode_count = 0
                
                def on_episode_end(self, episode, logs={}):
                    self.episode_count += 1
                    episode_reward = logs.get('episode_reward', 0)
                    episode_length = logs.get('nb_steps', 0)
                    
                    training_state['current_episode'] = self.episode_count
                    training_state['current_reward'] = episode_reward
                    training_state['episode_rewards'].append(episode_reward)
                    training_state['episode_lengths'].append(episode_length)
                    training_state['progress'] = min(100, (episode / (total_steps / 1000)) * 100)
                    
                    # Add to training log
                    training_state['training_log'].append({
                        'episode': self.episode_count,
                        'reward': episode_reward,
                        'length': episode_length,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Train the agent
            history = train_agent(env, agent, total_steps=total_steps, save_weights=True)
            
            training_state['is_training'] = False
            training_state['progress'] = 100
            
        except Exception as e:
            training_state['is_training'] = False
            training_state['error'] = str(e)
    
    # Start training thread
    training_thread = threading.Thread(target=train_worker)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started successfully',
        'total_steps': total_steps
    })

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    return jsonify(training_state)

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    """Stop training"""
    global training_state
    
    if not training_state['is_training']:
        return jsonify({
            'success': False,
            'error': 'No training in progress'
        }), 400
    
    training_state['is_training'] = False
    
    return jsonify({
        'success': True,
        'message': 'Training stopped'
    })

@app.route('/api/test/run', methods=['POST'])
def run_test():
    """Run agent testing"""
    global agent, env
    
    if agent is None or env is None:
        return jsonify({
            'success': False,
            'error': 'Environment and model not initialized'
        }), 400
    
    data = request.get_json() or {}
    num_episodes = data.get('episodes', 5)
    
    try:
        # Run test
        results = test_agent_performance(env, agent, num_episodes=num_episodes, render=False)
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/demo/step', methods=['POST'])
def demo_step():
    """Run one step of demo"""
    global agent, env
    
    if agent is None or env is None:
        return jsonify({
            'success': False,
            'error': 'Environment and model not initialized'
        }), 400
    
    try:
        # Get current state
        if not hasattr(demo_step, 'obs') or demo_step.obs is None:
            demo_step.obs = env.reset()
            demo_step.total_reward = 0
            demo_step.step_count = 0
            demo_step.done = False
        
        if demo_step.done:
            # Reset for new episode
            demo_step.obs = env.reset()
            demo_step.total_reward = 0
            demo_step.step_count = 0
            demo_step.done = False
        
        # Get action from agent
        action = agent.forward(demo_step.obs)
        
        # Take step
        demo_step.obs, reward, demo_step.done, info = env.step(action)
        demo_step.total_reward += reward
        demo_step.step_count += 1
        
        # Get frame for visualization
        frame = env.render()
        frame_base64 = frame_to_base64(frame)
        
        return jsonify({
            'success': True,
            'action': int(action),
            'reward': float(reward),
            'total_reward': float(demo_step.total_reward),
            'step_count': demo_step.step_count,
            'done': demo_step.done,
            'frame': frame_base64
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/demo/reset', methods=['POST'])
def demo_reset():
    """Reset demo environment"""
    global env
    
    if env is None:
        return jsonify({
            'success': False,
            'error': 'Environment not initialized'
        }), 400
    
    try:
        obs = env.reset()
        frame = env.render()
        frame_base64 = frame_to_base64(frame)
        
        # Reset demo state
        demo_step.obs = obs
        demo_step.total_reward = 0
        demo_step.step_count = 0
        demo_step.done = False
        
        return jsonify({
            'success': True,
            'frame': frame_base64,
            'message': 'Demo environment reset'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    global model
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not initialized'
        }), 400
    
    return jsonify({
        'success': True,
        'model_info': {
            'total_parameters': model.count_params(),
            'input_shape': list(model.input_shape[1:]) if model.input_shape else None,
            'output_shape': list(model.output_shape[1:]) if model.output_shape else None,
            'layers': len(model.layers)
        }
    })

@app.route('/api/plots/training', methods=['GET'])
def get_training_plots():
    """Generate and return training progress plots"""
    global training_state
    
    if not training_state['episode_rewards']:
        return jsonify({
            'success': False,
            'error': 'No training data available'
        }), 400
    
    try:
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot episode rewards
        ax1.plot(training_state['episode_rewards'])
        ax1.set_title('Episode Rewards Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(training_state['episode_lengths'])
        ax2.set_title('Episode Lengths Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        
        # Convert to base64
        plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'success': True,
            'plot': f"data:image/png;base64,{plot_base64}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/weights/load', methods=['POST'])
def load_weights():
    """Load pre-trained weights"""
    global agent
    
    if agent is None:
        return jsonify({
            'success': False,
            'error': 'Agent not initialized'
        }), 400
    
    data = request.get_json() or {}
    weights_path = data.get('weights_path', 'dqn_spaceinvaders_weights.h5f')
    
    try:
        if os.path.exists(weights_path):
            agent.load_weights(weights_path)
            return jsonify({
                'success': True,
                'message': f'Weights loaded from {weights_path}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Weights file not found: {weights_path}'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/weights/save', methods=['POST'])
def save_weights():
    """Save current weights"""
    global agent
    
    if agent is None:
        return jsonify({
            'success': False,
            'error': 'Agent not initialized'
        }), 400
    
    data = request.get_json() or {}
    weights_path = data.get('weights_path', f'dqn_spaceinvaders_weights_{int(time.time())}.h5f')
    
    try:
        agent.save_weights(weights_path)
        return jsonify({
            'success': True,
            'message': f'Weights saved to {weights_path}',
            'weights_path': weights_path
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
