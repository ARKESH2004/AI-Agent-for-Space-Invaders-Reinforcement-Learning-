"""
CNN Model Architecture for Space Invaders DQN
Defines the Convolutional Neural Network used for Deep Q-Learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def create_dqn_model(input_shape, num_actions, learning_rate=1e-4):
    """
    Create a Deep Q-Network model for Space Invaders
    
    Args:
        input_shape (tuple): Shape of input frames (height, width, channels)
        num_actions (int): Number of possible actions
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        keras.Model: Compiled DQN model
    """
    model = keras.Sequential([
        # Input layer
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
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def get_model_summary(model):
    """
    Print model architecture summary
    
    Args:
        model (keras.Model): The model to summarize
    """
    print("=" * 50)
    print("DQN Model Architecture Summary")
    print("=" * 50)
    model.summary()
    print("=" * 50)
    print(f"Total parameters: {model.count_params():,}")
    print("=" * 50)


if __name__ == "__main__":
    # Test the model creation
    input_shape = (84, 84, 4)  # Height, Width, Channels (4 stacked frames)
    num_actions = 6  # Space Invaders has 6 possible actions
    
    model = create_dqn_model(input_shape, num_actions)
    get_model_summary(model)


