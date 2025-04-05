import numpy as np
import random
from tensorflow import keras
from .replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_dqn_model()
        # Add target model for more stable learning
        self.target_model = self.build_dqn_model() 
        self.update_target_model()  # Initialize target model with same weights
        self.target_update_freq = 100  # Update target network every 100 steps
        self.update_count = 0

    def build_dqn_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def update_target_model(self):
        """Update target model with weights from main model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target model periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_model()

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        # Ensure the file has the correct .weights.h5 extension
        if not name.endswith('.h5'):
            if name.endswith('.weights'):
                name = name + '.h5'
            else:
                name = name + '.weights.h5'
        self.model.save_weights(name)