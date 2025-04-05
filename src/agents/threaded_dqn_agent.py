from agents.dqn_agent import DQNAgent
import threading
import tensorflow as tf
import numpy as np
import time
import random

class ThreadedDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.is_training = False
        self.training_thread = None
        self.training_lock = threading.Lock()
        self.stopping = False
        
        # Enable soft device placement
        tf.config.set_soft_device_placement(True)
        
        # Explicitly create the model on CPU
        with tf.device('/CPU:0'):
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.target_model.set_weights(self.model.get_weights())
        
    def start_background_training(self):
        """Start background training thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.stopping = False
            self.training_thread = threading.Thread(target=self._background_training)
            self.training_thread.daemon = True  # Thread will exit when main program exits
            self.training_thread.start()
            print("Background training started")
    
    def stop_background_training(self):
        """Signal the background thread to stop"""
        self.stopping = True
        if self.training_thread:
            self.training_thread.join(timeout=1.0)
    
    def _background_training(self):
        """Background thread for neural network training"""
        print("Training thread started")
        while not self.stopping:
            # Only train if we have enough samples
            if len(self.memory) > 64:
                # Use a smaller batch size for more frequent updates
                with self.training_lock:
                    self.replay(batch_size=32, verbose=0)

            # Sleep to reduce CPU usage
            time.sleep(0.1)
    
    def act(self, state):
        """Choose an action based on state"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        # Force CPU execution and no verbose output
        with tf.device('/CPU:0'):
            q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size, verbose=0):
        """Train the neural network on batch_size samples"""
        if len(self.memory) < batch_size:
            return
        
        # Get random batch from memory
        minibatch = self.memory.sample(batch_size)
        
        # Extract data from minibatch
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Batch predict to optimize performance
        with tf.device('/CPU:0'):
            # Predict Q-values for current states
            current_qs = self.model.predict(states, verbose=verbose)
            # Predict Q-values for next states
            future_qs = self.target_model.predict(next_states, verbose=verbose)
        
        # Update target Q values
        for i in range(batch_size):
            if dones[i]:
                current_qs[i][actions[i]] = rewards[i]
            else:
                current_qs[i][actions[i]] = rewards[i] + self.gamma * np.max(future_qs[i])
        
        # Train the model
        with tf.device('/CPU:0'):
            self.model.fit(states, current_qs, epochs=1, verbose=verbose, batch_size=batch_size)
        
        # Update epsilon for exploration-exploitation trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Periodically update target model
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_model()