import random
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Any, Union

class SumTree:
    """
    A sum tree data structure for efficient prioritized experience replay.
    Allows O(log n) sampling based on priorities.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
    
    def add(self, priority, data):
        # Find the leaf index
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Update data
        self.data[self.data_pointer] = data
        
        # Update tree with new priority
        self.update(tree_idx, priority)
        
        # Move pointer and update buffer size
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_idx, priority):
        # Change = new priority - old priority
        change = priority - self.tree[tree_idx]
        
        # Update the leaf
        self.tree[tree_idx] = priority
        
        # Propagate changes through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get(self, value):
        """
        Get the leaf index, priority value, and experience for a given sample value
        """
        parent_idx = 0
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            # Leaf node reached
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # Otherwise, traverse down the tree
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]
                parent_idx = right_idx
        
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for more efficient learning.
    Samples important transitions with higher probability.
    """
    def __init__(self, max_size, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta    # Importance sampling correction (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment  # Annealing rate for beta
        self.epsilon = epsilon  # Small constant to ensure all priorities > 0
        self.max_priority = 1.0  # Initial max priority for new experiences
        
    def add(self, state, action, reward, next_state, done, error=None):
        """Add an experience to the buffer with its priority"""
        experience = (state, action, reward, next_state, done)
        
        # If no error is provided (e.g., first insertion), use max priority
        priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        
        # Add to sum tree
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """Sample a batch based on priorities"""
        batch = []
        indices = []
        priorities = []
        
        # Calculate segment size
        segment = self.tree.total_priority() / batch_size
        
        # Increase beta for importance sampling correction
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample from each segment
        for i in range(batch_size):
            # Ensure some randomness within the segment
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            
            # Get sample from tree
            idx, priority, experience = self.tree.get(value)
            
            if experience is not None:  # Ensure we got valid data
                batch.append(experience)
                indices.append(idx)
                priorities.append(priority)
        
        # Calculate importance sampling weights
        N = self.tree.size
        if N == 0:
            return []
            
        # Normalize weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        weights = (N * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize to stabilize updates
        
        return batch, indices, weights
    
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.size

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience tuple to the buffer
        
        Parameters:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)