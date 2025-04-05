import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from environment.species import SpeciesType

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuralAgent:
    def __init__(self, species_type, state_size=24, action_size=8, hidden_size=128, learning_rate=0.001):
        self.species_type = species_type
        self.state_size = state_size
        self.action_size = action_size
        self.memory_capacity = 10000
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for {species_type.name} agent")
        
        # Create main network and target network
        self.model = NeuralNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_model = NeuralNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Counter for target model updates
        self.target_update_counter = 0
    
    def get_action(self, state, explore=True):
        if explore and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def select_actions(self, state, explore=True):
        """
        Returns a list of actions for the given state or batch of states.
        If state is a single sample, the returned list will contain one action.
        """
        # Check if state is batch-like (has more than one sample)
        if (hasattr(state, 'ndim') and state.ndim > 1) or isinstance(state, (list, tuple)) and len(state) > 1:
            actions = []
            for s in state:
                actions.append(self.get_action(s, explore))
            return actions
        else:
            # Single state: return list containing one action
            return [self.get_action(state, explore)]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Process batch data in parallel using GPU
        states = torch.FloatTensor([experience[0] for experience in batch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in batch]).to(self.device)
        next_states = torch.FloatTensor([experience[3] for experience in batch]).to(self.device)
        dones = torch.FloatTensor([experience[4] for experience in batch]).to(self.device)
        
        # Compute Q values
        current_q_values = self.model(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        # Compute target
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Update network
        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter >= 100:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
    
    def save(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

# Factory method for creating agents
def create_agent(species_type, **kwargs):
    return NeuralAgent(species_type, **kwargs)
