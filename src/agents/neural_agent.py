import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from environment.species import SpeciesType
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_batch_norm=True, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        
        # More efficient architecture with batch normalization and dropout
        self.use_batch_norm = use_batch_norm
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) if use_batch_norm else None
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size) if use_batch_norm else None
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Advantage and Value streams for dueling architecture
        self.advantage = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Check if we need to handle batches of size 1 for batch norm
        if self.use_batch_norm and x.size(0) == 1 and self.training:
            # Skip batch norm for single samples during training
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
        else:
            # Regular forward pass with batch norm if enabled
            h1 = self.fc1(x)
            if self.use_batch_norm and self.bn1 is not None:
                h1 = self.bn1(h1)
            h1 = F.relu(h1)
            h1 = self.dropout1(h1)
            
            h2 = self.fc2(h1)
            if self.use_batch_norm and self.bn2 is not None:
                h2 = self.bn2(h2)
            h2 = F.relu(h2)
            h2 = self.dropout2(h2)
        
        # Dueling network architecture: separate advantage and value streams
        advantage = self.advantage(h2)
        value = self.value(h2)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class NeuralAgent:
    def __init__(self, species_type, state_size=24, action_size=8, hidden_size=128, learning_rate=0.001):
        self.species_type = species_type
        self.state_size = state_size
        self.action_size = action_size
        self.memory_capacity = 20000  # Increased memory capacity for better learning
        self.memory = PrioritizedReplayBuffer(self.memory_capacity)  # Enhanced replay buffer with prioritized experience
        self.batch_size = 64
        self.gamma = 0.99
        
        # Improved epsilon decay settings for faster learning
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.992  # Slightly faster decay
        
        # Add this line to initialize update_count
        self.update_count = 0
        self.target_update_freq = 50  # More frequent target updates for faster convergence
        
        # Learning rate scheduler for adaptive learning
        self.initial_lr = learning_rate
        self.min_lr = learning_rate / 10
        
        # Use GPU if available with optimized settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for {species_type.name} agent")
        
        # Configure optimized CUDA settings if available
        if torch.cuda.is_available():
            # Set memory allocation strategies
            torch.cuda.set_per_process_memory_fraction(0.95)  # Allow using up to 95% of GPU memory
            torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmarking for faster conv operations
            
        # Create enhanced network with batch norm and dropout
        self.model = NeuralNetwork(
            state_size, 
            hidden_size, 
            action_size, 
            use_batch_norm=True, 
            dropout_rate=0.2
        ).to(self.device)
        self.target_model = NeuralNetwork(
            state_size, 
            hidden_size, 
            action_size,
            use_batch_norm=True, 
            dropout_rate=0.2
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimized Adam settings with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5  # L2 regularization for better generalization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=self.min_lr
        )
        
        # Loss function
        self.criterion = nn.HuberLoss()  # Huber loss is more robust than MSE
        
        # Track losses for learning rate scheduling
        self.avg_loss = 0
        self.loss_counter = 0
        
        # Counter for target model updates
        self.target_update_counter = 0
        self.update_target_model()
    
    def get_action(self, state, explore=True):
        # Add error handling for state processing
        try:
            if explore and np.random.rand() <= self.epsilon:
                return np.random.choice(self.action_size)
            
            # Process state into tensor appropriately based on its type
            if isinstance(state, dict):
                # Convert dict to flat list with consistent ordering
                sorted_keys = sorted(state.keys())
                processed_state = []
                for key in sorted_keys:
                    if hasattr(state[key], '__iter__') and not isinstance(state[key], (str, dict)):
                        processed_state.extend(state[key])
                    else:
                        processed_state.append(state[key])
                state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
            elif isinstance(state, np.ndarray):
                # Fast path for numpy arrays - most common case
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:
                # Handle regular state (list or array)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
        except Exception as e:
            print(f"Error getting action: {e}")
            print(f"State: {state}")
            # Return a random action as fallback
            return np.random.choice(self.action_size)

    def get_actions_batch(self, states, explore=True):
        """Optimized batch action selection with CUDA performance enhancements"""
        try:
            batch_size = len(states)
            actions = np.zeros(batch_size, dtype=np.int64)
            
            # Fast path: if not exploring, process all states at once
            if not explore or self.epsilon < 0.01:
                # Convert all states to tensors in one batch
                state_tensors = self.prepare_state_batch(states)
                
                # Set model to evaluation mode for inference
                self.model.eval()
                
                # Get actions in a single efficient forward pass
                with torch.no_grad():
                    q_values = self.model(state_tensors)
                    best_actions = torch.argmax(q_values, dim=1).cpu().numpy()
                
                # Convert to regular Python integers
                for i in range(batch_size):
                    action_value = best_actions[i]
                    actions[i] = int(action_value) if not isinstance(action_value, np.ndarray) else int(action_value.item() if action_value.size == 1 else action_value[0])
                
                return actions
            
            # Regular path with exploration
            explore_indices = np.random.rand(batch_size) <= self.epsilon
            
            # Generate exploration actions efficiently
            if np.any(explore_indices):
                explore_idx_list = np.where(explore_indices)[0]
                # Vectorized random action generation
                random_actions = np.random.randint(0, self.action_size, size=len(explore_idx_list))
                actions[explore_idx_list] = random_actions
            
            # If all states are for exploration, return early
            if np.all(explore_indices):
                return actions
            
            # Process exploitation states
            exploit_indices = ~explore_indices
            exploit_idx_list = np.where(exploit_indices)[0]
            
            if len(exploit_idx_list) == 0:
                return actions
                
            # Get corresponding states for exploitation
            exploit_states = [states[i] for i in exploit_idx_list]
            
            # Convert to tensors and get actions
            state_tensors = self.prepare_state_batch(exploit_states)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Single forward pass
            with torch.no_grad():
                q_values = self.model(state_tensors)
                best_actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            # Assign exploitation actions
            for i, idx in enumerate(exploit_idx_list):
                if i < len(best_actions):
                    action_value = best_actions[i]
                    actions[idx] = int(action_value) if not isinstance(action_value, np.ndarray) else int(action_value.item() if action_value.size == 1 else action_value[0])
            
            return actions
            
        except Exception as e:
            print(f"Error in optimized batch action selection: {e}")
            # Fallback to safer approach
            actions = np.zeros(len(states), dtype=np.int64)
            for i in range(len(states)):
                actions[i] = np.random.randint(0, self.action_size)
            return actions
    
    def prepare_state_batch(self, states):
        """Efficiently prepare a batch of states for the neural network"""
        # Handle different state formats
        if isinstance(states[0], dict):
            # Handle dictionary states
            try:
                processed_states = []
                for state in states:
                    # Fast path for common case
                    if hasattr(state, 'values') and isinstance(next(iter(state.values())), np.ndarray):
                        # Just use the first array if it's the right size
                        values = next(iter(state.values()))
                        if values.size == self.state_size:
                            processed_states.append(values.flatten())
                            continue
                    
                    # Fall back to more careful processing
                    sorted_keys = sorted(state.keys())
                    processed_state = []
                    for key in sorted_keys:
                        if hasattr(state[key], '__iter__') and not isinstance(state[key], (str, dict)):
                            processed_state.extend(state[key])
                        else:
                            processed_state.append(state[key])
                    processed_states.append(processed_state)
                return torch.FloatTensor(processed_states).to(self.device)
            except Exception as e:
                print(f"Error processing dict states: {e}")
                # Fall back to simpler approach
                return torch.FloatTensor([list(s.values())[0].flatten() for s in states]).to(self.device)
                
        elif isinstance(states[0], np.ndarray):
            # Fast path for numpy arrays - most common case
            try:
                # Use stack for same-shaped arrays, more efficient than manual looping
                return torch.from_numpy(np.stack([s.flatten() for s in states])).float().to(self.device)
            except Exception as e:
                print(f"Error stacking np arrays: {e}")
                # Fall back to tensor conversion one by one
                return torch.FloatTensor([s.flatten() for s in states]).to(self.device)
        else:
            # Handle any other format
            return torch.FloatTensor([s if not hasattr(s, 'flatten') else s.flatten() for s in states]).to(self.device)

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
        # Add error checking for better stability
        try:
            self.memory.add(state, action, reward, next_state, done)
        except Exception as e:
            print(f"Error adding experience to memory: {e}")
            print(f"State type: {type(state)}, Action: {action}, Reward: {reward}")
    
    def replay(self, batch_size):
        """Enhanced experience replay with prioritization for faster learning"""
        if len(self.memory) < batch_size:
            return
        
        try:
            # Sample batch using prioritization
            batch_data = self.memory.sample(batch_size)
            if not batch_data or len(batch_data) < 3:  # Make sure we got valid data
                return
                
            batch, indices, weights = batch_data
            
            # Process batch to handle different state formats
            processed_batch = []
            for i, experience in enumerate(batch):
                state, action, reward, next_state, done = experience
                
                # Process dictionary states into arrays
                if isinstance(state, dict):
                    # Convert state dict to numpy array
                    if isinstance(list(state.values())[0], np.ndarray):
                        state = list(state.values())[0]
                    else:
                        # Skip this experience if we can't convert it
                        continue
                
                if isinstance(next_state, dict):
                    # Convert next_state dict to numpy array
                    if isinstance(list(next_state.values())[0], np.ndarray):
                        next_state = list(next_state.values())[0]
                    else:
                        # Skip this experience if we can't convert it
                        continue
                
                # Ensure state has proper shape for flattening
                if not hasattr(state, 'shape') or not hasattr(next_state, 'shape'):
                    continue
                
                processed_batch.append((state, action, reward, next_state, done, i))  # Include original index
            
            # Skip further processing if no valid experiences remain
            if not processed_batch:
                return
            
            # Now process all the valid experiences
            try:
                states_np = np.array([experience[0].flatten() for experience in processed_batch])
                states = torch.FloatTensor(states_np).to(self.device)
                actions = torch.LongTensor([experience[1] for experience in processed_batch]).to(self.device)
                rewards = torch.FloatTensor([experience[2] for experience in processed_batch]).to(self.device)
                next_states_np = np.array([experience[3].flatten() for experience in processed_batch]) 
                next_states = torch.FloatTensor(next_states_np).to(self.device)
                dones = torch.BoolTensor([experience[4] for experience in processed_batch]).to(self.device)
                orig_indices = [experience[5] for experience in processed_batch]  # Original indices in the batch
                
                # Convert importance sampling weights to tensor
                weights_tensor = torch.FloatTensor(weights[orig_indices]).to(self.device)
            except Exception as e:
                print(f"Error processing batch data: {e}")
                return
            
            # Print GPU memory usage periodically
            if self.update_count % 100 == 0 and torch.cuda.is_available():
                print(f"GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB, {torch.cuda.memory_reserved()/1024**2:.2f}MB reserved")
            
            # Set model to training mode
            self.model.train()
            
            # Compute current Q-values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            
            # Compute target Q-values with Double DQN approach
            with torch.no_grad():
                # For Double DQN, we use the online network to select actions
                # and the target network to evaluate those actions
                best_actions = self.model(next_states).argmax(dim=1, keepdim=True)
                # Then evaluate those actions using the target network
                next_q_values = self.target_model(next_states).gather(1, best_actions)
                # Compute target Q values
                target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
            
            # Compute element-wise loss with Huber Loss for robustness
            elementwise_loss = self.criterion(current_q_values, target_q_values)
            
            # Apply importance sampling weights to loss
            weighted_loss = (elementwise_loss * weights_tensor.unsqueeze(1)).mean()
            
            # Optimize the model
            self.optimizer.zero_grad()
            weighted_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Store loss for learning rate scheduler
            self.avg_loss = 0.95 * self.avg_loss + 0.05 * weighted_loss.item() if self.avg_loss > 0 else weighted_loss.item()
            self.loss_counter += 1
            
            # Update learning rate periodically based on loss trend
            if self.loss_counter % 10 == 0:
                self.scheduler.step(self.avg_loss)
            
            # Calculate TD errors for priority updates
            with torch.no_grad():
                td_errors = torch.abs(target_q_values - current_q_values).cpu().numpy().flatten()
            
            # Update priorities in buffer based on TD errors
            self.memory.update_priorities([indices[i] for i in orig_indices], td_errors)
            
            # Update target model if needed with soft update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.soft_update_target_model()
            
            # Decay epsilon (slightly slower when loss is high)
            decay_factor = min(1.0, max(0.98, self.avg_loss / 2.0)) if self.avg_loss > 0 else 1.0
            self.epsilon = max(self.epsilon_min, self.epsilon * (self.epsilon_decay * decay_factor))
            
        except Exception as e:
            import traceback
            print(f"Error during replay: {str(e)}")
            print(traceback.format_exc())
    
    def update_target_model(self):
        """Update target model with weights from the main model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def soft_update_target_model(self, tau=0.01):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            tau (float): Interpolation parameter between 0 and 1
        """
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
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
