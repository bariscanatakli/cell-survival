"""
Debugging utilities for the cell survival simulation
"""
import os
import torch
import numpy as np
import logging
import traceback
from datetime import datetime

# Set up logging
logger = logging.getLogger("SimDebugger")
logger.setLevel(logging.DEBUG)

class SimulationDebugger:
    """Tool to help debug simulation issues"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or self._create_debug_dir()
        
        # Set up file handler
        file_handler = logging.FileHandler(os.path.join(self.output_dir, 'debug_log.txt'))
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.info("Debug session started")
        
        # Track state dimensions
        self.expected_state_size = 24
        self.agent_input_shapes = {}
        
    def _create_debug_dir(self):
        """Create a directory for debug output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = f"debug_output_{timestamp}"
        os.makedirs(debug_dir, exist_ok=True)
        return debug_dir
        
    def log_world_state(self, world):
        """Log the current state of the world"""
        logger.info(f"World state - Cells: {len(world.cells)}, Foods: {len(world.foods)}, Hazards: {len(world.hazards)}")
        logger.info(f"Day cycle: {world.day_cycle}/{world.max_day_cycle}, Episode: {world.episode}")
        logger.info(f"Season: {world.challenges.current_season.name}, Active disasters: {[d.name for d in world.challenges.get_active_disasters()]}")
        
    def log_info(self, message):
        """Log an informational message"""
        logger.info(message)
        
    def check_state_dimensions(self, state, expected_size=None):
        """Check state dimensions and log any mismatches"""
        expected = expected_size or self.expected_state_size
        
        if isinstance(state, np.ndarray):
            if len(state.shape) == 2:  # (1, state_size)
                actual_size = state.shape[1]
            else:
                actual_size = state.shape[0]  # (state_size,)
                
            if actual_size != expected:
                logger.warning(f"State dimension mismatch: expected {expected}, got {actual_size}")
                return False
        elif isinstance(state, dict):
            logger.warning(f"State is a dictionary, not a numpy array: {state}")
            return False
        else:
            logger.warning(f"Unknown state type: {type(state)}")
            return False
            
        return True
    
    def analyze_agent(self, agent, state_size=None):
        """Analyze agent model structure and expected inputs"""
        if state_size:
            self.expected_state_size = state_size
            
        # Determine input layer size from the model
        if hasattr(agent, 'model'):
            if hasattr(agent.model, 'layers') and len(agent.model.layers) > 0:
                input_size = agent.model.layers[0].input_shape[1]
                logger.info(f"Agent model input size: {input_size}")
                self.agent_input_shapes[type(agent).__name__] = input_size
            elif hasattr(agent.model, '_modules'):
                # PyTorch models
                for name, layer in agent.model._modules.items():
                    if hasattr(layer, 'in_features'):
                        input_size = layer.in_features
                        logger.info(f"Agent model input size: {input_size}")
                        self.agent_input_shapes[type(agent).__name__] = input_size
                        break
        
    def log_exception(self, e, context=""):
        """Log an exception with context"""
        logger.error(f"Exception in {context}: {str(e)}")
        logger.error(traceback.format_exc())
