import os
import time
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys

# Try to import torch, but handle the case when it's not installed
try:
    import torch
    import pandas as pd
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Running in CPU-only mode.")
    print("To enable GPU acceleration, install PyTorch with: pip install torch")

from environment.world import World
from environment.species import SpeciesType
from agents.neural_agent import create_agent
from visualization.pygame_render import GameVisualizer
from evolution.evolution_manager import EvolutionManager
from agent_manager import AgentManager  # Import the AgentManager

def setup_gpu():
    """Setup GPU and return device"""
    if not TORCH_AVAILABLE:
        return None
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Set memory optimization parameters
        torch.backends.cudnn.benchmark = True
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
        
        # Set memory limits to avoid OOM errors
        for i in range(torch.cuda.device_count()):
            # Reserve 90% of GPU memory to avoid fragmentation
            torch.cuda.set_per_process_memory_fraction(0.9, i)
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")
    
    return device

def run_simulation_step(world, agent_manager):
    # Get all cell types present in the world
    cell_types = set(cell.species.type for cell in world.cells)  # Changed from cell.cell_type to cell.species.type
    
    actions = {}  # Dictionary to store actions for each cell
    
    for cell_type in cell_types:
        # Get indices of cells of this type - handle empty list case
        indices = [i for i, cell in enumerate(world.cells) if cell.species.type == cell_type]
        if not indices:
            continue  # Skip if no cells of this type
        
        # Get the appropriate agent for this cell type
        agent = agent_manager.get_agent_for_type(cell_type, world.device)
        
        # Get observations for these cells
        observations = world.get_observations(indices)
        
        # If we have observations, get actions from the agent
        if len(observations) > 0:
            cell_actions = agent.select_actions(observations)
            
            # Apply actions to cells
            for idx, action in zip(indices, cell_actions):
                if idx < len(world.cells):  # Safety check
                    actions[idx] = action
    
    return actions  # Return the actions dictionary

def main():
    parser = argparse.ArgumentParser(description='Cell Survival Simulation')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=5000, help='Maximum steps per episode')
    parser.add_argument('--world-size', type=int, default=1024, help='Size of the world (square)')
    parser.add_argument('--num-cells', type=int, default=30, help='Initial number of cells')
    parser.add_argument('--num-foods', type=int, default=150, help='Number of food sources')
    parser.add_argument('--num-hazards', type=int, default=30, help='Number of hazards')
    parser.add_argument('--no-render', action='store_true', help='Disable visualization')
    parser.add_argument('--fullscreen', action='store_true', help='Run in fullscreen mode')
    parser.add_argument('--load-models', action='store_true', help='Load previously trained models')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gpu-profile', action='store_true', help='Enable GPU profiling')
    args = parser.parse_args()
    
    # Setup GPU
    device = setup_gpu()
    
    # Enable GPU profiling if requested
    if TORCH_AVAILABLE and args.gpu_profile and torch.cuda.is_available():
        try:
            from torch.cuda import profiler
            profiler.start()
        except ImportError:
            print("Warning: CUDA profiler not available. GPU profiling disabled.")
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    # Save configuration
    with open(f"{output_dir}/config.txt", 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Create world
    world = World(
        width=args.world_size, 
        height=args.world_size,
        num_cells=args.num_cells,
        num_foods=args.num_foods,
        num_hazards=args.num_hazards,
        device=device
    )
    
    # Initialize agents for each species type
    agents = {
        species_type: create_agent(species_type, state_size=24, action_size=8, hidden_size=256)
        for species_type in SpeciesType
    }
    
    # Create the agent manager
    agent_manager = AgentManager(agents)
    
    # Load pre-trained models if requested
    if args.load_models:
        for species_type, agent in agents.items():
            agent.load(f"models/{species_type.name}.weights")
    
    # Setup metrics tracking
    num_episodes = args.episodes
    metrics = {
        'episode': [],
        'step': [],
        'total_cells': [],
        'food_count': [],
        'hazard_count': []
    }
    
    # Add species-specific metrics
    for species_type in SpeciesType:
        metrics[f'{species_type.name.lower()}_count'] = []
        metrics[f'{species_type.name.lower()}_avg_energy'] = []
        metrics[f'{species_type.name.lower()}_avg_age'] = []
    
    # Create evolution manager
    evolution_manager = EvolutionManager()
    
    # Additional metrics for evolution and challenges
    metrics['disaster_count'] = []
    metrics['adaptation_score'] = []
    metrics['avg_mutation_rate'] = []
    metrics['competition_factor'] = []
    metrics['season'] = []
    
    # If we're doing a large simulation with many agents, increase batch size
    if args.num_cells > 100:
        for agent in agents.values():
            agent.batch_size = max(128, args.batch_size)
    
    # Main simulation loop with progress bar
    total_steps = num_episodes * args.max_steps
    progress_bar = tqdm(total=total_steps, desc="Simulation Progress")
    
    try:
        # Initialize visualizer if rendering is enabled
        visualizer = None if args.no_render else GameVisualizer(world.width, world.height)
        
        # Toggle fullscreen if requested
        if visualizer and args.fullscreen:
            visualizer.toggle_fullscreen()
        
        # Main simulation loop
        for episode in range(num_episodes):
            state = world.reset()
            done = {i: False for i in range(len(world.cells))}
            step = 0
            
            # Episode metrics
            episode_rewards = {species_type: 0 for species_type in SpeciesType}
            species_counts = {species_type: 0 for species_type in SpeciesType}
            
            # Apply evolutionary adaptations at the start of each episode
            for cell in world.cells:
                # If the cell has an apply_adaptations method:
                if hasattr(cell, 'apply_adaptations'):
                    cell.apply_adaptations()
                species_counts[cell.species.type] = species_counts.get(cell.species.type, 0) + 1
            
            # Run environment steps in batches for GPU efficiency
            while not all(done.values()) and step < args.max_steps:
                # Get actions from agents
                actions = run_simulation_step(world, agent_manager)
                
                # Step world simulation
                next_state, reward, done, info = world.step(actions)
                
                # Store experiences
                for i, cell in enumerate(world.cells):
                    if i in actions and not done.get(i, True):
                        agents[cell.species.type].remember(state[i], actions[i], reward[i], next_state[i], done[i])
                        episode_rewards[cell.species.type] += reward[i]
                
                # Move to next state
                state = next_state
                
                # Update metrics
                if step % 10 == 0:
                    metrics['episode'].append(episode)
                    metrics['step'].append(step)
                    metrics['total_cells'].append(len(world.cells))
                    metrics['food_count'].append(len(world.foods))
                    metrics['hazard_count'].append(len(world.hazards))
                    
                    # Calculate per-species metrics
                    for species_type in SpeciesType:
                        species_cells = [cell for cell in world.cells if cell.species.type == species_type]
                        metrics[f'{species_type.name.lower()}_count'].append(len(species_cells))
                        
                        avg_energy = 0 if len(species_cells) == 0 else sum(cell.energy_level for cell in species_cells) / len(species_cells)
                        metrics[f'{species_type.name.lower()}_avg_energy'].append(avg_energy)
                        
                        avg_age = 0 if len(species_cells) == 0 else sum(cell.age for cell in species_cells) / len(species_cells)
                        metrics[f'{species_type.name.lower()}_avg_age'].append(avg_age)
                    
                    # Evolution metrics
                    metrics['adaptation_score'].append(evolution_manager.calculate_adaptation_score(world.cells))
                    metrics['avg_mutation_rate'].append(evolution_manager.calculate_average_mutation_rate(world.cells))
                    metrics['competition_factor'].append(world.competition_factor)
                    metrics['season'].append(world.challenges.current_season.name)
                
                # Train agents every 5 steps
                if step % 5 == 0:
                    for agent_type, agent in agents.items():
                        if hasattr(agent, 'replay'):
                            agent.replay(args.batch_size)
                
                # Run evolution process every 50 steps
                if step % 50 == 0 and len(world.cells) > 0:
                    # Only call if the method exists
                    if hasattr(evolution_manager, 'evolve_population'):
                        evolution_manager.evolve_population(world.cells, world.challenges)
                
                # Render less frequently for performance if enabled
                if visualizer and step % 3 == 0:
                    visualizer.render(world.cells, world.foods, world.hazards, episode+1, step, world)
                
                step += 1
                progress_bar.update(1)
                
            # Save agent models periodically
            if (episode + 1) % 5 == 0:
                for species_type, agent in agents.items():
                    agent.save(f"{output_dir}/models/{species_type.name}_ep{episode+1}.weights")
            
            # Clear memory between episodes to save GPU memory
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Save metrics
        try:
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(f"{output_dir}/logs/simulation_metrics.csv", index=False)
        except NameError:
            # In case pandas is not available
            print(f"Warning: Could not save metrics. pandas is required.")
        
        print(f"Simulation completed. Results saved to {output_dir}")
        
        # Stop profiler if enabled
        if TORCH_AVAILABLE and args.gpu_profile and torch.cuda.is_available():
            try:
                profiler.stop()
            except:
                pass
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    finally:
        progress_bar.close()
        # Clean up PyTorch CUDA memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()