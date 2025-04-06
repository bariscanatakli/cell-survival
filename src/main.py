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
from utils.debug_utils import SimulationDebugger

def setup_gpu():
    """Setup GPU and return device"""
    if not TORCH_AVAILABLE:
        return None
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Set memory optimization parameters
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Better performance but less reproducible
        # Enable TF32 precision on Ampere GPUs for better performance
        torch.set_float32_matmul_precision('high')
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
        
        # Set memory limits to avoid OOM errors
        for i in range(torch.cuda.device_count()):
            # Reserve 90% of GPU memory to avoid fragmentation
            torch.cuda.set_per_process_memory_fraction(0.9, i)
            
        # More aggressive memory caching
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")
    
    return device

def run_simulation_step(world, agent_manager):
    """Run one step of the simulation for all cells"""
    states = {}
    actions = {}
    
    # Get all cell states in one batch for better performance
    all_cells = world.cells
    if not all_cells:
        return states, actions
    
    # Group cells by species type for efficient batch processing
    species_groups = {}
    for idx, cell in enumerate(all_cells):
        try:
            # Get cell state
            cell_state = world.get_state(cell)
            states[idx] = cell_state
            
            # Group by species type
            species_type = cell.species.type
            if species_type not in species_groups:
                species_groups[species_type] = {"indices": [], "states": []}
            species_groups[species_type]["indices"].append(idx)
            species_groups[species_type]["states"].append(cell_state)
        except Exception as e:
            print(f"Error processing cell {idx}: {str(e)}")
            # Use default state
            states[idx] = np.zeros(world.state_size)
    
    # Process each species group in a batch
    for species_type, group_data in species_groups.items():
        try:
            agent = agent_manager.get_agent_for_type(species_type)
            
            # Use batch processing if available
            if hasattr(agent, 'get_actions_batch'):
                batch_actions = agent.get_actions_batch(group_data["states"])
                # Assign actions back to their original indices
                for i, idx in enumerate(group_data["indices"]):
                    actions[idx] = batch_actions[i]
            else:
                # Fallback to individual processing
                for i, idx in enumerate(group_data["indices"]):
                    actions[idx] = agent.get_action(group_data["states"][i])
        except Exception as e:
            print(f"Error getting actions for {species_type}: {str(e)}")
            # Provide safe default actions (0 = no movement)
            for idx in group_data["indices"]:
                actions[idx] = 0

    return states, actions

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
        'step': [],
        'episode': [],
        'total_cells': [],
        'food_count': [],
        'hazard_count': [],
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
    
    # Create debugger
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debugger = SimulationDebugger(debug_dir)
    
    # Analyze agents
    for species_type, agent in agents.items():
        debugger.analyze_agent(agent, world.state_size)
    
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
                states, actions = run_simulation_step(world, agent_manager)
                
                # Step world simulation
                next_state, reward, done, info = world.step(actions)
                
                # Store experiences
                for i, cell in enumerate(world.cells):
                    if i in actions and not done.get(i, True):
                        try:
                            # Get agent for this cell's species
                            agent = agents[cell.species.type]
                            
                            # Ensure we have reward and done information
                            cell_reward = reward.get(i, 0) if isinstance(reward, dict) else 0
                            cell_done = done.get(i, False) if isinstance(done, dict) else False
                            
                            # Get state information consistently
                            if isinstance(state, dict) and i in state:
                                cell_state = state[i]
                                cell_next_state = next_state.get(i, cell_state) if isinstance(next_state, dict) else next_state
                            else:
                                # If no specific state for this cell, use world.get_state to get current state
                                cell_state = world.get_state(cell)
                                cell_next_state = world.get_state(cell)  # Use current state as next state if not available
                            
                            # Remember the experience
                            agent.remember(cell_state, actions[i], cell_reward, cell_next_state, cell_done)
                            episode_rewards[cell.species.type] += cell_reward
                            
                        except Exception as err:
                            print(f"Warning: Error storing experience for cell {i}: {err}")
                            print(f"Debug info - state type: {type(state)}, "
                                  f"actions type: {type(actions)}, "
                                  f"action keys: {list(actions.keys())[:5] if isinstance(actions, dict) else 'N/A'}")
                
                # Move to next state
                state = next_state
                
                # Update metrics
                if step % 10 == 0:
                    metrics['episode'].append(episode)
                    metrics['step'].append(step)
                    metrics['total_cells'].append(len(world.cells))
                    metrics['food_count'].append(len(world.foods))
                    metrics['hazard_count'].append(len(world.hazards))
                    
                    # Calculate per-species metrics with safer access
                    for species_type in SpeciesType:
                        # Using list comprehension with type checking to avoid errors
                        try:
                            species_cells = [cell for cell in world.cells if hasattr(cell, 'species') and hasattr(cell.species, 'type') and cell.species.type == species_type]
                            
                            metrics[f'{species_type.name.lower()}_count'].append(len(species_cells))
                            
                            # Safely calculate averages with explicit zero handling
                            if len(species_cells) > 0:
                                avg_energy = sum(getattr(cell, 'energy_level', 0) for cell in species_cells) / len(species_cells)
                                avg_age = sum(getattr(cell, 'age', 0) for cell in species_cells) / len(species_cells)
                            else:
                                avg_energy = 0
                                avg_age = 0
                                
                            metrics[f'{species_type.name.lower()}_avg_energy'].append(avg_energy)
                            metrics[f'{species_type.name.lower()}_avg_age'].append(avg_age)
                        except Exception as species_error:
                            # Log the error but continue processing
                            print(f"Error calculating metrics for {species_type.name}: {species_error}")
                            # Add zeros to maintain array lengths
                            metrics[f'{species_type.name.lower()}_count'].append(0)
                            metrics[f'{species_type.name.lower()}_avg_energy'].append(0)
                            metrics[f'{species_type.name.lower()}_avg_age'].append(0)
                    
                    # Evolution metrics
                    metrics['adaptation_score'].append(evolution_manager.calculate_adaptation_score(world.cells))
                    metrics['avg_mutation_rate'].append(evolution_manager.calculate_average_mutation_rate(world.cells))
                    metrics['competition_factor'].append(world.competition_factor)
                    metrics['season'].append(world.challenges.current_season.name)
                
                # Train agents less frequently for better performance
                if step % 10 == 0:  # Changed from 5 to 10 - training less often
                    for agent_type, agent in agents.items():
                        if hasattr(agent, 'replay'):
                            # Use larger batch sizes for better GPU utilization
                            batch_size = args.batch_size * 2 if len(world.cells) > 50 else args.batch_size
                            agent.replay(batch_size)
                
                # Run evolution process less frequently
                if step % 100 == 0 and len(world.cells) > 0:  # Changed from 50 to 100
                    # Only call if the method exists
                    if hasattr(evolution_manager, 'evolve_population'):
                        evolution_manager.evolve_population(world.cells, world.challenges)
                
                # Render less frequently for performance if enabled
                if visualizer and step % 10 == 0:  # Changed from 3 to 10
                    visualizer.render(world.cells, world.foods, world.hazards, episode+1, step, world)
                
                # Log world state periodically - reduced frequency
                if step % 250 == 0:  # Changed from 100 to 250
                    debugger.log_world_state(world)
                
                # Only check state dimensions occasionally to improve performance
                if step % 25 == 0:  # Only check every 25 steps instead of every step
                    # Define active_cells as the indices of the current cells
                    active_cells = list(range(len(world.cells)))

                    # Check state dimensions before using - use safer access method
                    for i in active_cells:
                        if i < len(active_cells) and i in state:
                            # Skip dimension check most of the time for better performance
                            if step % 100 == 0:  # Only run this expensive check every 100 steps
                                debugger.check_state_dimensions(state[i], world.state_size)
                        # If cell index doesn't exist in state, log it
                        elif i not in state and i < len(active_cells):
                            debugger.log_info(f"Warning: Cell {i} exists but has no state entry")
                
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
            # Before creating DataFrame, ensure all arrays have the same length
            min_length = min(len(value) for value in metrics.values() if isinstance(value, list) and len(value) > 0)
            
            # Log the lengths for debugging
            print(f"Normalizing metrics arrays to length {min_length}")
            for key in metrics:
                if isinstance(metrics[key], list):
                    if len(metrics[key]) > min_length:
                        print(f"Trimming {key} from {len(metrics[key])} to {min_length} entries")
                    metrics[key] = metrics[key][:min_length]
            
            # Then create DataFrame
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(f"{output_dir}/logs/simulation_metrics.csv", index=False)
        except Exception as metrics_error:
            print(f"Error saving metrics: {metrics_error}")
            # Try a more robust approach to save at least some data
            try:
                print("Attempting alternative method to save metrics...")
                # Create a new dict with only columns of equal length
                safe_metrics = {}
                for key, values in metrics.items():
                    if not isinstance(values, list):
                        continue
                    if not safe_metrics:
                        # First list becomes the reference length
                        ref_length = len(values)
                        safe_metrics[key] = values
                    elif len(values) == ref_length:
                        # Only add lists with matching length
                        safe_metrics[key] = values
                
                if safe_metrics:
                    pd.DataFrame(safe_metrics).to_csv(f"{output_dir}/logs/partial_metrics.csv", index=False)
                    print(f"Saved partial metrics with {len(safe_metrics)} columns")
            except Exception as e:
                print(f"Could not save any metrics: {e}")
        
        print(f"Simulation completed. Results saved to {output_dir}")
        
        # Stop profiler if enabled
        if TORCH_AVAILABLE and args.gpu_profile and torch.cuda.is_available():
            try:
                profiler.stop()
            except:
                pass
        
    except Exception as e:
        # Use a default value for step if it's not defined yet
        current_step = step if 'step' in locals() else 0
        current_episode = episode if 'episode' in locals() else 0
        debugger.log_exception(e, f"Episode {current_episode}, step {current_step}")
        import traceback
        error_msg = f"Error during simulation: {str(e)}"
        traceback_str = traceback.format_exc()
        
        # Print error with traceback
        print(error_msg)
        print("\nTraceback:")
        print(traceback_str)
        
        # Log the error to a file
        try:
            error_log_path = f"{output_dir}/logs/error_log.txt"
            with open(error_log_path, 'w') as f:
                f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error message: {str(e)}\n\n")
                f.write("Traceback:\n")
                f.write(traceback_str)
                f.write("\n\nSystem info:\n")
                if TORCH_AVAILABLE:
                    f.write(f"PyTorch version: {torch.__version__}\n")
                    if torch.cuda.is_available():
                        f.write(f"CUDA version: {torch.version.cuda}\n")
                        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"Python version: {sys.version}\n")
                
                # Add simulation state information
                f.write("\nSimulation state at crash:\n")
                f.write(f"Episode: {current_episode}, Step: {current_step}\n")
                f.write(f"Number of cells: {len(world.cells)}\n")
                f.write(f"Number of foods: {len(world.foods)}\n")
                f.write(f"Number of hazards: {len(world.hazards)}\n")
                
                # Add state shape information
                if 'state' in locals():
                    f.write(f"State type: {type(state)}\n")
                    if isinstance(state, dict):
                        f.write(f"State keys: {list(state.keys())[:10]}\n")
                    elif isinstance(state, (list, np.ndarray)):
                        f.write(f"State shape: {np.shape(state)}\n")
                
                # Add info about the actions that were being processed
                if 'actions' in locals():
                    f.write(f"Actions type: {type(actions)}\n")
                    if isinstance(actions, dict):
                        f.write(f"Action keys: {list(actions.keys())[:10]}\n")
            
            print(f"Error details have been saved to {error_log_path}")
        except Exception as log_error:
            print(f"Failed to write error log: {log_error}")
        
        # Clean up resources
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                print("CUDA memory cache cleared")
            except Exception as gpu_error:
                print(f"Failed to clear CUDA memory: {gpu_error}")
                
        # Try to save any collected metrics up to the error point
        try:
            if 'metrics' in locals() and len(metrics.get('step', [])) > 0:
                print("Attempting to save partial metrics before exit...")
                
                # Ensure all arrays have the same length before creating DataFrame
                min_length = min(len(value) for value in metrics.values() if isinstance(value, list))
                emergency_metrics = {key: value[:min_length] if isinstance(value, list) else value 
                                   for key, value in metrics.items()}
                
                emergency_df = pd.DataFrame(emergency_metrics)
                emergency_df.to_csv(f"{output_dir}/logs/partial_metrics_before_crash.csv", index=False)
                print(f"Partial metrics saved to {output_dir}/logs/partial_metrics_before_crash.csv")
        except Exception as metrics_error:
            print(f"Could not save partial metrics: {metrics_error}")
            
    finally:
        # Always clean up regardless of success or failure
        if 'progress_bar' in locals():
            progress_bar.close()
        
        # Close visualization if it was initialized
        if 'visualizer' in locals() and visualizer is not None:
            if hasattr(visualizer, 'close'):
                try:
                    visualizer.close()
                except Exception as vis_error:
                    print(f"Error closing visualizer: {vis_error}")
        
        # Final GPU memory cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Simulation exited.")

if __name__ == "__main__":
    main()