#!/usr/bin/env python3
"""
Unified launcher for the Cell Survival RL simulation.
This script combines functionality from run_simulation.py, run_large_simulation.py, and launch.py
to provide a comprehensive interface for running simulations.
"""

import os
import sys
import argparse
import subprocess
import time
import psutil
from datetime import datetime
from pathlib import Path

# Try to import torch, but handle the case when it's not installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# GPU monitoring tools are only imported if torch is available
if TORCH_AVAILABLE:
    try:
        # Check if GPUMonitor is available in utils
        from src.utils.gpu_utils import GPUMonitor
        GPU_MONITOR_AVAILABLE = True
    except ImportError:
        GPU_MONITOR_AVAILABLE = False


def get_system_info():
    """Get system information for proper resource allocation"""
    info = {
        "cpu_count": os.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
        "gpu_available": False,
        "gpu_count": 0,
    }
    
    if TORCH_AVAILABLE:
        info["gpu_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count() if info["gpu_available"] else 0
        
        if info["gpu_available"]:
            # Get GPU memory for each device
            info["gpu_memory"] = []
            for i in range(info["gpu_count"]):
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                info["gpu_memory"].append(gpu_mem)
                info["gpu_name"] = torch.cuda.get_device_name(i)
    
    return info


def recommend_settings(system_info):
    """Recommend simulation settings based on system capabilities"""
    settings = {}
    
    # Base recommendations
    if system_info["gpu_available"]:
        # Scale with GPU memory
        gpu_mem = system_info["gpu_memory"][0]  # Using first GPU
        
        if gpu_mem < 4:  # Under 4GB
            settings["world_size"] = 2048
            settings["num_cells"] = 200
            settings["num_foods"] = 800
            settings["batch_size"] = 128
        elif gpu_mem < 8:  # 4-8GB
            settings["world_size"] = 4096
            settings["num_cells"] = 500
            settings["num_foods"] = 2000
            settings["batch_size"] = 256
        elif gpu_mem < 16:  # 8-16GB
            settings["world_size"] = 8192
            settings["num_cells"] = 1000
            settings["num_foods"] = 5000
            settings["batch_size"] = 512
        else:  # 16GB+
            settings["world_size"] = 16384
            settings["num_cells"] = 2000
            settings["num_foods"] = 10000
            settings["batch_size"] = 1024
        
        # Number of hazards is proportional to world size
        settings["num_hazards"] = max(30, int(settings["world_size"] / 100))
        
        # Recommend episodes and steps based on available memory
        settings["episodes"] = 5
        settings["max_steps"] = 10000
        
    else:  # CPU recommendations
        if system_info["memory_gb"] < 8:
            settings["world_size"] = 1024
            settings["num_cells"] = 50
            settings["num_foods"] = 200
            settings["batch_size"] = 32
        else:
            settings["world_size"] = 2048
            settings["num_cells"] = 100
            settings["num_foods"] = 400
            settings["batch_size"] = 64
            
        settings["num_hazards"] = 30
        settings["episodes"] = 3
        settings["max_steps"] = 5000
    
    return settings


def run_simulation(args):
    """Run the simulation with the specified parameters"""
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Initialize GPU monitoring if available
    gpu_monitor = None
    if args.gpu_monitoring and TORCH_AVAILABLE and GPU_MONITOR_AVAILABLE:
        try:
            system_info = get_system_info()
            if system_info["gpu_available"]:
                gpu_monitor = GPUMonitor(log_interval=30, log_dir=os.path.join(output_dir, "logs"))
                print("GPU monitoring enabled")
        except Exception as e:
            print(f"Error initializing GPU monitor: {e}")
    
    # Build command for simulation
    cmd = ['python3', 'src/main.py']
    
    # Add parameters
    cmd.extend(['--episodes', str(args.episodes)])
    
    if hasattr(args, 'max_steps') and args.max_steps:
        cmd.extend(['--max-steps', str(args.max_steps)])
    
    cmd.extend(['--world-size', str(args.world_size)])
    
    if hasattr(args, 'num_cells') and args.num_cells:
        cmd.extend(['--num-cells', str(args.num_cells)])
    
    if hasattr(args, 'num_foods') and args.num_foods:
        cmd.extend(['--num-foods', str(args.num_foods)])
    
    if hasattr(args, 'num_hazards') and args.num_hazards:
        cmd.extend(['--num-hazards', str(args.num_hazards)])
    
    if hasattr(args, 'batch_size') and args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    
    if args.no_render:
        cmd.append('--no-render')
    
    if hasattr(args, 'fullscreen') and args.fullscreen:
        cmd.append('--fullscreen')
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
        for arg_name, arg_value in vars(args).items():
            f.write(f"{arg_name}: {arg_value}\n")
        
        if TORCH_AVAILABLE:
            system_info = get_system_info()
            if system_info["gpu_available"]:
                f.write(f"\nGPU: {system_info['gpu_name']}\n")
                f.write(f"GPU Memory: {system_info['gpu_memory'][0]:.2f} GB\n")
    
    # Run simulation
    print("\n=== Starting Simulation ===")
    print(f"Output directory: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(cmd)
        
        # Monitor GPU usage during simulation
        if gpu_monitor:
            while process.poll() is None:
                gpu_monitor.update()
                time.sleep(5)
        else:
            # Simple progress indication if no GPU monitor
            process.wait()
        
        if process.returncode != 0:
            print(f"\nSimulation ended with an error (code {process.returncode}).")
        else:
            duration = time.time() - start_time
            print(f"\nSimulation completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
            
            # Generate GPU usage report
            if gpu_monitor:
                gpu_monitor.plot_usage()
                gpu_monitor.summary()
            
            # Run analysis if requested
            if args.analyze:
                print("\nRunning analysis...")
                
                # Find the metrics file
                metrics_file = os.path.join(output_dir, "logs", "simulation_metrics.csv")
                if not os.path.exists(metrics_file):
                    metrics_file = os.path.join(output_dir, "logs", "partial_metrics.csv")
                
                if os.path.exists(metrics_file):
                    analysis_cmd = ['python3', 'src/analyze.py', 
                                    '--metrics', metrics_file,
                                    '--output-dir', os.path.join(output_dir, "analysis")]
                    subprocess.run(analysis_cmd)
                else:
                    print(f"Could not find metrics file in {os.path.join(output_dir, 'logs')}")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        if process and process.poll() is None:
            process.terminate()
    
    except Exception as e:
        print(f"\nError running simulation: {e}")


def run_interactive_menu():
    """Display an interactive menu for users"""
    system_info = get_system_info()
    recommended = recommend_settings(system_info)
    
    print("Cell Survival RL Simulation")
    print("==========================")
    print("1. Run basic simulation")
    print("2. Run large-scale simulation")
    print("3. Run simulation with analysis")
    print("4. Run simulation in fullscreen mode")
    print("5. Analyze existing results")
    print("6. Clean up old output directories")
    print("7. Exit")
    
    choice = input("\nEnter your choice (1-7): ")
    
    if choice == '1':
        episodes = input(f"Enter number of episodes (default: {recommended['episodes']}): ")
        episodes = int(episodes) if episodes.strip() and episodes.isdigit() else recommended['episodes']
        
        world_size = input(f"Enter world size (default: 1024): ")
        world_size = int(world_size) if world_size.strip() and world_size.isdigit() else 1024
        
        args = argparse.Namespace(
            episodes=episodes,
            world_size=world_size,
            no_render=False,
            analyze=False,
            fullscreen=False,
            gpu_monitoring=False
        )
        run_simulation(args)
    
    elif choice == '2':
        print("\n=== System Information ===")
        print(f"CPU Cores: {system_info['cpu_count']}")
        print(f"Memory: {system_info['memory_gb']:.2f} GB")
        if system_info.get("gpu_available", False):
            print(f"GPU: {system_info.get('gpu_name', 'Unknown')} ({system_info.get('gpu_memory', [0])[0]:.2f} GB)")
        else:
            print("No GPU detected. Running in CPU mode.")
        
        print("\n=== Recommended Settings ===")
        print(f"World Size: {recommended['world_size']} x {recommended['world_size']}")
        print(f"Initial Cells: {recommended['num_cells']}")
        print(f"Food Sources: {recommended['num_foods']}")
        print(f"Hazards: {recommended['num_hazards']}")
        print(f"Episodes: {recommended['episodes']}")
        print(f"Max Steps: {recommended['max_steps']}")
        print(f"Batch Size: {recommended['batch_size']}")
        
        use_recommended = input("\nUse recommended settings? (y/n, default: y): ")
        if use_recommended.lower() != 'n':
            args = argparse.Namespace(
                episodes=recommended['episodes'],
                max_steps=recommended['max_steps'],
                world_size=recommended['world_size'],
                num_cells=recommended['num_cells'],
                num_foods=recommended['num_foods'],
                num_hazards=recommended['num_hazards'],
                batch_size=recommended['batch_size'],
                no_render=False,
                analyze=False,
                gpu_monitoring=True
            )
        else:
            episodes = input(f"Enter number of episodes (default: {recommended['episodes']}): ")
            episodes = int(episodes) if episodes.strip() and episodes.isdigit() else recommended['episodes']
            
            max_steps = input(f"Enter max steps per episode (default: {recommended['max_steps']}): ")
            max_steps = int(max_steps) if max_steps.strip() and max_steps.isdigit() else recommended['max_steps']
            
            world_size = input(f"Enter world size (default: {recommended['world_size']}): ")
            world_size = int(world_size) if world_size.strip() and world_size.isdigit() else recommended['world_size']
            
            num_cells = input(f"Enter initial number of cells (default: {recommended['num_cells']}): ")
            num_cells = int(num_cells) if num_cells.strip() and num_cells.isdigit() else recommended['num_cells']
            
            num_foods = input(f"Enter number of food sources (default: {recommended['num_foods']}): ")
            num_foods = int(num_foods) if num_foods.strip() and num_foods.isdigit() else recommended['num_foods']
            
            num_hazards = input(f"Enter number of hazards (default: {recommended['num_hazards']}): ")
            num_hazards = int(num_hazards) if num_hazards.strip() and num_hazards.isdigit() else recommended['num_hazards']
            
            batch_size = input(f"Enter batch size (default: {recommended['batch_size']}): ")
            batch_size = int(batch_size) if batch_size.strip() and batch_size.isdigit() else recommended['batch_size']
            
            no_render = input("Disable visualization? (y/n, default: n): ")
            no_render = no_render.lower() == 'y'
            
            args = argparse.Namespace(
                episodes=episodes,
                max_steps=max_steps,
                world_size=world_size,
                num_cells=num_cells,
                num_foods=num_foods,
                num_hazards=num_hazards,
                batch_size=batch_size,
                no_render=no_render,
                analyze=False,
                gpu_monitoring=True
            )
        
        analyze = input("Run analysis after simulation? (y/n, default: y): ")
        if analyze.lower() != 'n':
            args.analyze = True
        
        run_simulation(args)
    
    elif choice == '3':
        episodes = input(f"Enter number of episodes (default: {recommended['episodes']}): ")
        episodes = int(episodes) if episodes.strip() and episodes.isdigit() else recommended['episodes']
        
        world_size = input(f"Enter world size (default: 1024): ")
        world_size = int(world_size) if world_size.strip() and world_size.isdigit() else 1024
        
        args = argparse.Namespace(
            episodes=episodes,
            world_size=world_size,
            no_render=False,
            analyze=True,
            fullscreen=False,
            gpu_monitoring=False
        )
        run_simulation(args)
    
    elif choice == '4':
        episodes = input(f"Enter number of episodes (default: {recommended['episodes']}): ")
        episodes = int(episodes) if episodes.strip() and episodes.isdigit() else recommended['episodes']
        
        args = argparse.Namespace(
            episodes=episodes,
            world_size=1024,
            no_render=False,
            analyze=False,
            fullscreen=True,
            gpu_monitoring=False
        )
        run_simulation(args)
    
    elif choice == '5':
        # Find the latest output directory
        output_dirs = sorted([d for d in os.listdir('.') if d.startswith('output_')], reverse=True)
        
        if not output_dirs:
            print("No output directories found.")
            return
        
        print("\nAvailable output directories:")
        for i, dir_name in enumerate(output_dirs[:10], 1):  # Show only the 10 most recent
            print(f"{i}. {dir_name}")
        
        dir_choice = input(f"\nSelect directory (1-{min(len(output_dirs), 10)}, or enter full name): ")
        
        selected_dir = None
        if dir_choice.isdigit() and 1 <= int(dir_choice) <= min(len(output_dirs), 10):
            selected_dir = output_dirs[int(dir_choice) - 1]
        elif dir_choice in output_dirs:
            selected_dir = dir_choice
        
        if selected_dir:
            # Look for metrics files
            metrics_file = os.path.join(selected_dir, "logs", "simulation_metrics.csv")
            if not os.path.exists(metrics_file):
                metrics_file = os.path.join(selected_dir, "logs", "partial_metrics.csv")
            
            if os.path.exists(metrics_file):
                subprocess.run(['python3', 'src/analyze.py', '--metrics', metrics_file])
            else:
                print(f"No metrics files found in {selected_dir}/logs")
        else:
            print("Invalid selection.")
    
    elif choice == '6':
        # Find all output directories
        output_dirs = sorted([d for d in os.listdir('.') if d.startswith('output_')])
        
        if not output_dirs:
            print("No output directories found.")
            return
        
        # Group by date
        from collections import defaultdict
        date_groups = defaultdict(list)
        
        for dir_name in output_dirs:
            # Extract date part (first 8 characters after "output_")
            date_part = dir_name[7:15]  # YYYYMMDD
            date_groups[date_part].append(dir_name)
        
        print("\nOutput directories by date:")
        dates = sorted(date_groups.keys(), reverse=True)
        
        for i, date in enumerate(dates, 1):
            count = len(date_groups[date])
            year, month, day = date[:4], date[4:6], date[6:8]
            print(f"{i}. {year}-{month}-{day}: {count} directories")
        
        print("\nOptions:")
        print("1. Keep only the latest N directories")
        print("2. Remove directories from a specific date")
        print("3. Back to main menu")
        
        clean_choice = input("\nEnter your choice (1-3): ")
        
        if clean_choice == '1':
            keep_count = input("How many recent directories to keep? ")
            if keep_count.isdigit() and int(keep_count) > 0:
                keep_count = int(keep_count)
                dirs_to_remove = output_dirs[:-keep_count] if keep_count < len(output_dirs) else []
                
                if dirs_to_remove:
                    print(f"\nThe following {len(dirs_to_remove)} directories will be removed:")
                    for i, dir_name in enumerate(dirs_to_remove, 1):
                        if i <= 10:  # Show only the first 10
                            print(f"  {dir_name}")
                        elif i == 11:
                            print(f"  ... and {len(dirs_to_remove) - 10} more")
                    
                    confirm = input("\nConfirm deletion? (y/n): ")
                    if confirm.lower() == 'y':
                        import shutil
                        for dir_name in dirs_to_remove:
                            shutil.rmtree(dir_name)
                        print(f"Removed {len(dirs_to_remove)} directories.")
                else:
                    print(f"Only {len(output_dirs)} directories exist, none will be removed.")
        
        elif clean_choice == '2':
            date_choice = input(f"Select date to remove (1-{len(dates)}): ")
            if date_choice.isdigit() and 1 <= int(date_choice) <= len(dates):
                date_to_remove = dates[int(date_choice) - 1]
                dirs_to_remove = date_groups[date_to_remove]
                
                print(f"\nThe following {len(dirs_to_remove)} directories will be removed:")
                for i, dir_name in enumerate(dirs_to_remove, 1):
                    if i <= 10:  # Show only the first 10
                        print(f"  {dir_name}")
                    elif i == 11:
                        print(f"  ... and {len(dirs_to_remove) - 10} more")
                
                confirm = input("\nConfirm deletion? (y/n): ")
                if confirm.lower() == 'y':
                    import shutil
                    for dir_name in dirs_to_remove:
                        shutil.rmtree(dir_name)
                    print(f"Removed {len(dirs_to_remove)} directories.")
    
    elif choice == '7':
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("Invalid choice. Please try again.")
    
    # Return to the menu
    input("\nPress Enter to continue...")
    run_interactive_menu()


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Cell Survival RL Simulation Launcher')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive menu mode')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, help='Maximum steps per episode')
    parser.add_argument('--world-size', type=int, default=1024, help='Size of the world (square)')
    parser.add_argument('--num-cells', type=int, help='Initial number of cells')
    parser.add_argument('--num-foods', type=int, help='Number of food sources')
    parser.add_argument('--num-hazards', type=int, help='Number of hazards')
    parser.add_argument('--batch-size', type=int, help='Batch size for neural networks')
    parser.add_argument('--no-render', action='store_true', help='Disable visualization')
    parser.add_argument('--analyze', action='store_true', help='Run analysis after simulation')
    parser.add_argument('--fullscreen', action='store_true', help='Run in fullscreen mode')
    parser.add_argument('--gpu-monitoring', action='store_true', help='Enable GPU monitoring')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage completely')
    
    args = parser.parse_args()
    
    # Disable GPU if requested
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("GPU usage disabled")
    
    # Run in interactive mode if requested or if no arguments provided
    if args.interactive or (len(sys.argv) == 1):
        run_interactive_menu()
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()