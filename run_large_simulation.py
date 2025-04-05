#!/usr/bin/env python3
"""
Script for running large-scale cell survival simulations with GPU optimization.
"""

import os
import argparse
import subprocess
import time
import sys
import psutil
from datetime import datetime
from pathlib import Path

# Try to import torch, but handle the case when it's not installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Running in CPU-only mode.")
    print("To enable GPU acceleration, install PyTorch with: pip install torch")

# Only import GPUMonitor if torch is available
if TORCH_AVAILABLE:
    try:
        from src.utils.gpu_utils import GPUMonitor
    except ImportError:
        print("Warning: Could not import GPUMonitor. GPU monitoring will be disabled.")


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

def main():
    system_info = get_system_info()
    recommended = recommend_settings(system_info)
    
    # Print system information
    print("=== System Information ===")
    print(f"CPU Cores: {system_info['cpu_count']}")
    print(f"Memory: {system_info['memory_gb']:.2f} GB")
    if system_info["gpu_available"]:
        print(f"GPU: {system_info['gpu_name']} ({system_info['gpu_memory'][0]:.2f} GB)")
    else:
        print("No GPU detected. Running in CPU mode.")
        print("For GPU acceleration, install PyTorch with CUDA support.")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run large-scale cell survival simulation with GPU optimization')
    parser.add_argument('--episodes', type=int, default=recommended["episodes"], help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=recommended["max_steps"], help='Maximum steps per episode')
    parser.add_argument('--world-size', type=int, default=recommended["world_size"], help='Size of the world (square)')
    parser.add_argument('--num-cells', type=int, default=recommended["num_cells"], help='Initial number of cells')
    parser.add_argument('--num-foods', type=int, default=recommended["num_foods"], help='Number of food sources')
    parser.add_argument('--num-hazards', type=int, default=recommended["num_hazards"], help='Number of hazards')
    parser.add_argument('--batch-size', type=int, default=recommended["batch_size"], help='Batch size for neural networks')
    parser.add_argument('--no-render', action='store_true', help='Disable visualization')
    parser.add_argument('--analyze', action='store_true', help='Run analysis after simulation')
    parser.add_argument('--gpu-profile', action='store_true', help='Enable detailed GPU profiling')
    args = parser.parse_args()
    
    # Print recommended and actual settings
    print("\n=== Simulation Settings ===")
    print(f"World Size: {args.world_size} x {args.world_size}")
    print(f"Initial Cells: {args.num_cells}")
    print(f"Food Sources: {args.num_foods}")
    print(f"Hazards: {args.num_hazards}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Batch Size: {args.batch_size}")
    print(f"GPU Profiling: {'Enabled' if args.gpu_profile else 'Disabled'}")
    print(f"Visualization: {'Disabled' if args.no_render else 'Enabled'}")
    
    proceed = input("\nDo you want to proceed with these settings? (y/n): ")
    if proceed.lower() != 'y':
        print("Simulation cancelled")
        return
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Initialize GPU monitoring if available
    gpu_monitor = None
    if TORCH_AVAILABLE and system_info["gpu_available"]:
        try:
            gpu_monitor = GPUMonitor(log_interval=30, log_dir=os.path.join(output_dir, "logs"))
        except Exception as e:
            print(f"Error initializing GPU monitor: {e}")
    
    # Build command for simulation
    cmd = ['python3', 'src/main.py']
    
    # Add parameters
    cmd.extend(['--episodes', str(args.episodes)])
    cmd.extend(['--max-steps', str(args.max_steps)])
    cmd.extend(['--world-size', str(args.world_size)])
    cmd.extend(['--num-cells', str(args.num_cells)])
    cmd.extend(['--num-foods', str(args.num_foods)])
    cmd.extend(['--num-hazards', str(args.num_hazards)])
    cmd.extend(['--batch-size', str(args.batch_size)])
    
    if args.no_render:
        cmd.append('--no-render')
    
    if args.gpu_profile:
        cmd.append('--gpu-profile')
    
    # Save configuration
    with open(os.path.join(output_dir, 'simulation_config.txt'), 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        if system_info["gpu_available"]:
            f.write(f"\nGPU: {system_info['gpu_name']}\n")
            f.write(f"GPU Memory: {system_info['gpu_memory'][0]:.2f} GB\n")
    
    # Run simulation
    print("\n=== Starting Simulation ===")
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
            while process.poll() is None:
                print(".", end="", flush=True)
                time.sleep(10)
        
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
                
                # Find the latest metrics file
                metrics_file = os.path.join(output_dir, "logs", "simulation_metrics.csv")
                if os.path.exists(metrics_file):
                    analysis_cmd = ['python3', 'src/analyze_simulation.py', 
                                    '--metrics', metrics_file,
                                    '--output', os.path.join(output_dir, "analysis")]
                    subprocess.run(analysis_cmd)
                else:
                    print(f"Could not find metrics file at {metrics_file}")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        if process and process.poll() is None:
            process.terminate()
    
    except Exception as e:
        print(f"\nError running simulation: {e}")

if __name__ == "__main__":
    main()
