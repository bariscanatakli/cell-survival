#!/usr/bin/env python3

import os
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description='Run cell survival simulation')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--analyze', action='store_true', help='Run analysis after simulation')
    parser.add_argument('--render', action='store_true', default=True, help='Enable visualization')
    parser.add_argument('--world-size', type=int, default=1024, help='World size (width and height)')
    args = parser.parse_args()
    
    # Run simulation
    cmd = ['python3', 'src/main.py']
    
    if args.episodes != 10:
        cmd.extend(['--episodes', str(args.episodes)])
    
    if not args.render:
        cmd.append('--no-render')
    
    if args.world_size != 1024:
        cmd.extend(['--world-size', str(args.world_size)])
    
    print(f"Starting simulation with {args.episodes} episodes...")
    start_time = time.time()
    
    # Execute the simulation
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Simulation ended with an error.")
        return
    
    duration = time.time() - start_time
    print(f"Simulation completed in {duration:.2f} seconds")
    
    # Run analysis if requested
    if args.analyze:
        print("\nRunning analysis...")
        subprocess.run(['python3', 'src/analyze_results.py'])

if __name__ == "__main__":
    main()