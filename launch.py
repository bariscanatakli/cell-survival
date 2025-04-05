#!/usr/bin/env python3

import os
import sys
import subprocess

# Disable GPU usage completely
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    print("Cell Survival RL Simulation")
    print("==========================")
    print("1. Run simulation")
    print("2. Run simulation with analysis")
    print("3. Run simulation in fullscreen")
    print("4. Analyze existing results")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        episodes = input("Enter number of episodes (default: 10): ")
        if episodes.strip() and episodes.isdigit():
            subprocess.run(["python3", "src/main.py", "--episodes", episodes])
        else:
            subprocess.run(["python3", "src/main.py"])
    
    elif choice == '2':
        episodes = input("Enter number of episodes (default: 10): ")
        if episodes.strip() and episodes.isdigit():
            # First run the simulation
            subprocess.run(["python3", "src/main.py", "--episodes", episodes])
            # Then run analysis
            subprocess.run(["python3", "src/analyze_results.py"])
        else:
            subprocess.run(["python3", "src/main.py"])
            subprocess.run(["python3", "src/analyze_results.py"])
    
    elif choice == '3':
        episodes = input("Enter number of episodes (default: 10): ")
        if episodes.strip() and episodes.isdigit():
            subprocess.run(["python3", "src/main.py", "--episodes", episodes, "--fullscreen"])
        else:
            subprocess.run(["python3", "src/main.py", "--fullscreen"])
    
    elif choice == '4':
        subprocess.run(["python3", "src/analyze_results.py"])
    
    elif choice == '5':
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()