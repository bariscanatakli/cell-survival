#!/usr/bin/env python3

import os
import argparse
import glob
from utils.data_analysis import SimulationAnalyzer

def find_latest_output_dir():
    """Find the most recent output directory"""
    output_dirs = glob.glob("output_*")
    if not output_dirs:
        return None
    return max(output_dirs)

def main():
    parser = argparse.ArgumentParser(description='Analyze cell survival simulation results')
    parser.add_argument('--metrics', type=str, help='Path to metrics CSV file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    
    args = parser.parse_args()
    
    # If no metrics file specified, try to find the latest one
    if not args.metrics:
        output_dir = args.output_dir or find_latest_output_dir()
        if not output_dir:
            print("Error: No output directory found. Please specify --metrics or --output_dir")
            return
        
        metrics_file = os.path.join(output_dir, "logs", "simulation_metrics.csv")
        if not os.path.exists(metrics_file):
            print(f"Error: No metrics file found at {metrics_file}")
            return
    else:
        metrics_file = args.metrics
    
    print(f"Analyzing metrics from: {metrics_file}")
    
    # Create analyzer and generate all plots
    analyzer = SimulationAnalyzer(metrics_file)
    analyzer.print_summary_statistics()
    analyzer.generate_all_plots()

if __name__ == "__main__":
    main()