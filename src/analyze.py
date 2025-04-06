#!/usr/bin/env python3
"""
Unified simulation analysis tool for cell-survival-RL.
This script combines functionality from analyze_results.py and analyze_simulation.py
to provide a comprehensive analysis solution.
"""

import os
import sys
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_analysis import SimulationAnalyzer


def find_latest_output_dir():
    """Find the most recent output directory"""
    output_dirs = glob.glob("output_*")
    if not output_dirs:
        return None
    return max(output_dirs)


def find_metrics_file(output_dir=None, metrics_file=None):
    """
    Find the metrics file to analyze, with the following priority:
    1. Explicitly provided metrics file
    2. simulation_metrics.csv in the specified output directory
    3. partial_metrics.csv in the specified output directory
    4. Any of the above in the latest output directory
    
    Args:
        output_dir (str): Path to output directory
        metrics_file (str): Path to metrics file
        
    Returns:
        str: Path to metrics file, or None if not found
    """
    if metrics_file and os.path.exists(metrics_file):
        return metrics_file
        
    # Find the output directory if not specified
    if not output_dir:
        output_dir = find_latest_output_dir()
        if not output_dir:
            return None
    
    # Check for standard metrics files
    for filename in ["simulation_metrics.csv", "partial_metrics.csv"]:
        file_path = os.path.join(output_dir, "logs", filename)
        if os.path.exists(file_path):
            return file_path
    
    return None


def analyze_simulation(metrics_file, output_dir=None, show_plots=True, basic_only=False):
    """
    Analyze simulation metrics and generate comprehensive reports.
    
    Args:
        metrics_file (str): Path to the simulation metrics CSV file
        output_dir (str): Directory to save output files
        show_plots (bool): Whether to display plots interactively
        basic_only (bool): If True, only generate basic statistics without plots
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load metrics
    try:
        metrics = pd.read_csv(metrics_file)
        print(f"Loaded metrics from {metrics_file}")
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return
    
    # Create analyzer
    analyzer = SimulationAnalyzer(metrics_file)
    
    # Print basic stats
    print("\nBasic Statistics:")
    analyzer.print_summary_statistics(
        save_to_file=os.path.join(output_dir, "summary_stats.txt") if output_dir else None
    )
    
    if basic_only:
        return
    
    # Generate detailed report
    if output_dir:
        print("\nGenerating detailed report...")
        analyzer.generate_detailed_report(output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    if output_dir:
        analyzer.generate_all_plots()
        print(f"All plots saved to {os.path.join(output_dir, 'plots')}")
    
    # Create survival rates plot
    analyzer.plot_species_survival(save_dir=output_dir if output_dir else None)
    
    # Additional evolutionary metrics if available
    if 'adaptation_score' in metrics.columns or 'disaster_count' in metrics.columns:
        plot_evolution_metrics(metrics, output_dir, show_plots)
    
    print("\nAnalysis complete!")


def plot_evolution_metrics(metrics_df, output_dir=None, show_plots=True):
    """Plot evolution-related metrics"""
    plt.figure(figsize=(12, 6))
    
    # Plot adaptation score
    if 'adaptation_score' in metrics_df.columns:
        plt.plot(metrics_df['step'], metrics_df['adaptation_score'], label='Adaptation Score')
    
    # Plot mutation rate
    if 'avg_mutation_rate' in metrics_df.columns:
        plt.plot(metrics_df['step'], metrics_df['avg_mutation_rate'], label='Avg Mutation Rate')
    
    # Plot competition factor
    if 'competition_factor' in metrics_df.columns:
        plt.plot(metrics_df['step'], metrics_df['competition_factor'], label='Competition Factor')
    
    plt.title('Evolution Metrics Over Time')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        evolution_plot = os.path.join(plots_dir, "evolution_metrics.png")
        plt.savefig(evolution_plot)
        print(f"Evolution metrics plot saved to {evolution_plot}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Plot disaster occurrences if available
    if 'disaster_count' in metrics_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['step'], metrics_df['disaster_count'], 'r-', label='Active Disasters')
        plt.title('Environmental Challenges Over Time')
        plt.xlabel('Step')
        plt.ylabel('Number of Active Disasters')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_dir:
            plots_dir = os.path.join(output_dir, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            disasters_plot = os.path.join(plots_dir, "environmental_challenges.png")
            plt.savefig(disasters_plot)
            print(f"Environmental challenges plot saved to {disasters_plot}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Analyze simulation results for cell-survival-RL')
    parser.add_argument('--metrics', '-m', help='Path to the simulation metrics CSV file')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files')
    parser.add_argument('--no-plots', action='store_true', help='Do not display plots interactively')
    parser.add_argument('--basic', action='store_true', help='Only generate basic statistics')
    
    args = parser.parse_args()
    
    # Find the metrics file to analyze
    metrics_file = find_metrics_file(args.output_dir, args.metrics)
    if not metrics_file:
        print("Error: No metrics file found. Please specify --metrics or --output-dir")
        return
    
    # Set default output directory if not specified
    if not args.output_dir:
        base_dir = os.path.dirname(os.path.dirname(metrics_file))
        if os.path.basename(os.path.dirname(metrics_file)) == "logs":
            # If metrics file is in a logs subdirectory, use the parent directory
            args.output_dir = os.path.join(base_dir, "analysis")
        else:
            # Otherwise, create a new directory based on the metrics filename
            base_name = os.path.splitext(os.path.basename(metrics_file))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"analysis_{base_name}_{timestamp}"
    
    analyze_simulation(metrics_file, args.output_dir, not args.no_plots, args.basic)


if __name__ == "__main__":
    main()