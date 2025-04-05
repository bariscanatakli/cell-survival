#!/usr/bin/env python3
"""
Comprehensive simulation analysis tool for cell-survival-RL.
This script analyzes simulation metrics and produces detailed reports and visualizations.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_analysis import SimulationAnalyzer


def analyze_simulation(metrics_file, output_dir=None, show_plots=True):
    """
    Analyze simulation metrics and generate comprehensive reports.
    
    Args:
        metrics_file (str): Path to the simulation metrics CSV file
        output_dir (str): Directory to save output files
        show_plots (bool): Whether to display plots interactively
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
    analyzer = SimulationAnalyzer(metrics)
    
    # Print basic stats
    print("\nBasic Statistics:")
    analyzer.print_summary_statistics()
    
    # Generate detailed report
    if output_dir:
        print("\nGenerating detailed report...")
        analyzer.generate_detailed_report(output_dir)
    
    # Create and save plots
    if output_dir or show_plots:
        print("\nGenerating visualizations...")
        
        # Population over time
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['step'], metrics['total_cells'])
        plt.title('Total Population Over Time')
        plt.xlabel('Step')
        plt.ylabel('Total Population')
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            population_plot = os.path.join(output_dir, "population_over_time.png")
            plt.savefig(population_plot)
            print(f"Population plot saved to {population_plot}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Species populations
        species_cols = [col for col in metrics.columns if col.endswith('_count') 
                       and col not in ['total_cells', 'food_count', 'hazard_count']]
        
        plt.figure(figsize=(10, 6))
        for col in species_cols:
            species_name = col.split('_count')[0]
            plt.plot(metrics['step'], metrics[col], label=species_name)
        
        plt.title('Species Populations Over Time')
        plt.xlabel('Step')
        plt.ylabel('Population')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_dir:
            species_plot = os.path.join(output_dir, "species_populations.png")
            plt.savefig(species_plot)
            print(f"Species plot saved to {species_plot}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Survival rates plot
        analyzer.plot_species_survival(save_dir=output_dir if output_dir else None)
    
    # Add this before the "Analysis complete!" message
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
        evolution_plot = os.path.join(output_dir, "evolution_metrics.png")
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
            disasters_plot = os.path.join(output_dir, "environmental_challenges.png")
            plt.savefig(disasters_plot)
            print(f"Environmental challenges plot saved to {disasters_plot}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Analyze simulation results for cell-survival-RL')
    parser.add_argument('metrics_file', help='Path to the simulation metrics CSV file')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files')
    parser.add_argument('--no-plots', action='store_true', help='Do not display plots interactively')
    
    args = parser.parse_args()
    
    # If no output directory is specified, create one based on the metrics filename
    if not args.output_dir:
        base_name = os.path.splitext(os.path.basename(args.metrics_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"analysis_{base_name}_{timestamp}"
    
    analyze_simulation(args.metrics_file, args.output_dir, not args.no_plots)


if __name__ == "__main__":
    main()
