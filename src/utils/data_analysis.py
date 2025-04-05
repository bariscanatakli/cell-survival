import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class SimulationAnalyzer:
    """Utility class for analyzing and visualizing simulation results"""
    
    def __init__(self, metrics_path):
        """
        Initialize with a path to the metrics CSV file
        
        Args:
            metrics_path (str): Path to the CSV file with simulation metrics
        """
        self.metrics = pd.read_csv(metrics_path)
        self.output_dir = os.path.dirname(os.path.dirname(metrics_path))
        self.plots_dir = os.path.join(self.output_dir, "plots")
        
        # Create plots directory if it doesn't exist
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
    
    def plot_population_over_time(self):
        """Plot population of each species over time"""
        species_cols = [col for col in self.metrics.columns if col.endswith('_count') 
                        and not col == 'total_cells' 
                        and not col == 'food_count' 
                        and not col == 'hazard_count']
        
        plt.figure(figsize=(12, 6))
        for col in species_cols:
            species_name = col.split('_count')[0]
            plt.plot(self.metrics['step'], self.metrics[col], label=species_name)
        
        plt.title('Species Population Over Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Population')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(self.plots_dir, 'population_over_time.png'))
        plt.close()
    
    def plot_energy_levels(self):
        """Plot average energy levels of each species over time"""
        energy_cols = [col for col in self.metrics.columns if col.endswith('_avg_energy')]
        
        plt.figure(figsize=(12, 6))
        for col in energy_cols:
            species_name = col.split('_avg_energy')[0]
            plt.plot(self.metrics['step'], self.metrics[col], label=species_name)
        
        plt.title('Average Energy Levels Over Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Average Energy')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(self.plots_dir, 'energy_levels.png'))
        plt.close()
    
    def plot_cell_age(self):
        """Plot average age of each species over time"""
        age_cols = [col for col in self.metrics.columns if col.endswith('_avg_age')]
        
        plt.figure(figsize=(12, 6))
        for col in age_cols:
            species_name = col.split('_avg_age')[0]
            plt.plot(self.metrics['step'], self.metrics[col], label=species_name)
        
        plt.title('Average Cell Age Over Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Average Age')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(self.plots_dir, 'cell_age.png'))
        plt.close()
    
    def plot_resource_availability(self):
        """Plot food and hazard counts over time"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.metrics['step'], self.metrics['food_count'], 
                 label='Food', color='green')
        plt.plot(self.metrics['step'], self.metrics['hazard_count'], 
                 label='Hazards', color='red', linestyle='--')
        
        plt.title('Resource Availability Over Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(self.plots_dir, 'resource_availability.png'))
        plt.close()
    
    def generate_all_plots(self):
        """Generate all analysis plots"""
        self.plot_population_over_time()
        self.plot_energy_levels()
        self.plot_cell_age()
        self.plot_resource_availability()
        
        print(f"All plots saved to {self.plots_dir}")
    
    def print_summary_statistics(self, save_to_file=None):
        """
        Print summary statistics of the simulation and optionally save to file.
        
        Args:
            save_to_file (str, optional): Path to save the statistics summary
        """
        # Calculate basic statistics
        avg_population = self.metrics['total_cells'].mean()
        max_population = self.metrics['total_cells'].max()
        min_population = self.metrics['total_cells'].min()
        std_population = self.metrics['total_cells'].std()
        
        species_cols = [col for col in self.metrics.columns if col.endswith('_count') 
                       and not col == 'total_cells' 
                       and not col == 'food_count' 
                       and not col == 'hazard_count']
        
        # Get initial, final, and average counts for each species
        initial_counts = {}
        final_counts = {}
        avg_counts = {}
        max_counts = {}
        
        for col in species_cols:
            species_name = col.split('_count')[0]
            initial_counts[species_name] = self.metrics[col].iloc[0]
            final_counts[species_name] = self.metrics[col].iloc[-1]
            avg_counts[species_name] = self.metrics[col].mean()
            max_counts[species_name] = self.metrics[col].max()
        
        # Prepare the output
        separator = "=" * 50
        output = []
        
        # Header
        output.append(separator)
        output.append("SIMULATION SUMMARY")
        output.append(separator)
        
        # Basic stats
        output.append(f"Total Steps: {self.metrics['step'].max()}")
        output.append(f"Episodes: {self.metrics['episode'].max() + 1}")
        output.append(f"Average Population: {avg_population:.2f} ± {std_population:.2f}")
        output.append(f"Maximum Population: {max_population}")
        output.append(f"Minimum Population: {min_population}")
        
        # Species stats
        output.append("\nSpecies Counts:")
        output.append(f"{'Species':<10} {'Initial':<8} {'Final':<8} {'Avg':<8} {'Max':<8} {'Change':<10}")
        output.append("-" * 55)
        
        for species in sorted(initial_counts.keys()):
            initial = initial_counts[species]
            final = final_counts[species]
            avg = avg_counts[species]
            maximum = max_counts[species]
            
            if initial > 0:
                change_pct = ((final - initial) / initial) * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"
                
            output.append(f"{species:<10} {initial:<8} {final:<8} {avg:<8.2f} {maximum:<8} {change_str:<10}")
        
        # Population change analysis
        output.append("\nPopulation Change Analysis:")
        for species in sorted(initial_counts.keys()):
            initial = initial_counts[species]
            final = final_counts[species]
            
            if initial > 0:
                change_pct = ((final - initial) / initial) * 100
                status = "increased" if change_pct > 0 else "decreased" if change_pct < 0 else "unchanged"
                output.append(f"  {species}: {status} by {abs(change_pct):.1f}% ({initial} → {final})")
            elif final > 0:
                output.append(f"  {species}: started with 0, ended with {final}")
            else:
                output.append(f"  {species}: no population throughout simulation")
        
        # Print all lines
        for line in output:
            print(line)
        
        # Save to file if requested
        if save_to_file:
            with open(save_to_file, 'w') as f:
                for line in output:
                    f.write(line + '\n')
            print(f"\nSummary statistics saved to {save_to_file}")
            
    def generate_detailed_report(self, output_dir):
        """
        Generate a detailed report with statistics and insights.
        
        Args:
            output_dir (str): Directory to save the report
        """
        import os
        from datetime import datetime
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"simulation_report_{timestamp}.txt")
        
        # Generate stats
        self.print_summary_statistics(save_to_file=report_path)
        
        # Add extinction events if any
        with open(report_path, 'a') as f:
            f.write("\n\nExtinction Events:\n")
            
            species_cols = [col for col in self.metrics.columns if col.endswith('_count') 
                           and not col == 'total_cells' 
                           and not col == 'food_count' 
                           and not col == 'hazard_count']
            
            extinction_events = []
            
            for col in species_cols:
                species = col.split('_count')[0]
                # Find where the species went from >0 to 0
                extinct_steps = []
                
                for i in range(1, len(self.metrics)):
                    if self.metrics[col].iloc[i-1] > 0 and self.metrics[col].iloc[i] == 0:
                        extinct_steps.append((
                            self.metrics['episode'].iloc[i],
                            self.metrics['step'].iloc[i]
                        ))
                
                for episode, step in extinct_steps:
                    extinction_events.append(f"  {species} went extinct in episode {episode} at step {step}")
            
            if extinction_events:
                f.write("\n".join(extinction_events))
            else:
                f.write("  No extinction events detected")
            
            f.write("\n\nEnd of report.\n")
        
        print(f"Detailed report saved to {report_path}")
    
    def plot_species_survival(self, save_dir=None):
        """
        Plot the survival rates of each species over time.
        
        Args:
            save_dir (str, optional): Directory to save the plot
        """
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        
        plt.figure(figsize=(10, 6))
        
        # Find species columns
        species_cols = [col for col in self.metrics.columns if col.endswith('_count') 
                       and not col == 'total_cells' 
                       and not col == 'food_count' 
                       and not col == 'hazard_count']
        
        # Calculate survival rates for each species at each step
        for col in species_cols:
            species_name = col.split('_count')[0]
            initial_count = self.metrics[col].iloc[0]
            
            if initial_count > 0:  # Only plot if species started with non-zero population
                survival_rates = (self.metrics[col] / initial_count) * 100
                plt.plot(self.metrics['step'], survival_rates, label=f"{species_name}")
        
        plt.title('Species Survival Rates Over Time')
        plt.xlabel('Simulation Step')
        plt.ylabel('Survival Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"species_survival_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)
            print(f"Survival rates plot saved to {filepath}")
        else:
            plt.show()
