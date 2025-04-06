import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional


class EvolutionManager:
    """
    Manages the evolutionary process for cell populations in the reinforcement learning environment.
    Handles selection, reproduction, mutation, and tracking of evolution metrics.
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.7,
                 elitism_count: int = 5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.generation = 0
        self.population = []
        self.fitness_history = []
    
    def initialize_population(self, agent_factory):
        """Initialize a random population of agents"""
        self.population = [agent_factory() for _ in range(self.population_size)]
        self.generation = 0
    
    def select_parents(self, fitness_scores: List[float]) -> List[int]:
        """Select parents using tournament selection"""
        selected = []
        population_size = len(fitness_scores)
        
        # Ensure we have at least 1 individual
        if population_size == 0:
            return []
            
        # Make tournament size adaptive - minimum 2, maximum 3, never larger than population
        k = min(3, max(2, population_size - 1))
        
        for _ in range(population_size):
            # Tournament selection (select k random individuals and pick the best)
            tournament_indices = random.sample(range(population_size), k)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(winner_idx)
        return selected
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create offspring"""
        if random.random() < self.crossover_rate:
            # Implement your crossover logic here
            # This is a placeholder and should be adapted to your specific agent representation
            child = parent1.clone()
            # Mix parameters from both parents
            return child
        else:
            # No crossover, return clone of first parent
            return parent1.clone()
    
    def mutate(self, agent):
        """Apply mutation to an agent"""
        # Apply random mutations based on mutation rate
        # This is a placeholder and should be adapted to your specific agent representation
        if random.random() < self.mutation_rate:
            # Apply mutation
            pass
        return agent
    
    def calculate_fitness(self, cell, challenges):
        """
        Calculate fitness score for a cell based on various factors
        
        Args:
            cell: The cell to calculate fitness for
            challenges: The challenge manager instance
            
        Returns:
            float: Fitness score
        """
        # Base fitness calculation using energy level and age
        fitness = (cell.energy_level / cell.species.max_energy) * 0.4 + \
                  (cell.age / 100) * 0.3 + \
                  (cell.performance_score / 100) * 0.3
        
        # Apply environmental adaptation bonuses
        if hasattr(challenges, 'current_season') and challenges.current_season == 'WINTER':
            fitness *= (1 + getattr(cell, 'cold_resistance', 0) * 0.2)
        
        # Apply disaster-specific resistance bonuses if they exist
        if hasattr(challenges, 'get_active_disasters'):
            active_disasters = challenges.get_active_disasters()
            for disaster in active_disasters:
                if hasattr(disaster, 'name'):
                    if disaster.name == 'DROUGHT':
                        fitness *= (1 + getattr(cell, 'drought_resistance', 0) * 0.2)
                    elif disaster.name == 'DISEASE':
                        fitness *= (1 + getattr(cell, 'disease_resistance', 0) * 0.2)
                    elif disaster.name == 'RADIATION':
                        fitness *= (1 + getattr(cell, 'radiation_resistance', 0) * 0.2)
        
        return fitness
    
    def evolve_population(self, cells, challenges):
        """
        Evolve the population based on fitness
        
        Args:
            cells (list): List of all cells in the simulation
            challenges (ChallengeManager): The challenge manager instance
        """
        # Update the population with the current cells
        self.population = cells
        
        if not self.population:
            # No cells to evolve
            return
        
        # Calculate fitness for each cell
        fitness_scores = [self.calculate_fitness(cell, challenges) for cell in self.population]
        
        # Get indices sorted by fitness (higher is better)
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        
        # Safety check: ensure all indices are within the valid range
        sorted_indices = [i for i in sorted_indices if i < len(self.population)]
        
        if not sorted_indices:
            # No valid indices, nothing to evolve
            return
        
        # Get sorted population based on fitness
        sorted_population = [self.population[i] for i in sorted_indices]
        
        # Elitism: keep the best individuals
        new_population = sorted_population[:self.elitism_count]
        
        # Selection
        parent_indices = self.select_parents(fitness_scores)
        parents = [self.population[idx] for idx in parent_indices]
        
        # Crossover and mutation to fill the rest of the population
        while len(new_population) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            offspring = self.crossover(parent1, parent2)
            offspring = self.mutate(offspring)
            
            new_population.append(offspring)
        
        # Update population
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        return self.population
    
    def get_best_agent(self, fitness_scores: List[float]):
        """Return the best agent from the current population"""
        best_idx = np.argmax(fitness_scores)
        return self.population[best_idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        return {
            'generation': self.generation,
            'fitness_history': self.fitness_history,
            'population_size': self.population_size,
        }
    
    def calculate_adaptation_score(self, cells):
        """
        Calculate the overall adaptation score of the population
        
        Args:
            cells: List of all cells in the world
            
        Returns:
            float: Average adaptation score across all species
        """
        if not cells:
            return 0.0
            
        # Get species types from cells
        species_types = set(cell.species.type for cell in cells)
        
        # Use a simple metric based on population fitness
        species_scores = {}
        for cell in cells:
            species_type = cell.species.type
            if species_type not in species_scores:
                species_scores[species_type] = []
            
            # Use fitness score if available, otherwise use normalized energy level
            if hasattr(cell, 'fitness_score'):
                species_scores[species_type].append(cell.fitness_score)
            elif hasattr(cell, 'energy_level') and hasattr(cell.species, 'max_energy'):
                species_scores[species_type].append(cell.energy_level / cell.species.max_energy)
            else:
                species_scores[species_type].append(0.5)  # Default if no data available
        
        # Calculate average score per species
        total_score = 0
        count = 0
        for species_type, scores in species_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                total_score += avg_score
                count += 1
        
        return total_score / max(1, count)  # Avoid division by zero
    
    def calculate_average_mutation_rate(self, cells):
        """
        Calculate the average mutation rate across all cells
        
        Args:
            cells: List of all cells in the world
            
        Returns:
            float: Average mutation rate across all species
        """
        # If no cells or we don't track per-species mutation rates, return the base rate
        if not cells or not hasattr(self, 'mutation_rate'):
            return getattr(self, 'mutation_rate', 0.01)  # Default if not set
        
        return self.mutation_rate  # Return the base mutation rate
