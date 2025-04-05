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
        for _ in range(self.population_size):
            # Tournament selection (select k random individuals and pick the best)
            k = 3  # Tournament size
            tournament_indices = random.sample(range(len(fitness_scores)), k)
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
    
    def evolve_population(self, fitness_scores: List[float], agent_factory):
        """Evolve the population based on fitness scores"""
        # Keep track of best fitness
        self.fitness_history.append({
            'max': max(fitness_scores),
            'avg': sum(fitness_scores) / len(fitness_scores),
            'min': min(fitness_scores)
        })
        
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
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
