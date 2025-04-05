import random
import numpy as np
from environment.species import Species, SpeciesType

class EvolutionManager:
    def __init__(self):
        self.generation = 0
        self.mutation_rate = 0.1  # Base mutation rate
        self.adaptation_score = {species_type: 0 for species_type in SpeciesType}
        
        # Track environmental adaptations for each species
        self.adaptations = {
            SpeciesType.PREDATOR: {
                'cold_resistance': 0,
                'drought_resistance': 0,
                'disease_resistance': 0,
                'radiation_resistance': 0
            },
            SpeciesType.PREY: {
                'cold_resistance': 0,
                'drought_resistance': 0,
                'disease_resistance': 0,
                'radiation_resistance': 0
            },
            SpeciesType.GATHERER: {
                'cold_resistance': 0,
                'drought_resistance': 0,
                'disease_resistance': 0,
                'radiation_resistance': 0
            },
            SpeciesType.SCAVENGER: {
                'cold_resistance': 0,
                'drought_resistance': 0,
                'disease_resistance': 0,
                'radiation_resistance': 0
            }
        }
    
    def evolve_population(self, cells, environment_challenges):
        """Evolve the population based on environmental challenges"""
        self.generation += 1
        
        # Group cells by species
        species_groups = {}
        for cell in cells:
            if cell.species.type not in species_groups:
                species_groups[cell.species.type] = []
            species_groups[cell.species.type].append(cell)
        
        # Calculate fitness and select cells for reproduction
        for species_type, group in species_groups.items():
            if not group:
                continue
            
            # Calculate fitness based on energy, age, and survival in current conditions
            for cell in group:
                cell.fitness_score = (cell.energy_level / cell.species.max_energy) * 0.4 + \
                                    (cell.age / 100) * 0.3 + \
                                    (cell.performance_score / 100) * 0.3
                
                # Apply environmental adaptation bonuses
                if environment_challenges.current_season == 'WINTER':
                    cell.fitness_score *= (1 + self.adaptations[species_type]['cold_resistance'] * 0.1)
                
                for disaster in environment_challenges.get_active_disasters():
                    if disaster.name == 'DROUGHT':
                        cell.fitness_score *= (1 + self.adaptations[species_type]['drought_resistance'] * 0.1)
                    elif disaster.name == 'DISEASE':
                        cell.fitness_score *= (1 + self.adaptations[species_type]['disease_resistance'] * 0.1)
                    elif disaster.name == 'RADIATION':
                        cell.fitness_score *= (1 + self.adaptations[species_type]['radiation_resistance'] * 0.1)
            
            # Sort by fitness
            group.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Top 20% reproduce with mutations
            survivors = group[:int(len(group) * 0.2) + 1]
            
            # Track adaptation scores
            if survivors:
                adaptation_change = sum(c.fitness_score for c in survivors) / len(survivors)
                self.adaptation_score[species_type] = (
                    self.adaptation_score[species_type] * 0.9 + adaptation_change * 0.1
                )
                
                # Occasionally adapt specifically to current challenges
                if random.random() < 0.1:
                    if environment_challenges.current_season == 'WINTER':
                        self.adaptations[species_type]['cold_resistance'] += 0.05
                    
                    for disaster in environment_challenges.get_active_disasters():
                        if disaster.name == 'DROUGHT':
                            self.adaptations[species_type]['drought_resistance'] += 0.05
                        elif disaster.name == 'DISEASE':
                            self.adaptations[species_type]['disease_resistance'] += 0.05
                        elif disaster.name == 'RADIATION':
                            self.adaptations[species_type]['radiation_resistance'] += 0.05
        
        return self.adaptations
    
    def get_mutation_rate(self, species_type):
        """Calculate mutation rate based on environmental pressure"""
        # Higher adaptation score = lower mutation rate (already well-adapted)
        # Low adaptation score = higher mutation rate (need to adapt quickly)
        base_rate = self.mutation_rate
        adaptation_factor = self.adaptation_score.get(species_type, 0)
        
        # Inverse relationship: lower adaptation means higher mutation
        if adaptation_factor > 0:
            return base_rate * (1.5 - min(adaptation_factor, 1))
        
        return base_rate * 1.5  # Higher mutation when no adaptation data

    def apply_species_adaptations(self, species, adaptation_profile):
        """Apply accumulated adaptations to a species"""
        species_type = species.type
        adaptation_level = self.adaptations[species_type]
        
        # Apply cold resistance
        if adaptation_level['cold_resistance'] > 0:
            species.metabolism_rate *= (1 - adaptation_level['cold_resistance'] * 0.1)
        
        # Apply drought resistance
        if adaptation_level['drought_resistance'] > 0:
            species.energy_efficiency = 1 + adaptation_level['drought_resistance'] * 0.2
        else:
            species.energy_efficiency = 1.0
        
        # Apply disease resistance
        if adaptation_level['disease_resistance'] > 0:
            species.immune_strength = adaptation_level['disease_resistance']
        else:
            species.immune_strength = 0
        
        # Apply radiation resistance
        if adaptation_level['radiation_resistance'] > 0:
            species.radiation_resistance = adaptation_level['radiation_resistance']
        else:
            species.radiation_resistance = 0
            
        return species
