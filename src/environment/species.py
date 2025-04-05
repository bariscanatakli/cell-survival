from enum import Enum, auto
import numpy as np
import random

class SpeciesType(Enum):
    PREDATOR = auto()  # Red - apex predator, eats prey and gatherers
    PREY = auto()      # Yellow - mid-tier predator, eats gatherers
    GATHERER = auto()  # Green - herbivore, eats only plants
    SCAVENGER = auto() # Purple - bottom feeder, eats decaying matter

class Species:
    def __init__(self, species_type, **kwargs):
        self.type = species_type
        
        # Base attributes
        self.color = (255, 255, 255)  # Default white
        self.max_energy = kwargs.get('max_energy', 150)  # Increased base max_energy
        self.attack_power = 10
        self.speed = 2
        self.vision_range = 6
        self.metabolism_rate = 0.2
        self.reproduction_cost = 0.3
        self.mutation_rate = 0.05
        
        # New attributes for environmental adaptation
        self.energy_efficiency = kwargs.get('energy_efficiency', 0.9)  # Slightly reduced efficiency
        self.immune_strength = 0.0
        self.radiation_resistance = 0.0
        self.cold_resistance = 0.0
        self.detection_range = kwargs.get('detection_range', 8)  # Reduced from default value
        self.movement_speed = kwargs.get('movement_speed', 1.2)  # Increased to help find food
        
        # Species-specific attributes
        if species_type == SpeciesType.PREDATOR:
            self.color = (255, 50, 50)  # Red
            self.max_energy = 150
            self.attack_power = 15
            self.speed = 1
            self.vision_range = 8
            self.metabolism_rate = 0.3
            
        elif species_type == SpeciesType.PREY:
            self.color = (255, 255, 50)  # Yellow
            self.max_energy = 100
            self.attack_power = 10
            self.speed = 3
            self.vision_range = 10
            self.metabolism_rate = 0.2
            
        elif species_type == SpeciesType.GATHERER:
            self.color = (50, 255, 50)  # Green
            self.max_energy = 120
            self.attack_power = 5
            self.speed = 2
            self.vision_range = 5
            self.metabolism_rate = 0.15
            
        elif species_type == SpeciesType.SCAVENGER:
            self.color = (180, 50, 255)  # Purple
            self.max_energy = 80
            self.attack_power = 3
            self.speed = 3
            self.vision_range = 6
            self.metabolism_rate = 0.1
        
        # Add small random variation to attributes (for diversity)
        self._add_variation()
        
    def _add_variation(self, variation_range=0.1):
        """Add small random variations to attributes"""
        self.max_energy *= random.uniform(1-variation_range, 1+variation_range)
        self.attack_power *= random.uniform(1-variation_range, 1+variation_range)
        self.speed *= random.uniform(1-variation_range, 1+variation_range)
        self.vision_range *= random.uniform(1-variation_range, 1+variation_range)
        # Add variation to new attributes
        self.energy_efficiency *= random.uniform(1-variation_range, 1+variation_range)
        self.immune_strength = max(0, min(1, self.immune_strength + random.uniform(-0.05, 0.05)))
        self.radiation_resistance = max(0, min(1, self.radiation_resistance + random.uniform(-0.05, 0.05)))
        self.cold_resistance = max(0, min(1, self.cold_resistance + random.uniform(-0.05, 0.05)))
        
    @staticmethod
    def can_eat(predator_type, prey_type):
        """Define predator-prey relationships"""
        relationships = {
            SpeciesType.PREDATOR: [SpeciesType.PREY, SpeciesType.GATHERER],
            SpeciesType.PREY: [SpeciesType.GATHERER],  # Prey can eat gatherers
            SpeciesType.GATHERER: [],  # Gatherers only eat plants (handled separately)
            SpeciesType.SCAVENGER: []  # Scavengers eat decay (handled separately)
        }
        
        if predator_type in relationships and prey_type in relationships.get(predator_type, []):
            return True
        return False

    def mutate(self, mutation_strength=0.1):
        """Create a mutated copy of this species with slightly different attributes"""
        mutated = Species(self.type)
        
        # Copy existing attributes
        mutated.max_energy = self.max_energy
        mutated.attack_power = self.attack_power
        mutated.speed = self.speed
        mutated.vision_range = self.vision_range
        mutated.metabolism_rate = self.metabolism_rate
        mutated.energy_efficiency = self.energy_efficiency
        mutated.immune_strength = self.immune_strength
        mutated.radiation_resistance = self.radiation_resistance
        mutated.cold_resistance = self.cold_resistance
        
        # Apply mutations with increased diversity
        if random.random() < self.mutation_rate * 2:  # Double chance for mutation to speed up evolution
            # Apply mutations to various attributes
            mutated.max_energy *= random.uniform(1-mutation_strength, 1+mutation_strength*1.5)
            mutated.attack_power *= random.uniform(1-mutation_strength, 1+mutation_strength*1.5)
            mutated.speed *= random.uniform(1-mutation_strength, 1+mutation_strength*1.2)
            mutated.vision_range *= random.uniform(1-mutation_strength, 1+mutation_strength*1.2)
            mutated.metabolism_rate *= random.uniform(1-mutation_strength*0.8, 1+mutation_strength)
            
            # Specialized adaptations
            mutated.energy_efficiency *= random.uniform(1-mutation_strength, 1+mutation_strength*1.5)
            mutated.immune_strength += random.uniform(-mutation_strength, mutation_strength*1.5)
            mutated.radiation_resistance += random.uniform(-mutation_strength, mutation_strength*1.5)
            mutated.cold_resistance += random.uniform(-mutation_strength, mutation_strength*1.5)
            
            # Keep values in valid ranges
            mutated.immune_strength = max(0, min(1, mutated.immune_strength))
            mutated.radiation_resistance = max(0, min(1, mutated.radiation_resistance))
            mutated.cold_resistance = max(0, min(1, mutated.cold_resistance))
            
            # Enforce minimum values
            mutated.speed = max(0.5, mutated.speed)
            mutated.max_energy = max(30, mutated.max_energy)
            mutated.attack_power = max(1, mutated.attack_power)
            
        return mutated