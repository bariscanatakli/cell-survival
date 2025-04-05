from enum import Enum, auto
import numpy as np

class SpeciesType(Enum):
    PREDATOR = auto()  # Red - apex predator, eats hunters and gatherers
    PREY = auto()      # New species type
    GATHERER = auto()  # Green - herbivore, eats only plants
    SCAVENGER = auto() # Purple - bottom feeder, eats decaying matter

class Species:
    def __init__(self, species_type):
        self.type = species_type
        
        # Define species-specific attributes
        if species_type == SpeciesType.PREDATOR:
            self.color = (255, 50, 50)  # Red
            self.max_energy = 150
            self.attack_power = 15
            self.speed = 1
            self.vision_range = 8
            
        elif species_type == SpeciesType.PREY:
            self.color = (255, 255, 50)  # Yellow
            self.max_energy = 100
            self.attack_power = 2
            self.speed = 3
            self.vision_range = 10
            
        elif species_type == SpeciesType.GATHERER:
            self.color = (50, 255, 50)  # Green
            self.max_energy = 100
            self.attack_power = 5
            self.speed = 1
            self.vision_range = 5
            
        elif species_type == SpeciesType.SCAVENGER:
            self.color = (180, 50, 255)  # Purple
            self.max_energy = 80
            self.attack_power = 3
            self.speed = 3
            self.vision_range = 4
        
    @staticmethod
    def can_eat(predator_type, prey_type):
        """Define predator-prey relationships"""
        relationships = {
            SpeciesType.PREDATOR: [SpeciesType.PREY],
            SpeciesType.PREY: [],  # Prey doesn't eat other cells
            SpeciesType.GATHERER: [],  # Gatherers only eat plants (handled separately)
            SpeciesType.SCAVENGER: []  # Scavengers eat decay (handled separately)
        }
        
        if predator_type in relationships and prey_type in relationships.get(predator_type, []):
            return True
        return False