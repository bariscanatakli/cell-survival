import numpy as np
from .species import Species, SpeciesType

class Cell:
    def __init__(self, position, species_type, energy_level=None):
        self.position = position  # Position as a tuple (x, y)
        self.species = Species(species_type)
        
        # Initialize energy based on species if not specified
        self.energy_level = energy_level if energy_level is not None else self.species.max_energy
        self.age = 0
        self.performance_score = 0
        self.last_action = None
        self.food_consumed = 0
        self.reproduction_penalty = 0.0  # For environmental effects
        self.fitness_score = 0.0  # For evolution
        
        # Resistances to environmental challenges
        self.cold_resistance = 0.0
        self.drought_resistance = 0.0
        self.disease_resistance = 0.0
        self.radiation_resistance = 0.0

        self.stagnation_time = 0  # Track how long cell has been in same position
        self.last_movement_direction = None  # Track previous movement direction
        self.repetitive_movement_count = 0  # Track repetitive movements
        self.energy_depletion_rate = 0.5  # Increased from 0.2 to make survival harder
        self.last_food_time = 0  # Track time since last food consumption
        self.starvation_threshold = 50  # Steps before starvation effects begin

    def move(self, direction, distance=1):
        """Move the cell in a given direction by its species speed"""
        distance = self.species.speed  # Use species-specific speed
        
        # Check if this is a repetitive movement pattern
        if direction == self.last_movement_direction:
            self.repetitive_movement_count += 1
        else:
            self.repetitive_movement_count = 0
            self.last_movement_direction = direction
        
        # Apply movement
        if direction == 'up':
            self.position = (self.position[0], self.position[1] - distance)
        elif direction == 'down':
            self.position = (self.position[0], self.position[1] + distance)
        elif direction == 'left':
            self.position = (self.position[0] - distance, self.position[1])
        elif direction == 'right':
            self.position = (self.position[0] + distance, self.position[1])
        
        # Basic energy cost for movement
        base_energy_cost = 0.5 + (distance * 0.2)  # Increased base cost
        
        # Add penalty for stagnation (staying still) by increasing metabolism
        stagnation_penalty = 0.0
        if self.stagnation_time > 30:  # Severe stagnation
            stagnation_penalty = 0.5  # Increased from 0.3
        elif self.stagnation_time > 15:  # Moderate stagnation
            stagnation_penalty = 0.3  # Increased from 0.1
            
        # Add small penalty for very repetitive movement patterns
        repetition_penalty = min(0.2, self.repetitive_movement_count * 0.01)
        
        # Calculate total energy cost with efficiency factored in
        energy_efficiency = getattr(self.species, 'energy_efficiency', 1.0)
        total_energy_cost = (base_energy_cost + stagnation_penalty + repetition_penalty) / energy_efficiency
        
        self.energy_level -= total_energy_cost
        self.age += 1

    def consume_food(self, food):
        """Consume food to gain energy"""
        energy_gain = food.energy_value
        # Apply energy efficiency if available
        energy_efficiency = getattr(self.species, 'energy_efficiency', 1.0)
        energy_gain *= energy_efficiency
        
        self.energy_level = min(self.energy_level + energy_gain, self.species.max_energy)
        self.food_consumed += 1
        self.performance_score += 5
        self.last_food_time = 0  # Reset starvation counter

    def attack(self, other_cell):
        """Attack another cell"""
        if Species.can_eat(self.species.type, other_cell.species.type):
            # Attack success depends on attack power difference
            attack_success = np.random.random() < (self.species.attack_power / 
                                                 (self.species.attack_power + other_cell.species.attack_power))
            
            if attack_success:
                # Transfer energy from prey to predator
                energy_gain = min(other_cell.energy_level * 0.5, self.species.max_energy - self.energy_level)
                self.energy_level += energy_gain
                other_cell.energy_level = 0  # Prey dies
                self.performance_score += 10
                return True
                
        return False

    def collide_with_hazard(self, hazard):
        """Handle collision with environmental hazard"""
        # Apply hazard resistance if available from species adaptation
        resistance = getattr(self.species, 'radiation_resistance', 0.0) if 'radiat' in str(hazard.damage_value) else 0.0
        damage = hazard.damage_value * max(0.2, 1.0 - resistance * 0.5)  # Resistance reduces damage by up to 80%
        
        self.energy_level -= damage
        self.performance_score -= 5

    def reproduce(self):
        """Create offspring if enough energy"""
        reproduction_threshold = self.species.max_energy * 0.7
        
        if self.energy_level >= reproduction_threshold and self.age > 20:
            # Create child with small random position offset
            offset = (np.random.randint(-1, 2), np.random.randint(-1, 2))
            child_pos = (self.position[0] + offset[0], self.position[1] + offset[1])
            
            # Child inherits species but mutates
            mutated_species = self.species.mutate(mutation_strength=0.2)
            child = Cell(child_pos, mutated_species.type)
            child.species = mutated_species
            
            # Inherit some parent's adaptations with noise
            child.cold_resistance = self.cold_resistance * 0.9 + np.random.uniform(-0.05, 0.15)
            child.drought_resistance = self.drought_resistance * 0.9 + np.random.uniform(-0.05, 0.15)
            child.disease_resistance = self.disease_resistance * 0.9 + np.random.uniform(-0.05, 0.15)
            child.radiation_resistance = self.radiation_resistance * 0.9 + np.random.uniform(-0.05, 0.15)
            
            # Keep values in valid range
            child.cold_resistance = max(0, min(1, child.cold_resistance))
            child.drought_resistance = max(0, min(1, child.drought_resistance)) 
            child.disease_resistance = max(0, min(1, child.disease_resistance))
            child.radiation_resistance = max(0, min(1, child.radiation_resistance))
            
            # Parent loses energy to reproduce
            self.energy_level -= self.species.max_energy * 0.3
            self.performance_score += 20
            
            return child
        return None
    
    def update(self, environment):
        """Update the cell's state each time step"""
        # Apply passive energy depletion (cells lose energy over time)
        self.deplete_energy()
        
        # Rest of update logic would continue here
        # ...existing code...
    
    def deplete_energy(self):
        """Deplete energy over time, representing metabolic costs"""
        if self.is_alive():
            # Apply base metabolism cost
            self.energy_level -= self.energy_depletion_rate
            
            # Apply starvation effects - increasing penalty for not finding food
            self.last_food_time += 1
            if self.last_food_time > self.starvation_threshold:
                starvation_penalty = min(1.0, (self.last_food_time - self.starvation_threshold) * 0.01)
                self.energy_level -= starvation_penalty
                
                # Visual indication of starvation
                if hasattr(self, 'color'):
                    # Make cell appear more pale/weak as it starves
                    self.color = (
                        min(255, self.color[0] + 10),
                        max(0, self.color[1] - 5),
                        max(0, self.color[2] - 5)
                    )
    
    def is_alive(self):
        """Check if the cell is still alive"""
        return self.energy_level > 0
    
    def apply_adaptations(self, adaptations=None):
        """
        Apply adaptations to the cell based on learning or environmental factors.
        
        Args:
            adaptations (dict, optional): A dictionary of adaptations to apply.
                Keys are the attributes to adapt, values are the new values or deltas.
        """
        if adaptations is None:
            adaptations = {}
        
        # Apply each adaptation
        for attr, value in adaptations.items():
            if hasattr(self, attr):
                current_value = getattr(self, attr)
                
                # Check if we're applying an absolute value or a delta
                if isinstance(value, dict) and 'delta' in value:
                    new_value = current_value + value['delta']
                else:
                    new_value = value
                    
                # Apply any constraints if needed
                if isinstance(value, dict) and 'min' in value and new_value < value['min']:
                    new_value = value['min']
                if isinstance(value, dict) and 'max' in value and new_value > value['max']:
                    new_value = value['max']
                    
                # Update the attribute
                setattr(self, attr, new_value)
        
        # Update any derived attributes that depend on the adapted attributes
        self._update_derived_attributes()
        
        return self
    
    def _update_derived_attributes(self):
        """
        Update any attributes that are derived from the basic attributes.
        This is called after applying adaptations.
        """
        # Adjust speed based on energy level
        if hasattr(self, 'energy_level') and hasattr(self.species, 'speed'):
            energy_ratio = max(0.2, min(1.0, self.energy_level / self.species.max_energy))
            # We don't modify species.speed directly, but could apply temporary speed adjustments
            # or track an effective_speed attribute if needed