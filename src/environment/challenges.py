import random
import numpy as np
from enum import Enum, auto

class SeasonType(Enum):
    SPRING = auto()  # Abundant food, moderate temperature
    SUMMER = auto()  # High temperature, moderate food
    AUTUMN = auto()  # Moderate temperature, high food
    WINTER = auto()  # Low temperature, scarce food

class DisasterType(Enum):
    DROUGHT = auto()      # Reduces food generation
    DISEASE = auto()      # Reduces cell energy over time
    RADIATION = auto()    # Damages cells in certain areas
    POLLUTION = auto()    # Reduces reproduction success rate

class EnvironmentalChallenge:
    def __init__(self):
        self.current_season = SeasonType.SPRING
        self.season_day = 0
        self.season_length = 100  # Days per season
        self.disasters = []
        self.disaster_chance = 0.005  # 0.5% chance per day of a new disaster
        self.global_difficulty = 1.0  # Scales with time to make game progressively harder
        self.difficulty_increase_rate = 0.01  # Difficulty increases 1% per episode
    
    def update(self, day, episode):
        # Update season
        self.season_day += 1
        if self.season_day >= self.season_length:
            self.season_day = 0
            # Rotate to next season
            seasons = list(SeasonType)
            current_idx = seasons.index(self.current_season)
            next_idx = (current_idx + 1) % len(seasons)
            self.current_season = seasons[next_idx]
        
        # Update global difficulty based on episode number
        self.global_difficulty = 1.0 + (episode * self.difficulty_increase_rate)
        
        # Random chance for disaster
        if random.random() < self.disaster_chance * self.global_difficulty:
            disaster = random.choice(list(DisasterType))
            duration = random.randint(10, 50)
            self.disasters.append((disaster, duration))
        
        # Update existing disasters (decrease duration)
        self.disasters = [(d, duration-1) for d, duration in self.disasters if duration > 1]
    
    def get_active_disasters(self):
        return [d for d, _ in self.disasters]
    
    def get_food_multiplier(self):
        """Return multiplier for food generation based on season and disasters"""
        multiplier = 1.0
        
        # Season effects
        if self.current_season == SeasonType.SPRING:
            multiplier *= 1.2
        elif self.current_season == SeasonType.SUMMER:
            multiplier *= 0.9
        elif self.current_season == SeasonType.AUTUMN:
            multiplier *= 1.4
        elif self.current_season == SeasonType.WINTER:
            multiplier *= 0.6
        
        # Disaster effects
        if DisasterType.DROUGHT in self.get_active_disasters():
            multiplier *= 0.5
        
        # Scale by global difficulty (food becomes more scarce over time)
        multiplier /= self.global_difficulty
        
        return multiplier
    
    def get_energy_drain_multiplier(self):
        """Return multiplier for how much extra energy cells lose"""
        multiplier = 1.0
        
        # Season effects
        if self.current_season == SeasonType.WINTER:
            multiplier *= 1.3  # Harsher winter conditions
        
        # Disaster effects
        if DisasterType.DISEASE in self.get_active_disasters():
            multiplier *= 1.5
        
        # Scale by global difficulty
        multiplier *= self.global_difficulty
        
        return multiplier
    
    def apply_effects(self, cell):
        """Apply environmental effects to a cell"""
        # Apply disaster effects
        if DisasterType.RADIATION in self.get_active_disasters():
            # Radiation damages 25% of cells each cycle
            if random.random() < 0.25:
                cell.energy_level -= 5 * self.global_difficulty
        
        if DisasterType.POLLUTION in self.get_active_disasters():
            # Pollution reduces reproduction success
            cell.reproduction_penalty = 0.5
        else:
            cell.reproduction_penalty = 0.0
