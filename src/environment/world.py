import numpy as np
import torch
import math
import random

from .cell import Cell
from .food import Food
from .hazard import Hazard
from .species import SpeciesType, Species
from .challenges import EnvironmentalChallenge, DisasterType


def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


class World:
    def __init__(self, width=1024, height=1024, num_cells=30, num_foods=200, num_hazards=30, 
                 initial_cell_distribution=None, device=None):
        self.width = width
        self.height = height
        self.num_cells = num_cells
        self.num_foods = num_foods
        self.num_hazards = num_hazards
        self.cells = []
        self.foods = []
        self.hazards = []
        self.challenges = EnvironmentalChallenge()
        self.competition_factor = 1.0
        self.initial_cell_distribution = initial_cell_distribution or {
            "PREY": 0.3,
            "PREDATOR": 0.3,
            "GATHERER": 0.2,
            "SCAVENGER": 0.2,
        }

        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"World simulation using device: {self.device}")

        self.grid_cell_size = 50
        self.grid_rows = math.ceil(height / self.grid_cell_size)
        self.grid_cols = math.ceil(width / self.grid_cell_size)
        self.spatial_grid = {}

        # GPU tensors for batch operations
        self.cell_positions_gpu = None
        self.food_positions_gpu = None
        self.hazard_positions_gpu = None

        # Grids for exploration and resource tracking
        self.exploration_bonus_grid = np.zeros((width, height))
        self.resource_depletion_grid = np.zeros((width, height))
        self.state_size = 16  # Expanded state size including environmental factors
        self.action_size = 6  # e.g., up, down, left, right, attack, reproduce
        self.day_cycle = 0
        self.max_day_cycle = 100
        self.episode = 0

        # Tracking structures for cell positions and stagnation
        self.cell_position_history = {}
        self.last_position_update = {}
        self.stagnation_threshold = 30
        self.severe_stagnation_threshold = 50

    def reset(self):
        """Reset the environment to its initial state."""
        self.cells = []
        self.foods = []
        self.hazards = []
        self.decay = []
        self.day_cycle = 0
        self.episode += 1

        # Update environmental challenges with new episode
        self.challenges.update(0, self.episode)

        # Add cells for each species
        initial_count = 5
        for species in list(SpeciesType):
            for _ in range(initial_count):
                pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
                cell = Cell(position=pos, species_type=species)
                self.cells.append(cell)

        # Add food based on food multiplier (seasonal factor)
        food_count = int(100 * self.challenges.get_food_multiplier())
        for _ in range(food_count):
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            self.foods.append(Food(position=pos))

        # Add hazards; hazard count increases with episode
        hazard_count = 30 + min(20, self.episode * 2)
        for _ in range(hazard_count):
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            damage = 5 + min(10, self.episode)
            self.hazards.append(Hazard(position=pos, damage_value=damage))

        # Initialize tracking dictionaries for each cell
        self.cell_position_history = {id(cell): [] for cell in self.cells}
        self.last_position_update = {id(cell): 0 for cell in self.cells}
        return self.get_state(self.cells[0]) if self.cells else np.zeros((1, self.state_size))

    def get_state(self, cell):
        """Get state representation for a specific cell."""
        state = np.zeros(self.state_size)
        # Normalize cell positions and energy level
        state[0] = cell.position[0] / self.width
        state[1] = cell.position[1] / self.height
        state[2] = cell.energy_level / cell.species.max_energy
        state[3] = cell.age / 100  # Assume age normalization

        # One-hot encode species type (positions 4,5,6,7)
        species_idx = list(SpeciesType).index(cell.species.type)
        state[4 + species_idx] = 1

        # Calculate distance to the nearest food
        if self.foods:
            min_dist = float("inf")
            for food in self.foods:
                dist = self.distance(cell.position, food.position)
                if dist < min_dist:
                    min_dist = dist
            state[8] = min_dist / (self.width * 0.1)
        else:
            state[8] = 1.0

        # Calculate distance to the nearest hazard
        if self.hazards:
            min_dist = float("inf")
            for hazard in self.hazards:
                dist = self.distance(cell.position, hazard.position)
                if dist < min_dist:
                    min_dist = dist
            state[9] = min_dist / (self.width * 0.1)
        else:
            state[9] = 1.0

        # Placeholder for predator and prey distances
        predator_dist = float("inf")
        prey_dist = float("inf")
        for other in self.cells:
            if other == cell:
                continue
            dist = self.distance(cell.position, other.position)
            if Species.can_eat(other.species.type, cell.species.type):
                predator_dist = min(predator_dist, dist)
            # Add prey distance logic if applicable here
        state[10] = predator_dist / (self.width * 0.1) if predator_dist != float("inf") else 1.0
        state[11] = prey_dist / (self.width * 0.1) if prey_dist != float("inf") else 1.0

        # Environmental factors (disaster state, day cycle, competition factor)
        active_disasters = self.challenges.get_active_disasters()
        state[12] = (
            float(list(DisasterType).index(active_disasters[0])) / len(list(DisasterType))
            if active_disasters
            else 0
        )
        state[13] = len(active_disasters) / 4.0
        state[14] = self.day_cycle / self.max_day_cycle
        state[15] = self.competition_factor

        return state.reshape(1, -1)

    def _step(self, cell_index, action):
        """Perform a simulation step for a single cell."""
        cell = self.cells[cell_index]
        prev_position = cell.position
        reward = 0

        # Example action dispatch
        if action == 0:  # Move up
            cell.move("up")
        elif action == 1:  # Move down
            cell.move("down")
        elif action == 2:  # Move left
            cell.move("left")
        elif action == 3:  # Move right
            cell.move("right")
        elif action == 4:  # Attack
            # Implement attack logic here
            pass
        elif action == 5:  # Reproduce
            child = cell.reproduce()
            if child:
                self.cells.append(child)
                reward = 10
            else:
                reward = -2

        # Ensure toroidal world boundaries
        cell.position = (cell.position[0] % self.width, cell.position[1] % self.height)

        # Process interactions and environmental effects
        self.process_interactions(cell)
        self.challenges.apply_effects(cell)

        # Update stagnation tracking
        cell_id = id(cell)
        if prev_position != cell.position:
            self.cell_position_history[cell_id].append(cell.position)
            if len(self.cell_position_history[cell_id]) > 10:
                self.cell_position_history[cell_id].pop(0)
            self.last_position_update[cell_id] = 0
        else:
            self.last_position_update[cell_id] = self.last_position_update.get(cell_id, 0) + 1
            if self.last_position_update[cell_id] > self.severe_stagnation_threshold:
                reward -= 1.0
                cell.energy_level -= 0.5 * self.challenges.get_energy_drain_multiplier()
            elif self.last_position_update[cell_id] > self.stagnation_threshold:
                reward -= 0.1

        # Remove expired food and possibly add new food
        expired_food = []
        for i, food in enumerate(self.foods):
            if food.update():
                expired_food.append(i)
        for i in sorted(expired_food, reverse=True):
            self.foods.pop(i)
        food_generation_chance = 0.05 * self.challenges.get_food_multiplier() + len(expired_food) * 0.01
        if random.random() < food_generation_chance:
            self.add_new_food()

        # Update resource grid and day cycle
        self.resource_depletion_grid = np.maximum(0, self.resource_depletion_grid - 0.05)
        self.day_cycle += 1
        if self.day_cycle >= self.max_day_cycle:
            self.day_cycle = 0
            self.challenges.update(self.day_cycle, self.episode)

        done = not cell.is_alive()
        self.manage_decay()

        return self.get_state(cell), reward, done

    def step(self, data, action=None):
        """
        Updated step method.
        If data is a dict (batch of actions), iterate over each (cell_index, action).
        Otherwise, treat 'data' as a single cell index and 'action' as its action.
        Returns (next_states, rewards, dones, info) for batch, or (next_state, reward, done, info) for single step.
        """
        if isinstance(data, dict):
            next_states = {}
            rewards = {}
            dones = {}
            for cell_index, act in data.items():
                state, rew, done = self._step(cell_index, act)
                next_states[cell_index] = state
                rewards[cell_index] = rew
                dones[cell_index] = done
            info = {}  # Add any additional info if needed
            return next_states, rewards, dones, info
        else:
            next_state, reward, done = self._step(data, action)
            info = {}  # Add any additional info if needed
            return next_state, reward, done, info
    def process_interactions(self, cell):
        """Process cell's interactions with food and hazards."""
        consumed_food = []
        # Process food consumption and resource depletion
        for i, food in enumerate(self.foods):
            if self.is_collision(cell, food):
                x, y = int(cell.position[0]) % self.width, int(cell.position[1]) % self.height
                depletion_radius = 2
                for dx in range(-depletion_radius, depletion_radius + 1):
                    for dy in range(-depletion_radius, depletion_radius + 1):
                        nx, ny = (x + dx) % self.width, (y + dy) % self.height
                        self.resource_depletion_grid[nx, ny] += 0.3
                depletion_factor = min(0.8, self.resource_depletion_grid[x, y])
                energy_gain = food.energy_value * max(0.2, 1.0 - depletion_factor)
                cell.energy_level = min(cell.species.max_energy, cell.energy_level + energy_gain)
                cell.performance_score += 5
                consumed_food.append(i)
        for i in sorted(consumed_food, reverse=True):
            self.foods.pop(i)

        # Process hazard interactions
        for hazard in self.hazards:
            if self.is_collision(cell, hazard):
                cell.collide_with_hazard(hazard)

    def add_new_food(self):
        """Add new food to the world with consideration for resource depletion."""
        cell_positions = [cell.position for cell in self.cells]
        candidate_positions = []
        if cell_positions:
            min_depletion = np.min(self.resource_depletion_grid)
            for _ in range(20):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if self.resource_depletion_grid[x, y] < min_depletion + 0.1:
                    candidate_positions.append((x, y))
            if candidate_positions:
                pos = random.choice(candidate_positions)
                pos = ((pos[0] + random.randint(-5, 5)) % self.width, (pos[1] + random.randint(-5, 5)) % self.height)
                self.foods.append(Food(position=pos))
                return
        pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        self.foods.append(Food(position=pos))

    def manage_decay(self):
        """Manage decay from dead cells and perform clean-up."""
        self.cells = [cell for cell in self.cells if cell.is_alive()]
        self.decay = [d for d in getattr(self, "decay", []) if random.random() > 0.2]
        for decay_item in self.decay:
            if decay_item not in self.foods:
                self.foods.append(decay_item)

    def update_gpu_tensors(self):
        """Update position tensors for GPU computation."""
        # ...existing GPU update logic...
        pass

    def update(self):
        """Update world state, adjust environmental challenges, and update GPU tensors."""
        self.update_gpu_tensors()
        # Other update logic...
        if self.challenges:
            self.challenges.apply(self)

    def batch_nearest_objects(self, reference_positions, target_positions, max_distance=float("inf")):
        """Find nearest objects using GPU-accelerated batch computation."""
        # ...implementation for batch nearest objects...
        pass

    def is_collision(self, entity1, entity2):
        """Check if two entities are at the same position."""
        return (
            entity1.position[0] == entity2.position[0]
            and entity1.position[1] == entity2.position[1]
        )

    def distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return calculate_distance(point1, point2)

    def get_observations(self, observation_radius=50):
        """
        Returns a dictionary of observations for each cell.
        Each observation includes the cell's position, nearby food and hazards.
        """
        # Ensure observation_radius is numeric
        if isinstance(observation_radius, list):
            try:
                observation_radius = float(observation_radius[0])
            except (TypeError, IndexError):
                raise ValueError("observation_radius must be a numeric value, not a list.")
    
        observations = {}
        for cell in self.cells:
            observation = {
                "position": cell.position,
                "nearby_foods": self._get_nearby(self.foods, cell.position, observation_radius),
                "nearby_hazards": self._get_nearby(self.hazards, cell.position, observation_radius),
            }
            observations[id(cell)] = observation
        return observations

    def _get_nearby(self, objects, position, radius):
        """
        Returns a list of objects (food or hazards) within the given radius of a position.
        """
        # Guard against radius being passed as a list
        if isinstance(radius, list):
            try:
                radius = float(radius[0])
            except (TypeError, IndexError):
                raise ValueError("radius must be a numeric value, not a list.")
        return [obj for obj in objects if self.distance(obj.position, position) <= radius]