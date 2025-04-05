class Food:
    def __init__(self, position, energy_value=10, is_decay=False):
        self.position = position
        self.energy_value = energy_value
        self.is_decay = is_decay
        self.lifetime = 0  # Track how long the food has existed

    def consume(self):
        # Logic for being consumed by a cell
        pass
        
    def update(self):
        """
        Update the food state and return True if food should be removed.
        For decaying food, reduce energy value over time and remove when depleted.
        """
        self.lifetime += 1
        
        if self.is_decay:
            # Decaying food loses energy over time
            self.energy_value = max(0, self.energy_value - 0.5)
            
            # Remove when energy is depleted
            if self.energy_value <= 0:
                return True
                
        return False