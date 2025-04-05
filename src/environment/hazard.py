class Hazard:
    def __init__(self, position, damage_value=5):
        self.position = position
        self.damage_value = damage_value

    def interact(self, cell):
        # Define interaction logic with the cell
        pass

    def get_position(self):
        return self.position