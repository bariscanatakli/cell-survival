import unittest
from src.environment.cell import Cell
from src.environment.food import Food
from src.environment.hazard import Hazard
from src.environment.world import World

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.world = World()
        self.cell = Cell(position=(5, 5), energy_level=100)
        self.food = Food(position=(3, 3))
        self.hazard = Hazard(position=(7, 7))
        self.world.add_entity(self.cell)
        self.world.add_entity(self.food)
        self.world.add_entity(self.hazard)

    def test_cell_movement(self):
        initial_position = self.cell.position
        self.cell.move('up')
        self.assertNotEqual(self.cell.position, initial_position)

    def test_cell_consumes_food(self):
        self.cell.position = self.food.position
        initial_energy = self.cell.energy_level
        self.cell.consume_food(self.food)
        self.assertGreater(self.cell.energy_level, initial_energy)

    def test_cell_collides_with_hazard(self):
        self.cell.position = self.hazard.position
        initial_energy = self.cell.energy_level
        self.cell.collide_with_hazard(self.hazard)
        self.assertLess(self.cell.energy_level, initial_energy)

    def test_world_update(self):
        initial_state = self.world.get_state()
        self.world.update()
        self.assertNotEqual(self.world.get_state(), initial_state)

if __name__ == '__main__':
    unittest.main()