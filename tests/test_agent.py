import unittest
from src.agents.dqn_agent import DQNAgent
from src.environment.cell import Cell
from src.environment.food import Food
from src.environment.hazard import Hazard

class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DQNAgent(state_size=5, action_size=4)
        self.cell = Cell(position=(0, 0), energy_level=100)
        self.food = Food(position=(1, 1))
        self.hazard = Hazard(position=(2, 2))

    def test_action_selection(self):
        action = self.agent.select_action(self.cell)
        self.assertIn(action, range(self.agent.action_size))

    def test_training(self):
        initial_q_values = self.agent.model.predict(self.cell.get_state())
        self.agent.train(self.cell, self.food, reward=10)
        updated_q_values = self.agent.model.predict(self.cell.get_state())
        self.assertNotEqual(initial_q_values, updated_q_values)

    def test_reward_system(self):
        reward = self.agent.calculate_reward(self.cell, self.food)
        self.assertEqual(reward, 10)

    def test_collision_with_hazard(self):
        reward = self.agent.calculate_reward(self.cell, self.hazard)
        self.assertEqual(reward, -5)

if __name__ == '__main__':
    unittest.main()