from matplotlib import pyplot as plt
import numpy as np

def render_environment(cells, food_sources, hazards, world_size):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, world_size[0])
    plt.ylim(0, world_size[1])
    
    # Render cells
    for cell in cells:
        plt.scatter(cell.position[0], cell.position[1], color='blue', s=100, label='Cell' if cell == cells[0] else "")
    
    # Render food sources
    for food in food_sources:
        plt.scatter(food.position[0], food.position[1], color='green', s=100, label='Food' if food == food_sources[0] else "")
    
    # Render hazards
    for hazard in hazards:
        plt.scatter(hazard.position[0], hazard.position[1], color='red', s=100, label='Hazard' if hazard == hazards[0] else "")
    
    plt.title('Cell Survival Simulation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid()
    plt.show()

def plot_learning_curve(learning_curve):
    plt.figure(figsize=(10, 5))
    plt.plot(learning_curve)
    plt.title('Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.grid()
    plt.show()