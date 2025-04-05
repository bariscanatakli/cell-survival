def calculate_survival_rate(survivors, total_cells):
    if total_cells == 0:
        return 0
    return survivors / total_cells

def calculate_food_collection_efficiency(total_food_collected, total_time):
    if total_time == 0:
        return 0
    return total_food_collected / total_time

def display_metrics(survivors, total_cells, total_food_collected, total_time):
    survival_rate = calculate_survival_rate(survivors, total_cells)
    food_efficiency = calculate_food_collection_efficiency(total_food_collected, total_time)
    
    print(f"Survival Rate: {survival_rate:.2%}")
    print(f"Food Collection Efficiency: {food_efficiency:.2f} food units per time unit")