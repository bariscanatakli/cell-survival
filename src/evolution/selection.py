def select_survivors(cells, survival_threshold):
    """
    Selects cells that perform above a certain threshold for reproduction.
    
    Parameters:
    - cells: List of cell objects to evaluate.
    - survival_threshold: Minimum performance score required for survival.
    
    Returns:
    - List of surviving cells.
    """
    return [cell for cell in cells if cell.performance_score >= survival_threshold]

def reproduce(selected_cells):
    """
    Generates offspring from selected cells.
    
    Parameters:
    - selected_cells: List of cells that have been selected for reproduction.
    
    Returns:
    - List of new offspring cells.
    """
    offspring = []
    for cell in selected_cells:
        # Create a new cell based on the parent's properties
        new_cell = cell.clone()  # Assuming the Cell class has a clone method
        offspring.append(new_cell)
    return offspring