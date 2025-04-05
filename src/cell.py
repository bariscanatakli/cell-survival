def get_cell_type_indices(self, cell_type):
    """Returns indices of cells matching the given type"""
    indices = [i for i, cell in enumerate(self.cells) if cell.cell_type == cell_type]
    
    # If no cells of this type exist, return an empty list rather than trying to access a non-existent item
    if not indices:
        return []
    
    return indices
