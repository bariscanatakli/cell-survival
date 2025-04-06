class AgentManager:
    def __init__(self):
        self.agents = {}  # Map from species type to agent
        
    def add_agent(self, species_type, agent):
        self.agents[species_type] = agent
        
    def get_agent_for_type(self, species_type, device=None):
        """Get agent for a specific species type
        
        Args:
            species_type: The type of species
            device: The device to use (CPU/GPU) - optional
        
        Returns:
            The agent for the specified species type
        """
        if species_type is None:
            return None
        
        # Handle potential string vs enum comparison
        if isinstance(species_type, str):
            for key in self.agents:
                if key.name == species_type:
                    return self.agents[key]
        
        # Direct access by enum value
        return self.agents.get(species_type)
        
    # Add this method for backward compatibility
    def get_agent_for_species(self, species_type):
        """Alias for get_agent_for_type for backward compatibility"""
        return self.get_agent_for_type(species_type)
