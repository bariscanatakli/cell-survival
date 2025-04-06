from environment.species import SpeciesType

class AgentManager:
    def __init__(self, agents_dict):
        """
        Initialize the agent manager with a dictionary of agents
        
        Args:
            agents_dict: Dictionary mapping SpeciesType to agent instances
        """
        self.predator_agent = agents_dict.get(SpeciesType.PREDATOR)
        self.prey_agent = agents_dict.get(SpeciesType.PREY)
        self.gatherer_agent = agents_dict.get(SpeciesType.GATHERER)
        self.scavenger_agent = agents_dict.get(SpeciesType.SCAVENGER)
        
    def get_agent_for_type(self, agent_type, device=None):
        """
        Get the appropriate agent for a given agent type
        
        Args:
            agent_type: The type of agent to retrieve
            device: The device to use for agent computation (optional)
            
        Returns:
            The agent instance for the specified type
        """
        if agent_type == SpeciesType.PREDATOR:
            return self.predator_agent
        elif agent_type == SpeciesType.PREY:
            return self.prey_agent
        elif agent_type == SpeciesType.GATHERER:
            return self.gatherer_agent
        elif agent_type == SpeciesType.SCAVENGER:
            return self.scavenger_agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
    def get_agent_for_species(self, species_type):
        """
        Get the appropriate agent for a given species type without requiring device parameter
        
        Args:
            species_type: The SpeciesType to get an agent for
            
        Returns:
            The agent associated with the species type
        """
        # Use None for device parameter since it's only used for initialization
        return self.get_agent_for_type(species_type, device=None)
