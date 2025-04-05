def mutate_weights(weights, mutation_rate=0.1, mutation_strength=0.05):
    """
    Apply mutations to the neural network weights.

    Parameters:
    weights (numpy.ndarray): The original weights of the neural network.
    mutation_rate (float): The probability of mutation for each weight.
    mutation_strength (float): The magnitude of the mutation.

    Returns:
    numpy.ndarray: The mutated weights.
    """
    mutated_weights = weights.copy()
    for i in range(mutated_weights.shape[0]):
        if np.random.rand() < mutation_rate:
            mutation = np.random.randn(*mutated_weights[i].shape) * mutation_strength
            mutated_weights[i] += mutation
    return mutated_weights

def mutate_population(population, mutation_rate=0.1, mutation_strength=0.05):
    """
    Apply mutations to a population of neural networks.

    Parameters:
    population (list): A list of neural network weights representing the population.
    mutation_rate (float): The probability of mutation for each weight.
    mutation_strength (float): The magnitude of the mutation.

    Returns:
    list: The mutated population of neural network weights.
    """
    mutated_population = []
    for weights in population:
        mutated_weights = mutate_weights(weights, mutation_rate, mutation_strength)
        mutated_population.append(mutated_weights)
    return mutated_population