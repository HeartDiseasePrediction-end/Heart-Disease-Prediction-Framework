import time
import numpy as np

# Barber Optimization Algorithm (BOA)
def BOA(population, obj_function, lower_bound, upper_bound, max_iter):
    """Main function for bobcat_optimization_algorithm"""
    # Initialize population and evaluate initial objective values
    population_size, num_variables = population.shape[0], population.shape[1]
    obj_values = np.array([obj_function(ind) for ind in population])

    best_position = population[np.argmin(obj_values)]
    best_value = np.min(obj_values)
    convergence = np.zeros(max_iter)

    ct = time.time()
    # Optimization process
    for t in range(max_iter):
        for i in range(population_size):

            if t < max_iter // 2:  # Exploration phase
                prey_indices = np.where(obj_values < obj_values[i])[0]
                prey_position = population[prey_indices[np.random.choice(len(prey_indices))]]
                r = np.random.rand(len(prey_position))
                new_position = population[i] + (1 - 2 * r) * (prey_position - population[i])
                new_position = np.clip(new_position, lower_bound, upper_bound)

            else:  # Exploitation phase
                r = np.random.rand(len(best_position))
                new_position = population[i] + (1 - 2 * r) * (best_position - population[i])
                new_position = np.clip(new_position, lower_bound, upper_bound)

            new_value = obj_function(new_position)

            # Update if new position is better
            if new_value < obj_values[i]:
                population[i] = new_position
                obj_values[i] = new_value
                if new_value < best_value:
                    best_position = new_position
                    best_value = new_value
        convergence[t] = best_value

    ct = time.time() - ct
    return best_value, convergence,  best_position, ct