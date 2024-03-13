import numpy as np
import pygad

# Definir la función Sphere
def sphere_function(solution):
    return np.sum(np.power(solution, 2))

# Definir la función de aptitud
def fitness_func(ga_instance, solution, solution_idx):
    return 1.0 / (sphere_function(solution) + 0.000001)

# Definir límites de búsqueda
search_space = np.array([[-5, 5]] * 3)  # Tres variables de decisión en el rango [-5, 5]

# Crear un objeto de configuración de PyGAD
ga_instance = pygad.GA(num_generations=50, 
                       num_parents_mating=2, 
                       sol_per_pop=10, 
                       num_genes=3, 
                       fitness_func=fitness_func, 
                       gene_space=search_space)

# Ejecutar la optimización
ga_instance.run()

# Obtener la mejor solución
solution, solution_fitness, solution_idx = ga_instance.best_solution()

# Imprimir la mejor solución
print("Solución óptima: {solution}".format(solution=solution))
print("Valor óptimo: {solution_fitness}".format(solution_fitness=1.0 / solution_fitness))
