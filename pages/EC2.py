import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns

# Streamlit UI for city input
st.title("City Coordinates Input")
st.write("Enter up to 10 cities with their coordinates (x, y) in range 1-10.")

# Function to input city data
num_cities = st.number_input("Number of Cities", min_value=2, max_value=10, step=1, value=5)
cities = []
for i in range(num_cities):
    city_name = st.text_input(f"City {i+1} Name", value=f"City {i+1}")
    x_coord = st.slider(f"x-coordinate (City {i+1})", min_value=1, max_value=10, value=random.randint(1, 10))
    y_coord = st.slider(f"y-coordinate (City {i+1})", min_value=1, max_value=10, value=random.randint(1, 10))
    cities.append((city_name, (x_coord, y_coord)))

if st.button("Run Genetic Algorithm"):
    # Prepare data for the genetic algorithm
    city_coords = dict(cities)
    cities_names = [name for name, _ in cities]
    
    # Parameters for the GA
    n_population = 250
    crossover_per = 0.8
    mutation_per = 0.2
    n_generations = 200
    
    # Helper functions for the GA
    def dist_two_cities(city_1, city_2):
        city_1_coords = city_coords[city_1]
        city_2_coords = city_coords[city_2]
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

    def total_dist_individual(individual):
        total_dist = 0
        for i in range(0, len(individual)):
            if i == len(individual) - 1:
                total_dist += dist_two_cities(individual[i], individual[0])
            else:
                total_dist += dist_two_cities(individual[i], individual[i + 1])
        return total_dist

    def initial_population(cities_list, n_population=250):
        population_perms = []
        possible_perms = list(permutations(cities_list))
        random_ids = random.sample(range(0, len(possible_perms)), n_population)
        for i in random_ids:
            population_perms.append(list(possible_perms[i]))
        return population_perms

    def fitness_prob(population):
        total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        population_fitness_sum = sum(population_fitness)
        population_fitness_probs = population_fitness / population_fitness_sum
        return population_fitness_probs

    def roulette_wheel(population, fitness_probs):
        population_fitness_probs_cumsum = fitness_probs.cumsum()
        selected_index = np.searchsorted(population_fitness_probs_cumsum, np.random.uniform(0, 1))
        return population[selected_index]

    def crossover(parent_1, parent_2):
        n_cities_cut = len(cities_names) - 1
        cut = round(random.uniform(1, n_cities_cut))
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    def mutation(offspring):
        n_cities_cut = len(cities_names) - 1
        index_1 = round(random.uniform(0, n_cities_cut))
        index_2 = round(random.uniform(0, n_cities_cut))
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        return offspring

    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(cities_names, n_population)
        for generation in range(n_generations):
            fitness_probs = fitness_prob(population)
            parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
            offspring_list = []
            for i in range(0, len(parents_list), 2):
                offspring_1, offspring_2 = crossover(parents_list[i], parents_list[(i + 1) % len(parents_list)])
                if random.random() > (1 - mutation_per):
                    offspring_1 = mutation(offspring_1)
                if random.random() > (1 - mutation_per):
                    offspring_2 = mutation(offspring_2)
                offspring_list.extend([offspring_1, offspring_2])
            mixed_offspring = parents_list + offspring_list
            fitness_probs = fitness_prob(mixed_offspring)
            sorted_indices = np.argsort(fitness_probs)[::-1]
            population = [mixed_offspring[i] for i in sorted_indices[:n_population]]
        return population

    best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
    total_dist_all_individuals = [total_dist_individual(ind) for ind in best_mixed_offspring]
    min_dist = min(total_dist_all_individuals)
    shortest_path = best_mixed_offspring[np.argmin(total_dist_all_individuals)]

    # Display results
    st.write(f"Shortest Path: {shortest_path}")
    st.write(f"Minimum Distance: {round(min_dist, 3)}")

    # Plotting the result
    x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
    y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', linewidth=2.5)
    for i, city in enumerate(shortest_path):
        ax.annotate(f"{i+1}- {city}", (x_shortest[i], y_shortest[i]), fontsize=12)
    plt.title(f"TSP Best Route Using GA\nTotal Distance: {round(min_dist, 3)}")
    fig.set_size_inches(10, 7)
    st.pyplot(fig)
