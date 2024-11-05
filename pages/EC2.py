import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st
import numpy as np
import random
from itertools import permutations
import seaborn as sns

# Initialize city data
x = [0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5]
y = [1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]
cities_names = ["PERLIS", "KEDAH", "PENANG", "PERAK", "KELANTAN", "PAHANG", "JOHOR", "MELAKA", "SELANGOR", "TERENGGANU"]
city_coords = dict(zip(cities_names, zip(x, y)))
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Set up Streamlit
st.title("Traveling Salesman Problem Using Genetic Algorithm")
st.write("Visualizing the progression of route optimization.")

# Function to calculate distance between two cities
def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

# Function to calculate total distance of an individual route
def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1])
    return total_dist

# Generate initial population
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

# Fitness function
def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    return population_fitness_probs

# Roulette wheel selection
def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    selected_index = np.searchsorted(population_fitness_probs_cumsum, np.random.uniform(0, 1))
    return population[selected_index]

# Crossover function
def crossover(parent_1, parent_2):
    cut = round(random.uniform(1, len(cities_names) - 1))
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

# Mutation function
def mutation(offspring):
    index_1, index_2 = random.sample(range(len(offspring)), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

# Run genetic algorithm
def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    best_distances = []
    
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
        
        min_dist = total_dist_individual(population[0])
        best_distances.append(min_dist)
        
        yield population[0], min_dist  # Yield the best route and its distance for each generation

# Plot animation function
fig, ax = plt.subplots()
line, = ax.plot([], [], '--go', linewidth=2.5)
title = ax.set_title("")

def update(frame):
    route, distance = frame
    x_vals = [city_coords[city][0] for city in route] + [city_coords[route[0]][0]]
    y_vals = [city_coords[city][1] for city in route] + [city_coords[route[0]][1]]
    line.set_data(x_vals, y_vals)
    title.set_text(f"Generation Distance: {round(distance, 2)}")
    return line, title

# Run the animation
ani = FuncAnimation(fig, update, frames=run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per), 
                    repeat=False, interval=200)
fig.set_size_inches(16, 12)

# Display the animation
st.pyplot(fig)
