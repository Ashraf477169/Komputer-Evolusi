import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import streamlit as st
from itertools import permutations

# User Input for Cities and Coordinates
st.title("Genetic Algorithm for TSP with Custom City Coordinates")

n_cities = st.number_input("Enter number of cities", min_value=2, max_value=20, value=10)

cities_names = []
city_coords = {}

st.subheader("Enter City Names and Coordinates:")
for i in range(n_cities):
    city_name = st.text_input(f"City {i+1} Name:", f"City{i+1}")
    x_coord = st.number_input(f"{city_name} X Coordinate:", min_value=0.0, max_value=20.0, value=float(i))
    y_coord = st.number_input(f"{city_name} Y Coordinate:", min_value=0.0, max_value=20.0, value=float(i+1))
    cities_names.append(city_name)
    city_coords[city_name] = (x_coord, y_coord)

n_population = st.slider("Population Size", min_value=100, max_value=1000, value=250)
crossover_per = st.slider("Crossover Percentage", min_value=0.0, max_value=1.0, value=0.8)
mutation_per = st.slider("Mutation Percentage", min_value=0.0, max_value=1.0, value=0.2)
n_generations = st.slider("Number of Generations", min_value=50, max_value=500, value=200)

# Pastel Palette
colors = sns.color_palette("pastel", len(cities_names))

# City Icons
city_icons = {
    "JOHOR": "♕", "MELAKA": "♖", "N.SEMBILAN": "♗", "K.LUMPUR": "♘", "SELANGOR": "♙",
    "PERAK": "♔", "KEDAH": "♚", "PERLIS": "♛", "KELANTAN": "♜", "TERENGGANU": "♝"
}

fig, ax = plt.subplots()
ax.grid(False)

# Plot initial city locations with icons and labels
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons.get(city, "⛳")  # Default icon if city not in predefined list
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                textcoords='offset points')
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

fig.set_size_inches(16, 12)
st.pyplot(fig)

# Genetic Algorithm
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        total_dist += dist_two_cities(individual[i], individual[(i + 1) % len(individual)])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(indiv) for indiv in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = np.sum(population_fitness)
    return population_fitness / population_fitness_sum

def roulette_wheel(population, fitness_probs):
    cumsum_probs = np.cumsum(fitness_probs)
    selected_index = np.where(cumsum_probs > np.random.uniform(0, 1))[0][0]
    return population[selected_index]

def crossover(parent_1, parent_2):
    cut = random.randint(1, len(cities_names) - 2)
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    index_1, index_2 = random.sample(range(len(cities_names)), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        new_population = []
        for _ in range(int(n_population * crossover_per // 2)):
            parent_1 = roulette_wheel(population, fitness_probs)
            parent_2 = roulette_wheel(population, fitness_probs)
            offspring_1, offspring_2 = crossover(parent_1, parent_2)
            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)
            new_population.extend([offspring_1, offspring_2])
        population = sorted(new_population, key=total_dist_individual)[:n_population]
    return population[0]

# Run the Genetic Algorithm
best_path = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
min_distance = total_dist_individual(best_path)

st.write(f"Shortest Path Distance: {min_distance}")
st.write(f"Best Path: {best_path}")

# Plot the best path
x_shortest = [city_coords[city][0] for city in best_path] + [city_coords[best_path[0]][0]]
y_shortest = [city_coords[city][1] for city in best_path] + [city_coords[best_path[0]][1]]

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
for i, txt in enumerate(best_path):
    ax.annotate(f"{i+1} - {txt}", (x_shortest[i], y_shortest[i]), fontsize=12)
plt.title("TSP Best Route Using GA")
plt.suptitle(f"Total Distance: {round(min_distance, 3)} | Generations: {n_generations} | Population: {n_population}", fontsize=14)
fig.set_size_inches(16, 12)
st.pyplot(fig)
