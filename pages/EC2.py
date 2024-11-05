import matplotlib.pyplot as plt
from itertools import permutations
from random import shuffle
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Koordinat kota dan inisialisasi
x = [0,3,6,7,15,10,16,5,8,1.5]
y = [1,2,1,4.5,-1,2.5,11,6,9,12]
cities_names = ["PERLIS", "KEDAH", "PENANG", "PERAK", "KELANTAN", "PAHANG", "JOHOR", "MELAKA", "SELANGOR", "TERENGGANU"]
city_coords = dict(zip(cities_names, zip(x, y)))
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Palet warna pastel untuk visualisasi
colors = sns.color_palette("pastel", len(cities_names))

# Ikon kota
city_icons = {
    "PERLIS": "♕",
    "KEDAH": "♖",
    "PENANG": "♗",
    "PERAK": "♘",
    "KELANTAN": "♙",
    "PAHANG": "♔",
    "JOHOR": "♚",
    "MELAKA": "♛",
    "SELANGOR": "♜",
    "TERENGGANU": "♝"
}

# Visualisasi awal kota dan rute
st.title("Travelling Salesman Problem using Genetic Algorithm")
fig, ax = plt.subplots()
ax.grid(False)
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons[city]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)
fig.set_size_inches(16, 12)
st.pyplot(fig)

# Fungsi untuk inisialisasi populasi
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

# Fungsi jarak antara dua kota
def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

# Fungsi jarak total individu
def total_dist_individual(individual):
    total_dist = 0
    for i in range(0, len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

# Fungsi probabilitas fitness
def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    return population_fitness / population_fitness_sum

# Fungsi seleksi roulette wheel
def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array]) - 1
    return population[selected_individual_index]

# Fungsi crossover
def crossover(parent_1, parent_2):
    n_cities_cut = len(cities_names) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = parent_1[0:cut] + [city for city in parent_2 if city not in parent_1[0:cut]]
    offspring_2 = parent_2[0:cut] + [city for city in parent_1 if city not in parent_2[0:cut]]
    return offspring_1, offspring_2

# Fungsi mutasi
def mutation(offspring):
    n_cities_cut = len(cities_names) - 1
    index_1 = round(random.uniform(0, n_cities_cut))
    index_2 = round(random.uniform(0, n_cities_cut))
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

# Menjalankan algoritma genetika
def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])
            if random.random() > (1 - mutation_per):
                offspring_1 = mutation(offspring_1)
            if random.random() > (1 - mutation_per):
                offspring_2 = mutation(offspring_2)
            offspring_list.extend([offspring_1, offspring_2])
        population = offspring_list + parents_list
    return population

# Jalankan dan tampilkan hasil terbaik
best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
total_dist_all_individuals = [total_dist_individual(ind) for ind in best_mixed_offspring]
index_minimum = np.argmin(total_dist_all_individuals)
minimum_distance = min(total_dist_all_individuals)
st.write("Minimum Distance:", minimum_distance)

shortest_path = best_mixed_offspring[index_minimum]
st.write("Shortest Path:", shortest_path)

# Visualisasi rute terbaik
x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i, txt in enumerate(shortest_path):
    ax.annotate(f"{i+1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

plt.title(f"TSP Best Route Using GA\n{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation", fontsize=18)
plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}", fontsize=20, y=1.047)
fig.set_size_inches(16, 12)
st.pyplot(fig)
