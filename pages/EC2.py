import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Inisialisasi koordinat bandar dan nama
cities_names = ["Kuala Lumpur", "Pahang", "Kelantan", "Terengganu", "Kedah", "Melaka", "Johor", "Perlis", "Perak"]
default_coords = {
    "Kuala Lumpur": (9, 6),
    "Pahang": (5, 3),
    "Kelantan": (2, 1),
    "Terengganu": (3, 2),
    "Kedah": (1, 1),
    "Melaka": (8, 5),
    "Johor": (9, 2),
    "Perlis": (2, 6),
    "Perak": (1, 3)
}

# Warna untuk visualisasi
colors = sns.color_palette("pastel", len(cities_names))

# Ikon untuk setiap bandar
city_icons = {
    "Kuala Lumpur": "♔",
    "Pahang": "♕",
    "Kelantan": "♖",
    "Terengganu": "♗",
    "Kedah": "♘",
    "Melaka": "♙",
    "Johor": "♚",
    "Perlis": "♛",
    "Perak": "♜"
}

# Fungsi untuk mendapatkan input dari pengguna
st.title("City Coordinates Input")
city_coords = {}
for city in cities_names:
    x = st.number_input(f"x-coordinate ({city})", value=default_coords[city][0])
    y = st.number_input(f"y-coordinate ({city})", value=default_coords[city][1])
    city_coords[city] = (x, y)

# Butang submit
if st.button("Submit"):
    # Visualisasi setelah butang submit ditekan
    fig, ax = plt.subplots()
    ax.grid(False)
    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons.get(city, "•")
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=30, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')
    
    fig.set_size_inches(10, 8)
    st.pyplot(fig)


# Fungsi-fungsi untuk algoritma genetika
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
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    return population_fitness / sum(population_fitness)

def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array]) - 1
    return population[selected_individual_index]

def crossover(parent_1, parent_2):
    n_cities_cut = len(city_coords) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    n_cities_cut = len(city_coords) - 1
    index_1, index_2 = random.sample(range(n_cities_cut + 1), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

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

# Jalankan algoritma dan paparkan hasil
best_population = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
best_route = min(best_population, key=total_dist_individual)
min_distance = total_dist_individual(best_route)

st.write("Best Route:", best_route)
st.write("Minimum Distance:", min_distance)


# Visualisasi rute terbaik
x_best_route = [city_coords[city][0] for city in best_route] + [city_coords[best_route[0]][0]]
y_best_route = [city_coords[city][1] for city in best_route] + [city_coords[best_route[0]][1]]

fig, ax = plt.subplots()
ax.plot(x_best_route, y_best_route, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i, city in enumerate(best_route):
    ax.annotate(f"{i+1}- {city}", (x_best_route[i], y_best_route[i]), fontsize=12)

plt.title(f"Best TSP Route Using GA\nTotal Distance: {round(min_distance, 3)}", fontsize=18)
fig.set_size_inches(10, 8)
st.pyplot(fig)
