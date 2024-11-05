import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Nama kota-kota yang tersedia
cities_names = ["Kuala Lumpur", "Pahang", "Kelantan", "Terengganu", "Kedah", "Melaka", "Johor", "Perlis", "Perak"]
# Pastel Pallete

# Input koordinat untuk setiap kota
st.title("Input Coordinates for Cities")
city_coords = {}
for i, city in enumerate(cities_names):
    col1, col2, col3 = st.columns(3)
    with col1:
        city_name = st.text_input(f"City {i+1}", city, key=f"city_name_{i}")
    with col2:
        x_coord = st.number_input(f"x-coordinate (City {i+1})", value=random.randint(0, 10), step=1, key=f"x_{i}")
    with col3:
        y_coord = st.number_input(f"y-coordinate (City {i+1})", value=random.randint(0, 10), step=1, key=f"y_{i}")
    city_coords[city_name] = (x_coord, y_coord)
# Butang submit
if st.button("Submit"):
# Parameter untuk algoritma genetika
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Palet warna pastel untuk visualisasi
colors = sns.color_palette("pastel", len(city_coords))
# Ikon kota yang baru dimasukkan oleh pengguna
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

# Visualisasi awal kota dan rute dengan ikon
fig, ax = plt.subplots()
ax.grid(False)
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons.get(city, "•")  # Gunakan ikon atau '•' jika ikon tiada
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=30, ha='center', va='center', zorder=3)  # Letak ikon
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')
    
    # Sambungkan garis antara kota
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

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
