import csv
import random
import streamlit as st

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)

        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings

    return program_ratings

# Load the CSV data (replace 'uploaded_file' with the actual file path)
file_path = '/mnt/data/program_ratings.csv'  # Replace with your uploaded file path
program_ratings_dict = read_csv_to_dict(file_path)

# Streamlit Interface for Parameters
st.title("Optimal TV Program Scheduling")

st.sidebar.header("Genetic Algorithm Parameters")
CO_R = st.sidebar.slider("Crossover Rate", min_value=0.0, max_value=0.95, value=0.8, step=0.01)
MUT_R = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.05, value=0.2, step=0.01)

GEN = 100
POP = 50
EL_S = 2

# Data Setup
all_programs = list(program_ratings_dict.keys())  # All programs
all_time_slots = list(range(6, 24))  # Time slots

# Fitness Function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += program_ratings_dict[program][time_slot]
    return total_rating

# Population Initialization
def initialize_population(programs, time_slots, population_size):
    population = []
    for _ in range(population_size):
        schedule = random.sample(programs, len(programs))
        population.append(schedule)
    return population

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Genetic Algorithm
def genetic_algorithm(generations, population_size, crossover_rate, mutation_rate, elitism_size):
    population = initialize_population(all_programs, all_time_slots, population_size)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

# Run the Genetic Algorithm
final_schedule = genetic_algorithm(GEN, POP, CO_R, MUT_R, EL_S)

# Display Results
st.header("Optimal Schedule")

data = []
for time_slot, program in enumerate(final_schedule[:len(all_time_slots)]):
    data.append({"Time Slot": f"{all_time_slots[time_slot]}:00", "Program": program})

st.table(data)

# Total Ratings
st.write("**Total Ratings:**", fitness_function(final_schedule))
