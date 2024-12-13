import csv
import random
import streamlit as st
import pandas as pd

##################################### FILE PROCESSING ###################################################
# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings
    return program_ratings

# Path to the CSV file
file_path = 'pages/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

all_programs = list(ratings.keys())  # all programs
all_time_slots = list(range(6, 24))  # time slots (06:00 to 23:00)

######################################### DEFINING FUNCTIONS ########################################################################
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# initializing the population
def initialize_pop(programs, time_slots):
    return [random.sample(programs, len(programs)) for _ in range(50)]  # Fixed population size to 50

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
def genetic_algorithm(initial_schedule, crossover_rate, mutation_rate):
    population = initialize_pop(all_programs, all_time_slots)

    for _ in range(100):  # Fixed number of generations to 100
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:2])  # Fixed elitism size to 2

        while len(new_population) < 50:
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

# Generate the schedule as a list of dictionaries for Streamlit table display
def generate_schedule_table(schedule, time_slots):
    table_data = []
    for time_slot, program in enumerate(schedule):
        start_time = f"{time_slots[time_slot]:02d}:00"
        end_time = f"{time_slots[time_slot] + 1:02d}:00"
        time_range = f"{start_time} - {end_time}"
        table_data.append({"Time Slot": time_range, "Program": program})
    return table_data

##################################### STREAMLIT UI ######################################################
# Streamlit UI
st.title("TV Program Scheduler with Genetic Algorithm")

# Input parameters
crossover_rate = st.slider("Crossover Rate (CO_R)", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
mutation_rate = st.slider("Mutation Rate (MUT_R)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# Genetic Algorithm
st.write("Running Genetic Algorithm...")
initial_schedule = random.sample(all_programs, len(all_programs))
genetic_schedule = genetic_algorithm(
    initial_schedule,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate
)

# Ensure schedule matches all time slots
if len(genetic_schedule) < len(all_time_slots):
    additional_programs = random.choices(all_programs, k=len(all_time_slots) - len(genetic_schedule))
    genetic_schedule += additional_programs

# Display the final schedule in a Streamlit table
st.write("### Final Optimal TV Schedule")
schedule_table = generate_schedule_table(genetic_schedule, all_time_slots)
st.table(schedule_table)

# Display total ratings
st.write(f"### Total Ratings: {fitness_function(genetic_schedule)}")
