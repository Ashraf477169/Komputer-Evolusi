import csv
import streamlit as st
import random
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

######################################### STREAMLIT UI ###################################################
# Default values for genetic algorithm parameters
DEFAULT_CO_R = 0.8
DEFAULT_MUT_R = 0.2
GEN = 100
POP = 50
EL_S = 2
SEED = 42  # Seed for reproducibility

# Streamlit UI
def main():
    st.title("Genetic Algorithm for Optimal Scheduling")

    # Genetic Algorithm Parameters
    st.header("Input Genetic Algorithm Parameters")
    co_r = st.slider(
        "Crossover Rate (CO_R)",
        min_value=0.0,
        max_value=0.95,
        value=DEFAULT_CO_R,
        step=0.01,
        help="Select the crossover rate (range: 0.0 to 0.95)."
    )

    mut_r = st.slider(
        "Mutation Rate (MUT_R)",
        min_value=0.01,
        max_value=0.1,
        value=DEFAULT_MUT_R,
        step=0.01,
        help="Select the mutation rate (range: 0.01 to 0.1)."
    )

    st.write("### Selected Parameters:")
    st.write(f"- **Crossover Rate (CO_R)**: {co_r}")
    st.write(f"- **Mutation Rate (MUT_R)**: {mut_r}")

    # Sample rating programs dataset for each time slot.
    ratings = program_ratings_dict

    all_programs = list(ratings.keys())  # all programs
    all_time_slots = list(range(6, 24))  # time slots

    ######################################### DEFINING FUNCTIONS ##########################################
    # defining fitness function
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += ratings[program][time_slot]
        return total_rating

    # initializing the population
    def initialize_pop(programs, time_slots):
        return [random.sample(programs, len(programs)) for _ in range(POP)]

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
    def genetic_algorithm(generations=GEN, population_size=POP, crossover_rate=co_r, mutation_rate=mut_r, elitism_size=EL_S):
        # Set random seed for reproducibility
        random.seed(SEED)

        # Initialize population
        population = initialize_pop(all_programs, all_time_slots)

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

            population = new_population[:population_size]

        return max(population, key=fitness_function)

    ############################################# RESULTS ###############################################
    # Run Genetic Algorithm
    optimal_schedule = genetic_algorithm()

    schedule_data = []
    for time_slot, program in enumerate(optimal_schedule):
        schedule_data.append({"Time Slot": f"{all_time_slots[time_slot]:02d}:00", "Program": program})

    # Display schedule in table format
    df_schedule = pd.DataFrame(schedule_data)
    st.write("\nFinal Optimal Schedule:")
    st.table(df_schedule)

    st.write("Total Ratings:", fitness_function(optimal_schedule))

if __name__ == "__main__":
    main()
