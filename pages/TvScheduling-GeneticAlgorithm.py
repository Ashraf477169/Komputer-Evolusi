import csv
import streamlit as st
import random
import pandas as pd

##################################### FILE PROCESSING ###################################################
# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    # Read the file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract program names and ratings
    for _, row in df.iterrows():
        program = row["Type of Program"]
        ratings = row.iloc[1:].values.tolist()  # Extract all hourly ratings as a list
        program_ratings[program] = [float(r) for r in ratings]  # Ensure ratings are floats

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
        max_value=0.05,
        value=DEFAULT_MUT_R,
        step=0.01,
        help="Select the mutation rate (range: 0.01 to 0.05)."
    )

    st.write("### Selected Parameters:")
    st.write(f"- **Crossover Rate (CO_R)**: {co_r}")
    st.write(f"- **Mutation Rate (MUT_R)**: {mut_r}")

    # Sample rating programs dataset for each time slot.
    ratings = program_ratings_dict

    all_programs = list(ratings.keys())  # all programs
    all_time_slots = list(range(len(next(iter(ratings.values())))))  # time slots based on ratings

    ######################################### DEFINING FUNCTIONS ##########################################
    # defining fitness function
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += ratings[program][time_slot]
        return total_rating

    # initializing the population
    def initialize_pop(programs, time_slots):
        population = []
        for _ in range(POP):
            random_schedule = random.sample(programs, len(time_slots))
            population.append(random_schedule)
        return population

    # selection
    def select_best(population):
        return sorted(population, key=fitness_function, reverse=True)[:EL_S]

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
    def genetic_algorithm(generations=GEN, population_size=POP, crossover_rate=co_r, mutation_rate=mut_r):
        population = initialize_pop(all_programs, all_time_slots)

        for generation in range(generations):
            new_population = select_best(population)

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

        return max(population, key=fitness_function)

    ############################################# RESULTS ###############################################
    optimal_schedule = genetic_algorithm()

    st.write("\nFinal Optimal Schedule:")
    schedule_data = []
    for time_slot, program in enumerate(optimal_schedule):
        schedule_data.append({"Time Slot": f"{time_slot + 6}:00", "Program": program})

    # Display schedule in table format
    df_schedule = pd.DataFrame(schedule_data)
    st.table(df_schedule)

    st.write("Total Ratings:", fitness_function(optimal_schedule))

if __name__ == "__main__":
    main()
