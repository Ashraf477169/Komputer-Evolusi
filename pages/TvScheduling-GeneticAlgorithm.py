import csv
import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt

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

# Path to the uploaded CSV file
file_path = '/mnt/data/program_ratings.csv'

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
    all_time_slots = list(range(6, 24))  # time slots

    ######################################### DEFINING FUNCTIONS ##########################################
    # defining fitness function
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += ratings[program][time_slot]
        return total_rating

    # initializing the population
    def initialize_pop(programs, population_size):
        population = []
        for _ in range(population_size):
            random_schedule = programs.copy()
            random.shuffle(random_schedule)
            population.append(random_schedule)
        return population

    # selection
    def select_best(population, elitism_size):
        return sorted(population, key=lambda schedule: fitness_function(schedule), reverse=True)[:elitism_size]

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
        population = initialize_pop(all_programs, population_size)
        best_fitness_over_time = []

        for generation in range(generations):
            # Elitism
            new_population = select_best(population, elitism_size)

            # Crossover and Mutation
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(population, 2)

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
            best_schedule = max(population, key=fitness_function)
            best_fitness_over_time.append(fitness_function(best_schedule))

        return best_schedule, best_fitness_over_time

    ############################################# RESULTS ###############################################
    # Run Genetic Algorithm
    best_schedule, fitness_over_time = genetic_algorithm(GEN, POP, co_r, mut_r, EL_S)

    # Display results
    st.write("\n### Final Optimal Schedule:")
    schedule_data = []
    for time_slot, program in enumerate(best_schedule):
        schedule_data.append({"Time Slot": f"{all_time_slots[time_slot]:02d}:00", "Program": program})

    # Display schedule in table format
    df_schedule = pd.DataFrame(schedule_data)
    st.table(df_schedule)

    st.write("### Total Ratings:", fitness_function(best_schedule))

    # Plot fitness over generations
    st.write("### Fitness Progress Over Generations")
    fig, ax = plt.subplots()
    ax.plot(range(1, GEN + 1), fitness_over_time, marker='o', color='b', label='Fitness')
    ax.set_title("Fitness Over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
