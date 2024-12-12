import streamlit as st

# Set default values
DEFAULT_CO_R = 0.8
DEFAULT_MUT_R = 0.2

# Streamlit UI
def main():
    st.title("Genetic Algorithm Parameter Input")
    st.write("Use the sliders below to adjust the genetic algorithm parameters.")

    # Input for crossover rate
    co_r = st.slider(
        "Crossover Rate (CO_R)",
        min_value=0.0,
        max_value=0.95,
        value=DEFAULT_CO_R,
        step=0.01,
        help="Select the crossover rate (range: 0.0 to 0.95)."
    )

    # Input for mutation rate
    mut_r = st.slider(
        "Mutation Rate (MUT_R)",
        min_value=0.01,
        max_value=0.05,
        value=DEFAULT_MUT_R,
        step=0.01,
        help="Select the mutation rate (range: 0.01 to 0.05)."
    )

    # Display the selected values
    st.write("### Selected Parameters:")
    st.write(f"- **Crossover Rate (CO_R)**: {co_r}")
    st.write(f"- **Mutation Rate (MUT_R)**: {mut_r}")

    # Placeholder for further actions
    st.write("Use these parameters in your genetic algorithm.")

if __name__ == "__main__":
    main()
