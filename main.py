"""
Warehouse Location Optimizer Streamlit App
==========================================
This file is the main entrypoint for Streamlit.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import math
import random
import base64

# Import other modules
from utils import (haversine, is_in_continental_us, calculate_transportation_cost, 
                  calculate_warehousing_cost, generate_example_data, download_link, 
                  generate_colors, CONTINENTAL_US)
from optimization import (kmeans_weighted_optimization, enhanced_iterative_optimization, 
                         standard_iterative_optimization, run_optimization_with_n_warehouses, 
                         run_open_solve)
from visualization import display_results

# Set page configuration
st.set_page_config(
    page_title="Warehouse Location Optimizer",
    page_icon="🏭",
    layout="wide"
)

# Title and description
st.title("Warehouse Location Optimizer")
st.markdown("""
This app helps you determine the optimal locations for warehouses based on store demand, transportation costs, and warehousing costs.
Upload your store data with locations and demand information to get started.
""")

# Sidebar for uploading data and parameters
st.sidebar.header("Upload Store Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with store locations and demand data", type="csv")

# Option to use example data
use_example_data = st.sidebar.checkbox("Use example data instead", value=False)

# Sample data format explanation
with st.sidebar.expander("CSV Format Requirements"):
    st.write("""
    Your CSV file should include the following columns:
    - `store_id`: Unique identifier for each store
    - `latitude`: Store latitude
    - `longitude`: Store longitude
    - `yearly_demand_lbs`: Annual demand in pounds
    """)
    
    # Display sample data
    sample_df = pd.DataFrame({
        "store_id": ["Store_1", "Store_2", "Store_3"],
        "latitude": [40.7128, 34.0522, 41.8781],
        "longitude": [-74.0060, -118.2437, -87.6298],
        "yearly_demand_lbs": [250000, 175000, 320000]
    })
    
    st.dataframe(sample_df)
    st.markdown(download_link(sample_df, "sample_store_data.csv", "Download Sample CSV"), unsafe_allow_html=True)

# Optimization parameters
st.sidebar.header("Optimization Parameters")

# Add warehousing cost parameters
st.sidebar.subheader("Transportation Cost Parameters")
cost_per_pound_mile = st.sidebar.number_input("Transportation Cost Rate ($ per pound-mile)", min_value=0.0001, max_value=1.0, value=0.001, format="%.5f")

st.sidebar.subheader("Warehousing Cost Parameters")
sq_ft_per_pound = st.sidebar.number_input("Warehouse Space Required (sq ft per pound)", min_value=0.01, max_value=10.0, value=0.1, format="%.2f")
cost_per_sq_ft = st.sidebar.number_input("Warehousing Cost ($ per sq ft per year)", min_value=0.1, max_value=100.0, value=10.0, format="%.2f")

# Solve type selection
st.sidebar.subheader("Solve Type")
solve_type = st.sidebar.radio(
    "Select solve type:",
    ["Fixed Warehouses", "Open Solve (Find Optimal Number)"],
    index=0  # Default to Fixed Warehouses
)

# Only show number of warehouses slider for fixed warehouse solve
if solve_type == "Fixed Warehouses":
    num_warehouses = st.sidebar.slider("Number of Warehouses", min_value=1, max_value=20, value=3)
else:
    # For open solve, define the range to search
    min_warehouses = st.sidebar.slider("Minimum Number of Warehouses", min_value=1, max_value=10, value=1)
    max_warehouses = st.sidebar.slider("Maximum Number of Warehouses", min_value=min_warehouses + 1, max_value=20, value=min(min_warehouses + 5, 20))

max_iterations = st.sidebar.slider("Max Optimization Iterations", min_value=10, max_value=100, value=50)

# Select optimization algorithm
st.sidebar.header("Optimization Method")
optimization_method = st.sidebar.radio(
    "Select optimization method:",
    ["KMeans-Weighted", "Enhanced Iterative", "Standard Iterative"],
    index=0  # Default to KMeans-Weighted
)

# Add an option to set seed for reproducibility
use_fixed_seed = st.sidebar.checkbox("Use fixed seed for reproducibility", value=True)
if use_fixed_seed:
    seed_value = st.sidebar.number_input("Seed value", min_value=1, max_value=10000, value=42)
else:
    seed_value = None

# Main app logic
if uploaded_file is not None:
    # Load the uploaded data
    df = pd.read_csv(uploaded_file)
    data_source = "uploaded"
elif use_example_data:
    # Generate example data with seed for reproducibility
    df = generate_example_data(seed=42 if use_fixed_seed else None)
    data_source = "example"
else:
    st.info("Please upload a CSV file or use example data to get started.")
    st.stop()

# Check if required columns exist
required_cols = ["store_id", "latitude", "longitude", "yearly_demand_lbs"]
if not all(col in df.columns for col in required_cols):
    st.error(f"The data must contain these columns: {', '.join(required_cols)}")
    st.stop()

# Display the data
st.subheader("Store Data")
st.dataframe(df)

# Run optimization button
if solve_type == "Fixed Warehouses":
    run_button_text = f"Run Optimization with {num_warehouses} Warehouses"
else:
    run_button_text = f"Run Open Solve ({min_warehouses}-{max_warehouses} Warehouses)"

if st.button(run_button_text):
    if solve_type == "Fixed Warehouses":
        # Run optimization with the specified number of warehouses
        optimized_warehouses, store_assignments, total_transportation_cost, total_warehousing_cost, total_cost = run_optimization_with_n_warehouses(
            num_warehouses, optimization_method, df, max_iterations, 
            cost_per_pound_mile, sq_ft_per_pound, cost_per_sq_ft,
            seed_value if use_fixed_seed else None)
        
        # Store results in session state
        st.session_state.optimized_warehouses = optimized_warehouses
        st.session_state.store_assignments = store_assignments
        st.session_state.total_transportation_cost = total_transportation_cost
        st.session_state.total_warehousing_cost = total_warehousing_cost
        st.session_state.total_cost = total_cost
        st.session_state.optimization_complete = True
        st.session_state.optimization_method = optimization_method
        st.session_state.open_solve_results = None
        st.session_state.solve_type = "Fixed Warehouses"
        st.session_state.num_warehouses = num_warehouses
    else:
        # Run open solve to find optimal number of warehouses
        best_solution, all_results = run_open_solve(
            min_warehouses, max_warehouses, optimization_method, df, max_iterations, 
            cost_per_pound_mile, sq_ft_per_pound, cost_per_sq_ft,
            seed_value if use_fixed_seed else None)
        
        # Extract results from the best solution
        optimized_warehouses = best_solution["warehouses_df"]
        store_assignments = best_solution["assignments_df"]
        total_transportation_cost = best_solution["transportation_cost"]
        total_warehousing_cost = best_solution["warehousing_cost"]
        total_cost = best_solution["total_cost"]
        optimal_num_warehouses = best_solution["num_warehouses"]
        
        # Store results in session state
        st.session_state.optimized_warehouses = optimized_warehouses
        st.session_state.store_assignments = store_assignments
        st.session_state.total_transportation_cost = total_transportation_cost
        st.session_state.total_warehousing_cost = total_warehousing_cost
        st.session_state.total_cost = total_cost
        st.session_state.optimization_complete = True
        st.session_state.optimization_method = optimization_method
        st.session_state.open_solve_results = all_results
        st.session_state.solve_type = "Open Solve"
        st.session_state.num_warehouses = optimal_num_warehouses
else:
    if 'optimization_complete' not in st.session_state:
        st.session_state.optimization_complete = False

# Display results if optimization is complete
if st.session_state.optimization_complete:
    # Pass parameters to visualization function
    display_results(
        df=df,
        optimized_warehouses=st.session_state.optimized_warehouses,
        store_assignments=st.session_state.store_assignments,
        total_transportation_cost=st.session_state.total_transportation_cost,
        total_warehousing_cost=st.session_state.total_warehousing_cost,
        total_cost=st.session_state.total_cost,
        used_optimization_method=st.session_state.optimization_method,
        solve_type=st.session_state.solve_type,
        num_warehouses=st.session_state.num_warehouses,
        open_solve_results=st.session_state.open_solve_results,
        cost_per_pound_mile=cost_per_pound_mile,
        sq_ft_per_pound=sq_ft_per_pound,
        cost_per_sq_ft=cost_per_sq_ft,
        seed_value=seed_value if use_fixed_seed else None
    )
