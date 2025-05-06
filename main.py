import streamlit as st
import pandas as pd

from utils import haversine, calculate_transportation_cost, calculate_warehousing_cost
from optimization import kmeans_weighted_optimization, choose_best_k
from visualization import plot_network_map, plot_summary_charts

st.set_page_config(page_title="Warehouse Optimizer", layout="wide", page_icon="🏭")

st.title("Warehouse Location Optimizer")

st.markdown("Upload a CSV with **Latitude**, **Longitude**, and **Sales** columns.")

uploaded = st.file_uploader("Upload store list", type=['csv'])

if uploaded:
    stores = pd.read_csv(uploaded)
    required_cols = {'Latitude', 'Longitude', 'Sales'}
    if not required_cols.issubset(stores.columns):
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
        st.stop()

    with st.sidebar:
        st.header("Parameters")
        auto_k = st.checkbox("Let app choose optimal number of warehouses", value=True)
        if auto_k:
            k_range = st.slider("k range", 1, 10, (2,5))
        else:
            n_warehouses = st.number_input("Number of warehouses", min_value=1, max_value=10, value=3, step=1)
        sqft_per_lb = st.number_input("Warehouse sqft per lb shipped", value=0.02)
        cost_per_sqft = st.number_input("Warehousing cost $/sqft", value=6.0)

    if auto_k:
        k, centers, assigned = choose_best_k(stores, range(k_range[0], k_range[1]+1))
        st.success(f"Optimal number of warehouses: {k}")
    else:
        centers, assigned = kmeans_weighted_optimization(stores, n_warehouses)

    # display map
    plot_network_map(assigned, centers)
    plot_summary_charts(assigned)

    # cost summary
    st.header("Cost Estimates")
    total_weight = assigned['Sales'].sum()
    warehouse_cost = calculate_warehousing_cost(total_weight, sqft_per_lb, cost_per_sqft)
    st.metric("Annual Warehousing Cost", f"${warehouse_cost:,.0f}")
