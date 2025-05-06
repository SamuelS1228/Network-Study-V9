import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import base64

from utils import generate_colors, download_link

def display_results(df, optimized_warehouses, store_assignments, 
                   total_transportation_cost, total_warehousing_cost, 
                   total_cost, used_optimization_method, solve_type, 
                   num_warehouses, open_solve_results=None,
                   cost_per_pound_mile=0.001, sq_ft_per_pound=0.1, 
                   cost_per_sq_ft=10.0, seed_value=None):
    """Display optimization results with visualizations and metrics"""
    
    # Display metrics
    st.subheader("Optimization Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Warehouses", num_warehouses)
    
    with col2:
        st.metric("Total Cost", f"${total_cost:,.2f}")
    
    with col3:
        st.metric("Optimization Method", used_optimization_method)
    
    # Display cost breakdown
    st.subheader("Cost Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Transportation Cost", f"${total_transportation_cost:,.2f}", 
                  f"{total_transportation_cost/total_cost:.1%}")
    
    with col2:
        st.metric("Warehousing Cost", f"${total_warehousing_cost:,.2f}",
                  f"{total_warehousing_cost/total_cost:.1%}")
    
    with col3:
        avg_cost_per_store = total_cost / len(df)
        st.metric("Avg. Cost per Store", f"${avg_cost_per_store:,.2f}")
    
    with col4:
        avg_cost_per_pound = total_cost / df["yearly_demand_lbs"].sum()
        st.metric("Avg. Cost per Pound", f"${avg_cost_per_pound:.4f}")
    
    # Calculate additional metrics
    warehouse_metrics = store_assignments.groupby("warehouse_id").agg(
        num_stores=("store_id", "count"),
        total_transportation_cost=("transportation_cost", "sum"),
        avg_distance=("distance_miles", "mean"),
        max_distance=("distance_miles", "max"),
        total_demand=pd.NamedAgg(column="store_id", aggfunc=lambda x: df.loc[df['store_id'].isin(x), 'yearly_demand_lbs'].sum())
    ).reset_index()
    
    # Join with warehouse locations and add warehousing costs
    warehouse_metrics = warehouse_metrics.merge(optimized_warehouses, on="warehouse_id")
    
    # Generate colors for warehouses
    warehouse_colors = generate_colors(len(optimized_warehouses))
    warehouse_color_map = {wh: color for wh, color in zip(optimized_warehouses['warehouse_id'], warehouse_colors)}
    
    # Create a DataFrame that includes both warehouse and store info for visualization
    warehouse_data_for_map = optimized_warehouses.copy()
    warehouse_data_for_map["type"] = "warehouse"
    warehouse_data_for_map = warehouse_data_for_map.merge(
        warehouse_metrics[["warehouse_id", "num_stores", "total_transportation_cost", "total_demand"]], 
        on="warehouse_id"
    )
    
    # Add color for each warehouse
    for i, wh_id in enumerate(warehouse_data_for_map['warehouse_id']):
        warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_r'] = warehouse_colors[i][0]
        warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_g'] = warehouse_colors[i][1]
        warehouse_data_for_map.loc[warehouse_data_for_map['warehouse_id'] == wh_id, 'color_b'] = warehouse_colors[i][2]
    
    # Add store information for visualization
    store_data_for_map = df.copy()
    store_data_for_map["type"] = "store"
    
    # Merge with store assignments
    store_assignments_with_ids = store_assignments.copy()
    store_assignments_with_ids["store_idx"] = store_assignments_with_ids["store_id"].apply(
        lambda x: df.index[df["store_id"] == x].tolist()[0] if any(df["store_id"] == x) else None
    )
    
    store_data_for_map = store_data_for_map.merge(
        store_assignments[["store_id", "warehouse_id", "distance_miles", "transportation_cost"]], 
        on="store_id"
    )
    
    # Add color for each store based on its assigned warehouse
    for wh_id in warehouse_data_for_map['warehouse_id']:
        color = warehouse_color_map[wh_id]
        mask = store_data_for_map['warehouse_id'] == wh_id
        store_data_for_map.loc[mask, 'color_r'] = color[0]
        store_data_for_map.loc[mask, 'color_g'] = color[1]
        store_data_for_map.loc[mask, 'color_b'] = color[2]
    
    # Create a list of lines connecting stores to warehouses for the map
    lines = []
    for _, store in store_data_for_map.iterrows():
        warehouse = warehouse_data_for_map[warehouse_data_for_map["warehouse_id"] == store["warehouse_id"]].iloc[0]
        # Get the color from the warehouse
        color = [
            warehouse['color_r'],
            warehouse['color_g'],
            warehouse['color_b']
        ]
        
        lines.append({
            "start_lat": store["latitude"],
            "start_lon": store["longitude"],
            "end_lat": warehouse["latitude"],
            "end_lon": warehouse["longitude"],
            "store_id": store["store_id"],
            "warehouse_id": warehouse["warehouse_id"],
            "color_r": color[0],
            "color_g": color[1],
            "color_b": color[2]
        })
    
    lines_df = pd.DataFrame(lines)
    
    # Map showing stores and warehouses with enhanced warehouse representation
    st.subheader("Map Visualization")
    
    # Create layers for the map
    store_layer = pdk.Layer(
        "ScatterplotLayer",
        data=store_data_for_map,
        get_position=["longitude", "latitude"],
        get_radius=100,  # Radius for better visibility
        get_fill_color=["color_r", "color_g", "color_b", 200],  # Color based on warehouse assignment
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    # Line layer connecting stores to warehouses
    line_layer = pdk.Layer(
        "LineLayer",
        data=lines_df,
        get_source_position=["start_lon", "start_lat"],
        get_target_position=["end_lon", "end_lat"],
        get_color=["color_r", "color_g", "color_b", 150],  # Increased opacity, colors match warehouse
        get_width=2,  # Increased line width
        pickable=True,
    )
    
    # Enhanced warehouse representation with diamond shape and border
    warehouse_layer = pdk.Layer(
        "ScatterplotLayer",
        data=warehouse_data_for_map,
        get_position=["longitude", "latitude"],
        get_radius=1200,  # Very large radius for warehouses
        get_fill_color=["color_r", "color_g", "color_b", 250],  # Fill color from palette
        get_line_color=[0, 0, 0, 200],  # Black border
        get_line_width=10,  # Very thick border
        pickable=True,
        opacity=1.0,
        stroked=True,
        filled=True,
    )
    
    # Text layer for warehouse labels
    text_layer = pdk.Layer(
        "TextLayer",
        data=warehouse_data_for_map,
        get_position=["longitude", "latitude"],
        get_text="warehouse_id",
        get_size=18,
        get_color=[0, 0, 0],  # Black text
        get_angle=0,
        get_text_anchor="middle",
        get_alignment_baseline="center",
        pickable=True,
    )
    
    # Create the map with the enhanced layers
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=np.mean(df["latitude"]),
            longitude=np.mean(df["longitude"]),
            zoom=3,
            pitch=0,
        ),
        layers=[line_layer, store_layer, warehouse_layer, text_layer],
        tooltip={
            "html": "<b>ID:</b> {store_id or warehouse_id}<br><b>Type:</b> {type}<br><b>Demand:</b> {yearly_demand_lbs} lbs<br><b>Cost:</b> ${transportation_cost or warehousing_cost}",
            "style": {"background": "white", "color": "black", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
        },
    ))
    
    # Add a legend to explain the visualization
    st.markdown("""
    ### Map Legend
    - **Large Circles with Black Borders**: Warehouses (optimized locations)
    - **Small Dots**: Stores (colored by their assigned warehouse)
    - **Lines**: Connections between stores and their assigned warehouses
    """)
    
    # Show warehouse details
    st.subheader("Warehouse Details")
    
    # Add advanced metrics
    warehouse_metrics_display = warehouse_metrics.copy()
    warehouse_metrics_display["warehousing_cost"] = optimized_warehouses["warehousing_cost"].values
    warehouse_metrics_display["total_cost"] = warehouse_metrics_display["total_transportation_cost"] + warehouse_metrics_display["warehousing_cost"]
    warehouse_metrics_display["avg_cost_per_store"] = warehouse_metrics_display["total_cost"] / warehouse_metrics_display["num_stores"]
    warehouse_metrics_display["cost_per_demand"] = warehouse_metrics_display["total_cost"] / warehouse_metrics_display["total_demand"]
    warehouse_metrics_display["warehousing_pct"] = warehouse_metrics_display["warehousing_cost"] / warehouse_metrics_display["total_cost"] * 100
    
    # Format metrics for display
    metrics_to_format = [
        "total_transportation_cost", "warehousing_cost", "total_cost", "avg_cost_per_store"
    ]
    for col in metrics_to_format:
        warehouse_metrics_display[col] = warehouse_metrics_display[col].map("${:,.2f}".format)
    
    warehouse_metrics_display["cost_per_demand"] = warehouse_metrics_display["cost_per_demand"].map("${:,.5f}".format)
    warehouse_metrics_display["total_demand"] = warehouse_metrics_display["total_demand"].map("{:,.0f} lbs".format)
    warehouse_metrics_display["avg_distance"] = warehouse_metrics_display["avg_distance"].map("{:,.1f} miles".format)
    warehouse_metrics_display["max_distance"] = warehouse_metrics_display["max_distance"].map("{:,.1f} miles".format)
    warehouse_metrics_display["warehousing_pct"] = warehouse_metrics_display["warehousing_pct"].map("{:,.1f}%".format)
    
    st.dataframe(warehouse_metrics_display)
    
    # Create expanded visualization section
    st.subheader("Performance Metrics")
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stores per Warehouse")
        # Create a DataFrame for the bar chart
        chart_data1 = warehouse_metrics[["warehouse_id", "num_stores"]].set_index("warehouse_id")
        st.bar_chart(chart_data1)
    
    with col2:
        st.subheader("Total Cost per Warehouse")
        # Calculate total cost per warehouse
        chart_data2 = pd.DataFrame({
            'warehouse_id': warehouse_metrics['warehouse_id'],
            'Transportation': warehouse_metrics['total_transportation_cost'],
            'Warehousing': optimized_warehouses['warehousing_cost'].values
        }).set_index('warehouse_id')
        st.bar_chart(chart_data2)
    
    # Additional metrics - Demand distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demand per Warehouse")
        # Create a DataFrame for the bar chart
        chart_data3 = warehouse_metrics[["warehouse_id", "total_demand"]].set_index("warehouse_id")
        st.bar_chart(chart_data3)
    
    with col2:
        st.subheader("Average Distance per Warehouse")
        # Create a DataFrame for the bar chart
        chart_data4 = warehouse_metrics[["warehouse_id", "avg_distance"]].set_index("warehouse_id")
        st.bar_chart(chart_data4)
    
    # Open Solve Results Chart (if available)
    if open_solve_results is not None:
        st.subheader("Open Solve Results")
        
        # Create a dataframe for the open solve results
        open_solve_df = pd.DataFrame([
            {
                "num_warehouses": result["num_warehouses"],
                "transportation_cost": result["transportation_cost"],
                "warehousing_cost": result["warehousing_cost"],
                "total_cost": result["total_cost"]
            }
            for result in open_solve_results
        ])
        
        # Create a chart showing the costs for different numbers of warehouses
        cost_chart_data = pd.DataFrame({
            'num_warehouses': open_solve_df['num_warehouses'],
            'Transportation': open_solve_df['transportation_cost'],
            'Warehousing': open_solve_df['warehousing_cost'],
            'Total': open_solve_df['total_cost']
        }).set_index('num_warehouses')
        
        st.line_chart(cost_chart_data)
        
        # Explanation
        st.markdown(f"""
        ### Open Solve Analysis
        
        The optimal number of warehouses was found to be **{num_warehouses}** with a total cost of **${total_cost:,.2f}**.
        
        - As the number of warehouses increases, transportation costs generally decrease (shorter distances)
        - However, warehousing costs increase with more warehouses (more fixed costs)
        - The optimal solution balances these two cost factors
        """)
        
        # Show the detailed results in a table
        with st.expander("View detailed open solve results"):
            # Format the costs for display
            display_df = open_solve_df.copy()
            for col in ['transportation_cost', 'warehousing_cost', 'total_cost']:
                display_df[col] = display_df[col].map("${:,.2f}".format)
            
            st.dataframe(display_df)
    
    # Store details section
    st.subheader("Store Assignments")
    
    # Store details with merge to show warehouse data
    store_details = store_data_for_map.merge(
        optimized_warehouses[["warehouse_id", "latitude", "longitude"]], 
        on="warehouse_id",
        suffixes=("_store", "_warehouse")
    )
    
    # Add formatted distance and cost columns
    store_details["distance_miles_formatted"] = store_details["distance_miles"].map("{:,.1f} miles".format)
    store_details["transportation_cost_formatted"] = store_details["transportation_cost"].map("${:,.2f}".format)
    
    # Display interactive table
    st.dataframe(store_details)
    
    # Download results section
    st.subheader("Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(download_link(warehouse_metrics_display, "optimized_warehouses.csv", "Download Warehouse Data"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(download_link(store_details, "store_assignments.csv", "Download Store Assignments"), unsafe_allow_html=True)
    
    with col3:
        # Export full solution details
        full_solution = {
            "optimization_method": used_optimization_method,
            "solve_type": solve_type,
            "num_warehouses": num_warehouses,
            "cost_per_pound_mile": cost_per_pound_mile,
            "sq_ft_per_pound": sq_ft_per_pound,
            "cost_per_sq_ft": cost_per_sq_ft,
            "total_transportation_cost": total_transportation_cost,
            "total_warehousing_cost": total_warehousing_cost,
            "total_cost": total_cost,
            "seed_value": seed_value if seed_value is not None else "None"
        }
        full_solution_df = pd.DataFrame([full_solution])
        st.markdown(download_link(full_solution_df, "optimization_solution.csv", "Download Solution Details"), unsafe_allow_html=True)
    
    # Footer with app info
    st.markdown("---")
    st.markdown("© 2025 Warehouse Location Optimizer - Powered by Streamlit")
    
    # Add information about the optimization methods
    with st.expander("About the Optimization Methods"):
        st.markdown("""
        ### Optimization Methods
        
        This app offers three different optimization methods:
        
        1. **KMeans-Weighted**: Uses K-means clustering with demand weighting to find optimal warehouse locations. This method provides the most consistent results and is the recommended approach for most scenarios. It leverages custom K-means with multiple initializations to find the global optimum.
        
        2. **Enhanced Iterative**: An improved version of the iterative approach that uses k-means++ style initialization for better starting points. This helps avoid poor local optima and generally converges faster than the standard method.
        
        3. **Standard Iterative**: The original approach with random initialization. This can sometimes get stuck in local optima, leading to different results on different runs.
        
        For consistent results, use the "KMeans-Weighted" method with the "Use fixed seed" option enabled.
        """)
        
    with st.expander("About the Cost Model"):
        st.markdown("""
        ### Cost Model
        
        This app uses a comprehensive cost model that considers both transportation and warehousing costs:
        
        1. **Transportation Costs**: Calculated as `distance (miles) × demand (lbs) × rate ($ per pound-mile)`. This represents the cost to transport goods from warehouses to stores.
        
        2. **Warehousing Costs**: Calculated as `demand (lbs) × space required (sq ft per lb) × cost ($ per sq ft)`. This represents the fixed and variable costs of operating warehouses.
        
        The **Open Solve** feature automatically determines the optimal number of warehouses by balancing these two costs:
        - More warehouses generally reduce transportation costs (shorter distances)
        - More warehouses increase ware