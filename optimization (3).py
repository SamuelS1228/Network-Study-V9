import streamlit as st
import pandas as pd
import numpy as np
import random
import math

from utils import (haversine, is_in_continental_us, calculate_transportation_cost, 
                  calculate_warehousing_cost, normalize_locations, CONTINENTAL_US)

# Custom k-means clustering algorithm
def custom_kmeans(data, n_clusters, max_iterations=100, seed=None):
    """
    Custom implementation of K-means clustering algorithm
    
    Parameters:
    - data: NumPy array of shape (n_samples, n_features)
    - n_clusters: int, number of clusters to form
    - max_iterations: int, maximum number of iterations
    - seed: int, random seed for reproducibility
    
    Returns:
    - centroids: NumPy array of shape (n_clusters, n_features)
    - labels: NumPy array of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n_samples, n_features = data.shape
    
    # Initialize centroids using k-means++ method
    # First centroid is chosen randomly
    centroids = np.zeros((n_clusters, n_features))
    centroids[0] = data[np.random.randint(n_samples)]
    
    # Choose remaining centroids
    for i in range(1, n_clusters):
        # Compute distances from points to the existing centroids
        dist_sq = np.array([min([np.sum((x-c)**2) for c in centroids[:i]]) for x in data])
        
        # Choose next centroid with probability proportional to dist_sq
        probs = dist_sq / dist_sq.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.searchsorted(cumprobs, r)
        centroids[i] = data[ind]
    
    # Main K-means loop
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        distances = np.zeros((n_samples, n_clusters))
        for i, centroid in enumerate(centroids):
            # Calculate squared Euclidean distance
            distances[:, i] = np.sum((data - centroid)**2, axis=1)
        
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(n_clusters):
            if np.sum(labels == i) > 0:  # Avoid empty clusters
                new_centroids[i] = np.mean(data[labels == i], axis=0)
            else:
                # If empty cluster, reinitialize with a random point
                new_centroids[i] = data[np.random.randint(n_samples)]
        
        # Check convergence
        if np.allclose(centroids, new_centroids, rtol=1e-5):
            break
            
        centroids = new_centroids
    
    return centroids, labels

# KMeans-based warehouse location optimization
def kmeans_weighted_optimization(stores_df, n_warehouses, max_iterations=50, 
                                cost_per_pound_mile=0.001, sq_ft_per_pound=0.1, 
                                cost_per_sq_ft=10.0, seed=None):
    # Set the seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Normalize latitude and longitude for better clustering
    locations = normalize_locations(stores_df)
    
    # Add demand weights
    weights = stores_df['yearly_demand_lbs'].values
    
    # Create a weighted dataset by repeating points based on their weights
    # Normalize weights first to avoid creating too many points
    norm_weights = np.ceil(weights / weights.min()).astype(int)
    
    # Cap the number of repetitions to keep computation manageable
    max_repetitions = 50
    if max(norm_weights) > max_repetitions:
        scaling_factor = max_repetitions / max(norm_weights)
        norm_weights = np.ceil(norm_weights * scaling_factor).astype(int)
    
    # Create weighted locations array
    weighted_locations = []
    for i, rep in enumerate(norm_weights):
        weighted_locations.extend([locations[i]] * rep)
    weighted_locations = np.array(weighted_locations)
    
    # Show progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Initializing K-Means clustering...")
    
    # Run our custom K-means algorithm
    best_centroids = None
    best_inertia = float('inf')
    
    # Run multiple times with different initializations to find the best solution
    n_init = 5  # Number of times to run kmeans with different initializations
    for i in range(n_init):
        progress = int((i + 1) / n_init * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Running K-Means optimization: {i+1}/{n_init}")
        
        # Run custom K-means
        run_seed = seed + i if seed is not None else None
        centroids, labels = custom_kmeans(weighted_locations, n_warehouses, max_iterations, seed=run_seed)
        
        # Calculate inertia (sum of squared distances to closest centroid)
        inertia = 0
        for j, point in enumerate(weighted_locations):
            closest_centroid = centroids[labels[j]]
            inertia += np.sum((point - closest_centroid) ** 2)
        
        # Keep track of the best solution
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
    
    # Denormalize to get actual latitude and longitude
    warehouse_locs = []
    for i, center in enumerate(best_centroids):
        # Denormalize coordinates
        lat = center[0] * (stores_df['latitude'].max() - stores_df['latitude'].min()) + stores_df['latitude'].min()
        lon = center[1] * (stores_df['longitude'].max() - stores_df['longitude'].min()) + stores_df['longitude'].min()
        
        # Ensure the warehouse is within continental US
        if not is_in_continental_us(lat, lon):
            # If not, find the closest valid point
            valid_points = stores_df[['latitude', 'longitude']].values
            distances = np.sqrt((valid_points[:, 0] - lat)**2 + (valid_points[:, 1] - lon)**2)
            closest_idx = np.argmin(distances)
            lat = stores_df.iloc[closest_idx]['latitude']
            lon = stores_df.iloc[closest_idx]['longitude']
        
        warehouse_locs.append({
            "warehouse_id": f"WH_{i+1}",
            "latitude": lat,
            "longitude": lon
        })
    
    # Create DataFrame for warehouse locations
    warehouses_df = pd.DataFrame(warehouse_locs)
    
    # Assign each store to the closest warehouse
    assignments = []
    total_transportation_cost = 0
    
    for _, store in stores_df.iterrows():
        min_cost = float('inf')
        assigned_wh = None
        min_distance = 0
        
        for _, wh in warehouses_df.iterrows():
            distance = haversine(store["longitude"], store["latitude"], 
                                wh["longitude"], wh["latitude"])
            cost = calculate_transportation_cost(distance, store["yearly_demand_lbs"], cost_per_pound_mile)
            
            if cost < min_cost:
                min_cost = cost
                assigned_wh = wh["warehouse_id"]
                min_distance = distance
        
        assignments.append({
            "store_id": store["store_id"],
            "warehouse_id": assigned_wh,
            "distance_miles": min_distance,
            "transportation_cost": min_cost
        })
        
        total_transportation_cost += min_cost
    
    assignments_df = pd.DataFrame(assignments)
    
    # Calculate warehousing costs
    warehouse_volumes = assignments_df.groupby("warehouse_id").apply(
        lambda x: stores_df.loc[stores_df['store_id'].isin(x['store_id']), 'yearly_demand_lbs'].sum()
    )
    
    warehousing_costs = {}
    total_warehousing_cost = 0
    
    for wh_id, volume in warehouse_volumes.items():
        wh_cost = calculate_warehousing_cost(volume, sq_ft_per_pound, cost_per_sq_ft)
        warehousing_costs[wh_id] = wh_cost
        total_warehousing_cost += wh_cost
    
    # Add warehousing costs to warehouse dataframe
    for wh_id, cost in warehousing_costs.items():
        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "warehousing_cost"] = cost
    
    # Add warehousing volumes
    for wh_id, volume in warehouse_volumes.items():
        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "volume_lbs"] = volume
    
    progress_bar.progress(100)
    progress_text.text("Optimization completed successfully!")
    
    total_cost = total_transportation_cost + total_warehousing_cost
    
    return warehouses_df, assignments_df, total_transportation_cost, total_warehousing_cost, total_cost

# Enhanced iterative optimization with better initialization
def enhanced_iterative_optimization(stores_df, n_warehouses, max_iterations=50, 
                                   cost_per_pound_mile=0.001, sq_ft_per_pound=0.1, 
                                   cost_per_sq_ft=10.0, seed=None):
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize warehouse locations using k-means++ strategy for better starting points
    # This helps ensure we start with a good initial distribution
    locations = stores_df[['latitude', 'longitude']].values
    
    # Initialize first warehouse at the point with highest demand
    highest_demand_idx = stores_df['yearly_demand_lbs'].argmax()
    centroids = [locations[highest_demand_idx]]
    
    # Initialize remaining warehouses with k-means++ logic
    for _ in range(1, n_warehouses):
        # Calculate distances from each point to nearest existing centroid
        dist_sq = np.array([min([np.sum((x-c)**2) for c in centroids]) for x in locations])
        
        # Select next centroid with probability proportional to dist_sq
        probs = dist_sq / dist_sq.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.searchsorted(cumprobs, r)
        centroids.append(locations[ind])
    
    # Create initial warehouse dataframe
    warehouses = []
    for i, centroid in enumerate(centroids):
        warehouses.append({
            "warehouse_id": f"WH_{i+1}",
            "latitude": centroid[0],
            "longitude": centroid[1]
        })
    
    warehouses_df = pd.DataFrame(warehouses)
    
    # Show initial progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Iterative optimization
    prev_cost = float('inf')
    for iteration in range(max_iterations):
        # Update progress
        progress = int((iteration + 1) / max_iterations * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Optimizing: Iteration {iteration + 1}/{max_iterations}")
        
        # Assign each store to closest warehouse
        assignments = []
        total_transportation_cost = 0
        
        for _, store in stores_df.iterrows():
            min_cost = float('inf')
            assigned_wh = None
            min_distance = 0
            
            for _, wh in warehouses_df.iterrows():
                distance = haversine(store["longitude"], store["latitude"], 
                                    wh["longitude"], wh["latitude"])
                cost = calculate_transportation_cost(distance, store["yearly_demand_lbs"], cost_per_pound_mile)
                
                if cost < min_cost:
                    min_cost = cost
                    assigned_wh = wh["warehouse_id"]
                    min_distance = distance
            
            assignments.append({
                "store_id": store["store_id"],
                "warehouse_id": assigned_wh,
                "distance_miles": min_distance,
                "transportation_cost": min_cost
            })
            
            total_transportation_cost += min_cost
        
        assignments_df = pd.DataFrame(assignments)
        
        # Calculate warehousing costs
        warehouse_volumes = assignments_df.groupby("warehouse_id").apply(
            lambda x: stores_df.loc[stores_df['store_id'].isin(x['store_id']), 'yearly_demand_lbs'].sum()
        )
        
        warehousing_costs = {}
        total_warehousing_cost = 0
        
        for wh_id, volume in warehouse_volumes.items():
            wh_cost = calculate_warehousing_cost(volume, sq_ft_per_pound, cost_per_sq_ft)
            warehousing_costs[wh_id] = wh_cost
            total_warehousing_cost += wh_cost
        
        total_cost = total_transportation_cost + total_warehousing_cost
        
        # Check convergence
        if abs(prev_cost - total_cost) < 0.1:  # Tighter convergence criterion
            progress_bar.progress(100)
            progress_text.text(f"Optimization completed in {iteration + 1} iterations")
            break
        
        prev_cost = total_cost
        
        # Update warehouse locations to weighted centroid of assigned stores
        for _, wh in warehouses_df.iterrows():
            wh_id = wh["warehouse_id"]
            assigned_stores = assignments_df[assignments_df["warehouse_id"] == wh_id]
            
            if len(assigned_stores) > 0:
                # Get the actual store data
                store_indices = []
                for store_id in assigned_stores["store_id"]:
                    idx = stores_df[stores_df["store_id"] == store_id].index
                    if len(idx) > 0:
                        store_indices.append(idx[0])
                
                if store_indices:
                    assigned_stores_data = stores_df.iloc[store_indices]
                    
                    # Calculate weighted centroid based on demand
                    total_demand = assigned_stores_data["yearly_demand_lbs"].sum()
                    
                    if total_demand > 0:
                        weighted_lat = (assigned_stores_data["latitude"] * assigned_stores_data["yearly_demand_lbs"]).sum() / total_demand
                        weighted_lon = (assigned_stores_data["longitude"] * assigned_stores_data["yearly_demand_lbs"]).sum() / total_demand
                        
                        # Ensure the warehouse is within continental US
                        if is_in_continental_us(weighted_lat, weighted_lon):
                            warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "latitude"] = weighted_lat
                            warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "longitude"] = weighted_lon
    
    # If we've reached max iterations without convergence
    if iteration == max_iterations - 1:
        progress_bar.progress(100)
        progress_text.text(f"Optimization completed after maximum {max_iterations} iterations")
    
    # Add warehousing costs and volumes to warehouse dataframe
    for wh_id, cost in warehousing_costs.items():
        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "warehousing_cost"] = cost
    
    for wh_id, volume in warehouse_volumes.items():
        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "volume_lbs"] = volume
    
    return warehouses_df, assignments_df, total_transportation_cost, total_warehousing_cost, total_cost

# Standard iterative optimization from the original code
def standard_iterative_optimization(stores_df, n_warehouses, max_iterations=50, 
                                   cost_per_pound_mile=0.001, sq_ft_per_pound=0.1, 
                                   cost_per_sq_ft=10.0, seed=None):
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Initialize random warehouse locations within continental US boundaries
    warehouses = []
    
    while len(warehouses) < n_warehouses:
        lat = random.uniform(CONTINENTAL_US["min_lat"], CONTINENTAL_US["max_lat"])
        lon = random.uniform(CONTINENTAL_US["min_lon"], CONTINENTAL_US["max_lon"])
        if is_in_continental_us(lat, lon):
            warehouses.append({
                "warehouse_id": f"WH_{len(warehouses)+1}",
                "latitude": lat,
                "longitude": lon
            })
    
    warehouses_df = pd.DataFrame(warehouses)
    
    # Show initial progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Iterative optimization
    prev_cost = float('inf')
    for iteration in range(max_iterations):
        # Update progress
        progress = int((iteration + 1) / max_iterations * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Optimizing: Iteration {iteration + 1}/{max_iterations}")
        
        # Assign each store to closest warehouse
        assignments = []
        total_transportation_cost = 0
        
        for _, store in stores_df.iterrows():
            min_cost = float('inf')
            assigned_wh = None
            min_distance = 0
            
            for _, wh in warehouses_df.iterrows():
                distance = haversine(store["longitude"], store["latitude"], 
                                    wh["longitude"], wh["latitude"])
                cost = calculate_transportation_cost(distance, store["yearly_demand_lbs"], cost_per_pound_mile)
                
                if cost < min_cost:
                    min_cost = cost
                    assigned_wh = wh["warehouse_id"]
                    min_distance = distance
            
            assignments.append({
                "store_id": store["store_id"],
                "warehouse_id": assigned_wh,
                "distance_miles": min_distance,
                "transportation_cost": min_cost
            })
            
            total_transportation_cost += min_cost
        
        assignments_df = pd.DataFrame(assignments)
        
        # Calculate warehousing costs
        warehouse_volumes = assignments_df.groupby("warehouse_id").apply(
            lambda x: stores_df.loc[stores_df['store_id'].isin(x['store_id']), 'yearly_demand_lbs'].sum()
        )
        
        warehousing_costs = {}
        total_warehousing_cost = 0
        
        for wh_id, volume in warehouse_volumes.items():
            wh_cost = calculate_warehousing_cost(volume, sq_ft_per_pound, cost_per_sq_ft)
            warehousing_costs[wh_id] = wh_cost
            total_warehousing_cost += wh_cost
        
        total_cost = total_transportation_cost + total_warehousing_cost
        
        # Check convergence
        if abs(prev_cost - total_cost) < 1:
            progress_bar.progress(100)
            progress_text.text(f"Optimization completed in {iteration + 1} iterations")
            break
        
        prev_cost = total_cost
        
        # Update warehouse locations to center of assigned stores
        for _, wh in warehouses_df.iterrows():
            wh_id = wh["warehouse_id"]
            assigned_stores = assignments_df[assignments_df["warehouse_id"] == wh_id]
            
            if len(assigned_stores) > 0:
                # Get the actual store data
                store_indices = []
                for store_id in assigned_stores["store_id"]:
                    idx = stores_df[stores_df["store_id"] == store_id].index
                    if len(idx) > 0:
                        store_indices.append(idx[0])
                
                if store_indices:
                    assigned_stores_data = stores_df.iloc[store_indices]
                    
                    # Calculate weighted centroid based on demand
                    total_demand = assigned_stores_data["yearly_demand_lbs"].sum()
                    
                    if total_demand > 0:
                        weighted_lat = (assigned_stores_data["latitude"] * assigned_stores_data["yearly_demand_lbs"]).sum() / total_demand
                        weighted_lon = (assigned_stores_data["longitude"] * assigned_stores_data["yearly_demand_lbs"]).sum() / total_demand
                        
                        # Ensure the warehouse is within continental US
                        if is_in_continental_us(weighted_lat, weighted_lon):
                            warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "latitude"] = weighted_lat
                            warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "longitude"] = weighted_lon
    
    # If we've reached max iterations without convergence
    if iteration == max_iterations - 1:
        progress_bar.progress(100)
        progress_text.text(f"Optimization completed after maximum {max_iterations} iterations")
    
    # Add warehousing costs and volumes to warehouse dataframe
    for wh_id, cost in warehousing_costs.items():
        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "warehousing_cost"] = cost
    
    for wh_id, volume in warehouse_volumes.items():
        warehouses_df.loc[warehouses_df["warehouse_id"] == wh_id, "volume_lbs"] = volume
    
    return warehouses_df, assignments_df, total_transportation_cost, total_warehousing_cost, total_cost
    
# Run optimization with a specified number of warehouses
def run_optimization_with_n_warehouses(n, method, stores_df, max_iterations, 
                                      cost_per_pound_mile, sq_ft_per_pound, 
                                      cost_per_sq_ft, seed_value=None):
    """Run optimization with a specific number of warehouses and return results"""
    
    if method == "KMeans-Weighted":
        return kmeans_weighted_optimization(stores_df, n, max_iterations, 
                                          cost_per_pound_mile, sq_ft_per_pound, 
                                          cost_per_sq_ft, seed=seed_value)
    elif method == "Enhanced Iterative":
        return enhanced_iterative_optimization(stores_df, n, max_iterations, 
                                             cost_per_pound_mile, sq_ft_per_pound, 
                                             cost_per_sq_ft, seed=seed_value)
    else:  # Standard Iterative
        return standard_iterative_optimization(stores_df, n, max_iterations, 
                                             cost_per_pound_mile, sq_ft_per_pound, 
                                             cost_per_sq_ft, seed=seed_value)

# Function to run open solve and find optimal number of warehouses
def run_open_solve(min_warehouses, max_warehouses, method, stores_df, max_iterations, 
                   cost_per_pound_mile, sq_ft_per_pound, cost_per_sq_ft, seed_value=None):
    """Find the optimal number of warehouses by trying different values"""
    
    # Progress indicators
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text(f"Starting open solve from {min_warehouses} to {max_warehouses} warehouses...")
    
    results = []
    
    # Try each number of warehouses
    for n in range(min_warehouses, max_warehouses + 1):
        progress = int((n - min_warehouses) / (max_warehouses - min_warehouses + 1) * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Testing {n} warehouses ({progress}% complete)")
        
        # Run optimization with n warehouses
        warehouses_df, assignments_df, transport_cost, warehouse_cost, total_cost = run_optimization_with_n_warehouses(
            n, method, stores_df, max_iterations, 
            cost_per_pound_mile, sq_ft_per_pound, cost_per_sq_ft, seed_value)
        
        # Store results
        results.append({
            "num_warehouses": n,
            "transportation_cost": transport_cost,
            "warehousing_cost": warehouse_cost,
            "total_cost": total_cost,
            "warehouses_df": warehouses_df,
            "assignments_df": assignments_df
        })
    
    progress_bar.progress(100)
    progress_text.text("Open solve completed successfully!")
    
    # Find best solution (lowest total cost)
    best_solution = min(results, key=lambda x: x["total_cost"])
    
    return best_solution, results