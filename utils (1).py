import math
import random
import numpy as np
import pandas as pd
import base64

# Continental US boundaries
CONTINENTAL_US = {
    "min_lat": 24.396308,  # Southern tip of Florida
    "max_lat": 49.384358,  # Northern border with Canada
    "min_lon": -124.848974,  # Western coast
    "max_lon": -66.885444   # Eastern coast
}

# Function to calculate distance between two points using Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 3956  # Radius of earth in miles
    return c * r

# Function to check if point is within continental US
def is_in_continental_us(lat, lon):
    return (CONTINENTAL_US["min_lat"] <= lat <= CONTINENTAL_US["max_lat"] and 
            CONTINENTAL_US["min_lon"] <= lon <= CONTINENTAL_US["max_lon"])

# Function to calculate transportation cost
def calculate_transportation_cost(distance, weight, rate):
    return distance * weight * rate

# Function to calculate warehousing cost
def calculate_warehousing_cost(yearly_demand, sq_ft_per_pound, cost_per_sq_ft):
    warehouse_size = yearly_demand * sq_ft_per_pound
    return warehouse_size * cost_per_sq_ft

# Function to generate example data with consistent seed
def generate_example_data(num_stores=100, seed=42):
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate random points within continental US
    data = []
    for i in range(num_stores):
        lat = random.uniform(CONTINENTAL_US["min_lat"], CONTINENTAL_US["max_lat"])
        lon = random.uniform(CONTINENTAL_US["min_lon"], CONTINENTAL_US["max_lon"])
        # Generate random yearly demand between 10,000 and 500,000 pounds
        yearly_demand = round(random.uniform(10000, 500000))
        data.append({"store_id": f"Store_{i+1}", "latitude": lat, "longitude": lon, "yearly_demand_lbs": yearly_demand})
    
    return pd.DataFrame(data)

# Function to download dataframe as CSV
def download_link(dataframe, filename, link_text):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to generate distinct colors for warehouses
def generate_colors(n):
    """Generate n distinct colors"""
    colors = []
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate colors using HSV color space for better distinction
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB (simplified version)
        h = hue * 6
        c = 255
        x = 255 * (1 - abs(h % 2 - 1))
        
        if h < 1:
            rgb = [c, x, 0]
        elif h < 2:
            rgb = [x, c, 0]
        elif h < 3:
            rgb = [0, c, x]
        elif h < 4:
            rgb = [0, x, c]
        elif h < 5:
            rgb = [x, 0, c]
        else:
            rgb = [c, 0, x]
            
        colors.append(rgb)
    return colors

# Function to normalize data for clustering
def normalize_locations(df):
    # Create a copy to avoid modifying the original dataframe
    locations = df[['latitude', 'longitude']].copy()
    
    # Min-max scaling for lat/lon to handle the Earth's curvature better
    for col in ['latitude', 'longitude']:
        min_val = locations[col].min()
        max_val = locations[col].max()
        locations[col] = (locations[col] - min_val) / (max_val - min_val)
    
    return locations.values  # Return a numpy array
