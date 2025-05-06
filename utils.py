import math

EARTH_RADIUS_MILES = 3958.8

def haversine(lon1, lat1, lon2, lat2):
    """Return great‑circle distance in miles between two (lon,lat) pairs."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return EARTH_RADIUS_MILES * 2 * math.asin(math.sqrt(a))

CONTINENTAL_US_BOUNDS = dict(min_lat=24.396308, max_lat=49.384358,
                             min_lon=-124.848974, max_lon=-66.885444)

def is_in_continental_us(lat, lon):
    b = CONTINENTAL_US_BOUNDS
    return b['min_lat'] <= lat <= b['max_lat'] and b['min_lon'] <= lon <= b['max_lon']

def calculate_transportation_cost(distance_miles, weight_lbs, cost_per_lb_mile=0.02):
    """Simple cost model: cost = distance * weight * rate"""
    return distance_miles * weight_lbs * cost_per_lb_mile

def calculate_warehousing_cost(total_weight_lbs, sqft_per_lb, cost_per_sqft):
    sqft = total_weight_lbs * sqft_per_lb
    return sqft * cost_per_sqft
