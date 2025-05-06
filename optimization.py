import numpy as np
from sklearn.cluster import KMeans

def _weighted_coords(df):
    return df[['Longitude', 'Latitude']].values, df['Sales'].values

def kmeans_weighted_optimization(stores, n_warehouses, random_state=42):
    """Return cluster centers using sales as sample weight."""
    coords, weights = _weighted_coords(stores)
    kmeans = KMeans(n_clusters=n_warehouses, n_init='auto', random_state=random_state)
    kmeans.fit(coords, sample_weight=weights)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    stores = stores.copy()
    stores['Warehouse'] = labels
    return centers, stores

def choose_best_k(stores, k_range=range(1, 8)):
    """Pick k that minimises total distance weighted by sales."""
    coords, weights = _weighted_coords(stores)
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(coords, sample_weight=weights)
        dist = 0.0
        for i, center in enumerate(km.cluster_centers_):
            mask = km.labels_ == i
            dist += (np.linalg.norm(coords[mask] - center, axis=1) * weights[mask]).sum()
        if best is None or dist < best[1]:
            best = (k, dist, km.cluster_centers_, km.labels_)
    k, _, centers, labels = best
    stores = stores.copy()
    stores['Warehouse'] = labels
    return k, centers, stores
