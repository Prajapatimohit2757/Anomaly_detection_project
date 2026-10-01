

"""
SkyGuard AI — central configuration
All paths and constants live here so every other file just does:
    from src.config import RAW_DATA_PATH, CITY_COORDS, ...
"""

# ---------------------------------------------------------------------------
# Paths (relative to project root, e.g. C:\Users\hp\sih_2026)
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "India_MSOC_8_17.csv"
CLEAN_DATA_PATH = "data/processed/skyguard_clean.csv"
FEATURED_DATA_PATH = "data/processed/skyguard_full_featured.csv"

MODEL_PATH = "models/skyguard_rf_model.pkl"
FEATURES_PATH = "models/feature_cols_v4.pkl"
EXPLAINER_PATH = "models/skyguard_shap_explainer.pkl"
NEIGHBORS_PATH = "models/nearest_neighbors.pkl"
STATION_COORDS_PATH = "models/station_coords.pkl"

# ---------------------------------------------------------------------------
# Model / feature constants
# ---------------------------------------------------------------------------
VALUE_COLS = [
    "temp_9am", "temp_3pm", "temp_min", "temp_max",
    "humidity_9am", "humidity_3pm", "pressure_9am", "pressure_3pm",
]

THRESHOLD = 0.524          # best-F1 decision threshold, chosen during evaluation
K_NEIGHBORS = 3             # number of nearest stations used for spatial comparison
CONTAMINATION = 0.123       # matches the injected anomaly rate

# ---------------------------------------------------------------------------
# City coordinates (lat, lon) — used for spatial (nearest-neighbor) features
# ---------------------------------------------------------------------------
CITY_COORDS = {
    'Agra': (27.18, 78.02), 'Ahmedabad': (23.03, 72.58), 'Amritsar': (31.63, 74.87),
    'Bengaluru': (12.97, 77.59), 'Bhavnagar': (21.76, 72.15), 'Bhopal': (23.26, 77.41),
    'Bhubaneswar': (20.30, 85.82), 'Chandigarh': (30.73, 76.78), 'Chennai': (13.08, 80.27),
    'Coimbatore': (11.02, 76.96), 'Cuttack': (20.46, 85.88), 'Dehradun': (30.32, 78.03),
    'Delhi': (28.70, 77.10), 'Dhanbad': (23.80, 86.43), 'Goa': (15.49, 73.83),
    'Guwahati': (26.14, 91.74), 'Gwalior': (26.22, 78.18), 'Hubballi': (15.36, 75.12),
    'Hyderabad': (17.39, 78.49), 'Indore': (22.72, 75.86), 'Jabalpur': (23.18, 79.99),
    'Jaipur': (26.91, 75.79), 'Jodhpur': (26.24, 73.02), 'Kanpur': (26.45, 80.33),
    'Kochi': (9.93, 76.27), 'Kolkata': (22.57, 88.36), 'Lucknow': (26.85, 80.95),
    'Madurai': (9.93, 78.12), 'Mumbai': (19.08, 72.88), 'Mysuru': (12.30, 76.64),
    'Nagpur': (21.15, 79.09), 'Nashik': (20.00, 73.79), 'Noida': (28.54, 77.39),
    'Patna': (25.59, 85.14), 'Pune': (18.52, 73.86), 'Raipur': (21.25, 81.63),
    'Rajkot': (22.30, 70.80), 'Ranchi': (23.34, 85.31), 'Shillong': (25.58, 91.89),
    'Shimla': (31.10, 77.17), 'Silchar': (24.83, 92.78), 'Srinagar': (34.08, 74.80),
    'Surat': (21.17, 72.83), 'Thiruvananthapuram': (8.52, 76.94), 'Tirupati': (13.63, 79.42),
    'Udaipur': (24.58, 73.68), 'Varanasi': (25.32, 82.97), 'Vijayawada': (16.51, 80.63),
    'Visakhapatnam': (17.69, 83.22),
}

# Human-readable labels for raw column names, used by explain.py
LABEL_MAP = {
    'temp_9am': 'morning temperature', 'temp_3pm': 'afternoon temperature',
    'temp_min': 'minimum temperature', 'temp_max': 'maximum temperature',
    'humidity_9am': 'morning humidity', 'humidity_3pm': 'afternoon humidity',
    'pressure_9am': 'morning pressure', 'pressure_3pm': 'afternoon pressure',
    'temp_avg': 'average temperature', 'humidity_avg': 'average humidity',
    'pressure_avg': 'average pressure',
}