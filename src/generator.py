import numpy as np
from scipy.ndimage import gaussian_filter

SIZE = 128
LIGHT_ELEVATION_DEG = 20.0
LIGHT_AZIMUTHS_DEG = [0.0, 90.0, 180.0, 270.0]
CAMO_TONES = [0.2, 0.4, 0.6, 0.8]
EXPOSURE_TARGET = 0.45
SENSOR_NOISE_STD = 0.01

DEFECT_SPECS = {
    "bubble": {"amplitude": 1.2, "length": (8, 14), "width_ratio": (0.8, 1.0)},
    "void": {"amplitude": -1.0, "length": (6, 11), "width_ratio": (0.8, 1.0)},
    "tunnel": {"amplitude": -0.9, "length": (18, 30), "width_ratio": (0.15, 0.3)},
    "wrinkle": {"amplitude": 0.8, "length": (20, 34), "width_ratio": (0.1, 0.2)},
}

def light_directions():
    """Unit vectors for the 4 lights: N/E/S/W azimuths at the 20C elevation angle"""
    elevation = np.radians(LIGHT_ELEVATION_DEG)
    directions = []
    for azimuth_deg in LIGHT_AZIMUTHS_DEG:
        azimuth = np.radians(azimuth_deg)
        lx = np.cos(elevation) * np.cos(azimuth)
        ly = np.cos(elevation) * np.sin(azimuth)
        lz = np.sin(elevation)
        directions.append([lx, ly, lz])
    return np.array(directions)
def make_base_height(rng):
    noise = rng.standard_normal((SIZE, SIZE))
    height = gaussian_filter(noise, sigma=6.0)
    height = height / (np.abs(height).max() + 1e-8)
    return height * 0.3


def gaussian_stamp(center_row, center_col, sigma_long, sigma_short, angle_deg):
    """An elongated Gaussian bump at a given position and orientation with peak value 1."""
    rows, cols = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    d_row = rows - center_row
    d_col = cols - center_col
    angle = np.radians(angle_deg)
    along = d_col * np.cos(angle) + d_row * np.sin(angle)
    across = -d_col * np.sin(angle) + d_row * np.cos(angle)
    stamp = np.exp(-(along**2 / (2 * sigma_long**2) + across**2 / (2 * sigma_short**2)))
    return stamp


def add_defect(height, rng, defect_type):
    spec = DEFECT_SPECS[defect_type]
    margin = 25
    center_row = rng.uniform(margin, SIZE - margin)
    center_col = rng.uniform(margin, SIZE - margin)
    sigma_long = rng.uniform(*spec["length"])
    sigma_short = sigma_long * rng.uniform(*spec["width_ratio"])
    angle_deg = rng.uniform(0, 180)

    stamp = gaussian_stamp(center_row, center_col, sigma_long, sigma_short, angle_deg)
    height = height + spec["amplitude"] * stamp
    mask = (stamp > 0.25).astype(np.float32)
    return height, mask

