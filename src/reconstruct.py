import numpy as np

#Use method of least squares to solve for 3 unknows (3D normal vector and albedo scaled with it)

def reconstruct(renders, lights):
    num_lights, height, width = renders.shape
    intensities = renders.reshape(num_lights, -1)
    g, _, _, _ = np.linalg.lstsq(lights, intensities, rcond=None)
    g = g.T.reshape(height, width, 3)
    magnitude = np.linalg.norm(g, axis=-1)
    normals = g / (magnitude[..., None] + 1e-8)
    return normals, magnitude


def angular_error_deg(normals_a, normals_b):
    dot = np.sum(normals_a * normals_b, axis=-1) #dot product gives cos(angl btwn 2 normal)
    dot = np.clip(dot, -1.0, 1.0) #to provent floating point error for arccos function
    return np.degrees(np.arccos(dot)).mean()