import numpy as np

def raster_point(
    point: np.ndarray,
    camera_pose: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:

    K = np.array([[fx, 0., cx, 0.],
                [0., -fy, cy, 0.],
                [0., 0., 1., 0.]])

    camera_transform = np.linalg.inv(camera_pose)
    point_in_camera = np.matmul(camera_transform, point)[0]
    p_proj = np.matmul(K, point_in_camera)

    u = int(p_proj[0]/p_proj[2])
    v = int(p_proj[1]/p_proj[2])

    return np.array([u, v])
