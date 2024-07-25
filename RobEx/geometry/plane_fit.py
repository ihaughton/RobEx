import numpy as np
from numpy.linalg import svd

def plane_fit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """

    points = np.swapaxes(points, 0, 1)
    # points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)

    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.

    return ctr, svd(M)[0][:,-1]

def hessian_fit(points):
    ##https://mathworld.wolfram.com/HessianNormalForm.html

    point, normal = plane_fit(points)
    d = -np.dot(point, normal)

    n_x = normal[0]/np.linalg.norm(normal)
    n_y = normal[1]/np.linalg.norm(normal)
    n_z = normal[2]/np.linalg.norm(normal)
    p = d/np.linalg.norm(normal)

    return np.array([n_x, n_y, n_z, p])

def point_on_plane(point, hessian_plane, tol=0.015):
    dist = np.dot(hessian_plane[:3], point) + hessian_plane[3]

    if np.abs(dist) > tol:
        return False
    return True
