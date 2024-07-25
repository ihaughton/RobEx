#!/usr/bin/env python3

import numpy as np

def main():

    normals = np.array([[[1., 2., 5.], [3., 2., 2.]],
                        [[6., 5., 5.], [3., 1., 2.]]])
    normals_mag = np.apply_along_axis(np.linalg.norm, 2, normals)
    normals_mag = np.full(normals_mag.shape, 1.)/normals_mag
    print(normals_mag.shape)

    print(normals)
    for i in range(0, normals.shape[0]):
        for j in range(0, normals.shape[1]):
            normals[i,j] = normals[i,j]*normals_mag[i,j]
    print("normals: ", normals)

    normals_c = np.roll(normals, -1, axis=0)
    print("normals_c: ", normals_c)
    normals_r = np.roll(normals, -1, axis=1)
    print("normals_r: ", normals_r)

    normals_dot = np.zeros(normals.shape[:2])
    for i in range(0, normals.shape[0]):
        for j in range(0, normals.shape[1]):
            normals_dot[i,j] = np.dot(normals[i,j], normals[i,j])
    print(normals_dot)

    depth = np.array([[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.]])

    print("depth.shape: ", depth.shape)
    depth_pre = depth[0,:][None,:]
    print("depth_pre.shape: ", depth_pre.shape)
    print("depth_pre: ", depth_pre)

    depth_c = np.diff(depth, axis=0, prepend=depth_pre)
    print("depth_c.shape: ", depth_c.shape)
    print("depth_c: ", depth_c)

    depth_pre = depth[:,0][:,None]
    print("depth_pre.shape: ", depth_pre.shape)
    print("depth_pre: ", depth_pre)

    depth_r = np.diff(depth, axis=1, prepend=depth_pre)
    print("depth_r.shape: ", depth_r.shape)
    print("depth_r: ", depth_r)

if __name__ == "__main__":
    main()
