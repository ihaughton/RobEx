#!/usr/bin/env python
from RobEx import trajectory_generator
from datetime import datetime
import os
import numpy as np
import trimesh

if __name__ == "__main__":
    save = True
    visual = True
    mode = 'random'

    path = (
        ""
    )

    if mode == 'ellipse':
        trans_eyes = [-0.5, 0.0, 0.1]
        scale_eyes = [0.5, 0.35, 0.4]
        trans_ats = [-0.5, 0.0, -0.6]
        scale_ats = [1, 1, 0.5]
        n_points = 600
    elif mode == 'random':

        trans_eyes = [-0.4, 0.0, 0.0]
        scale_eyes = [0.3, 0.6, 0.6]
        trans_ats = [0.6, 0.0, 0.0]
        scale_ats = [0.5, 1, 1]

        n_anchors = 50
        point_per_anchor = 40
        n_points = n_anchors * point_per_anchor

    traj_scene = trajectory_generator.scene.Scene(
        path, visual=visual,
        up=(0, 0, 1),
        trans_eyes=trans_eyes,
        scale_eyes=scale_eyes,
        trans_ats=trans_ats,
        scale_ats=scale_ats
    )

    rot = trimesh.transformations.rotation_matrix(-np.pi/2, (0,1,0))
    trans_z = 9.
    trans = trimesh.transformations.translation_matrix([0, 0, trans_z])
    traj_scene.scene.camera_transform = traj_scene.T_extent_to_scene @ rot @ trans

    traj_scene.generate_trajectory(n_anchors, n_points)
    traj_scene.show()
    if save:
        now = datetime.now()
        time_str = now.strftime('%m-%d-%y_%H-%M-%S')
        dir_name = "results/" + time_str + "/"
        os.makedirs(dir_name, exist_ok=True)
        traj_scene.save(dir_name + "traj.txt")
