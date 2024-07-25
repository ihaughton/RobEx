#!/usr/bin/env python
import torch

import cv2
import argparse
import queue
import numpy as np
import imgviz
import trimesh
import timeit
import os

from RobEx.train import trainer
from RobEx import geometry, visualisation
from RobEx.render import render_rays
from RobEx.mapping import occupancy
from RobEx.kinect_recorder import reader


def get_scene(render_trainer, T_WC, depth, im, mesh=None, level=0.5):
    if mesh is None:
        with torch.set_grad_enabled(False):
            alphas, _ = occupancy.chunks(
                render_trainer.grid_pc,
                render_trainer.chunk_size,
                render_trainer.fc_occ_map,
                render_trainer.n_embed_funcs,
                render_trainer.B_layer,
            )

        occ = render_rays.occupancy_activation(
            alphas, render_trainer.voxel_size)
        dim = render_trainer.grid_dim
        occ = occ.view(dim, dim, dim)

        fc_occ_map = render_trainer.fc_occ_map

        occ_mesh = trainer.draw_mesh(
            occ,
            level,
            render_trainer.scene_scale_np,
            render_trainer.bounds_tranform_np,
            render_trainer.chunk_size,
            render_trainer.B_layer,
            fc_occ_map,
            render_trainer.n_embed_funcs,
        )
    else:
        occ_mesh = mesh

    scene = trimesh.Scene()
    scene.set_camera()
    scene.camera.focal = (render_trainer.fx, render_trainer.fy)
    scene.camera.resolution = (render_trainer.W, render_trainer.H)
    scene.add_geometry(occ_mesh)

    return scene, occ_mesh


def main():
    # init trainer-------------------------------------------------------------
    render_trainer = trainer.RenderTrainer(
        "cuda:0",
        "config_azure.json",
        load_path=None,
        load_epoch=None,
        do_mesh=True,
        incremental=True,
        do_track=True,
        do_color=True
    )

    N_save = 3928
    network_freq = 10
    load_folder = "live_save_3/"

    extents = np.load(load_folder + "bounds_extents.npy")
    transform = np.load(load_folder + "bounds_T.npy")
    render_trainer.set_scene_bounds(extents, transform)
    load_t = f'{0:03}'
    T_fix = np.array([[0.48048025, -0.87174734, 0.0958923, -0.31581567],
                      [-0.5470025, -0.38335262, -0.74419691, -3.84190404],
                      [0.68551223, 0.30511859, -0.66104133, -1.81915485],
                      [0., 0., 0., 1.]]
                     )

    kf_counter = 0
    for t in range(N_save-20, N_save):
        print(t)
        load_t = f'{t:03}'
        T_WC = np.load(load_folder + "T_" + load_t + ".npy")

        kf_Ts_file = load_folder + "kf_Ts_" + load_t + ".npy"
        if os.path.isfile(kf_Ts_file):
            kf_Ts = np.load(kf_Ts_file)

        cam_center = T_WC[0, :3, 3]

        if render_trainer.trajectory is None:
            render_trainer.trajectory = cam_center[None, ...]

        else:
            render_trainer.trajectory = np.concatenate(
                (render_trainer.trajectory, cam_center[None, ...]), 0)

        kf_file = load_folder + "keyframe_" + load_t + ".png"
        kf_im = cv2.imread(kf_file)

        if kf_im is not None:
            cv2.imshow("kf_im", kf_im)
            cv2.waitKey(1)
            kf_counter += 1

        if (t % network_freq == 0) and (t>3440):
            chechpoint_load_file = (
                load_folder + "epoch_" + load_t + ".pth"
            )
            checkpoint = torch.load(chechpoint_load_file)
            render_trainer.fc_occ_map.load_state_dict(
                checkpoint["model_state_dict"])
            render_trainer.B_layer.load_state_dict(
                checkpoint["B_layer_state_dict"])
            mesh = None

            if t == 0:
                level = 0.
            else:
                level = 0.5
            scene, mesh = get_scene(
                render_trainer, T_WC[0], None, None, mesh=mesh, level=level)

            scene.camera_transform = T_WC[0]
            scene.camera_transform = scene.camera_transform @ geometry.transform.to_trimesh()
            png = scene.save_image()
            file_name = 'live_meshes/render_track_' + f'{t:03}' + '.png'
            with open(file_name, 'wb') as f:
                f.write(png)
                f.close()

            camera = trimesh.scene.Camera(
                fov=scene.camera.fov, resolution=scene.camera.resolution
            )
            marker_height = 0.5
            color = (1.0, 1.0, 1.0, 1.0)
            marker = visualisation.draw.draw_camera(
                camera, T_WC[0], color=color, marker_height=marker_height
            )
            scene.add_geometry(marker[1])
            scene.camera_transform = T_fix

            for kf_i in range(kf_counter):
                camera = trimesh.scene.Camera(
                    fov=scene.camera.fov, resolution=scene.camera.resolution
                )
                marker_height = 0.2
                color = (201. / 255, 26. / 255, 9. / 255, 1.0)
                marker = visualisation.draw.draw_camera(
                    camera, kf_Ts[kf_i],
                    color=color, marker_height=marker_height
                )
                scene.add_geometry(marker[1])

            png = scene.save_image()
            file_name = 'live_meshes/render_fix_' + f'{t:03}' + '.png'
            with open(file_name, 'wb') as f:
                f.write(png)
                f.close()

            scene.show()



if __name__ == "__main__":
    main()
