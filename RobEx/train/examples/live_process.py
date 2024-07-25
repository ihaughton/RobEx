#!/usr/bin/env python
import torch

import cv2
import argparse
import queue
import numpy as np
import imgviz
import trimesh
import timeit

from RobEx.train import trainer
from RobEx import geometry, visualisation
from RobEx.render import render_rays
from RobEx.mapping import occupancy
from RobEx.kinect_recorder import reader


def get_scene(render_trainer, T_WC, depth, im, mesh=None):
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
            0.5,
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

    N_save = 346
    network_freq = 2
    load_folder = "tree_save/"

    extents = np.load(load_folder + "bounds_extents.npy")
    transform = np.load(load_folder + "bounds_T.npy")
    render_trainer.set_scene_bounds(extents, transform)
    load_t = f'{0:03}'
    T_fix = np.array([[0.81096399, 0.16324127, -0.56186271, -2.11478115],
                      [0.1855416, -0.982478, -0.01764382, -0.13750715],
                      [-0.55489794, -0.08994041, -0.82704232, -1.94041565],
                      [0., 0., 0., 1.]]
                     )

    for t in range(network_freq - 1, N_save):
        load_t = f'{t:03}'
        T_WC = np.load(load_folder + "T_" + load_t + ".npy")

        if t % network_freq == 0:
            chechpoint_load_file = (
                load_folder + "epoch_" + load_t + ".pth"
            )
            checkpoint = torch.load(chechpoint_load_file)
            render_trainer.fc_occ_map.load_state_dict(
                checkpoint["model_state_dict"])
            render_trainer.B_layer.load_state_dict(
                checkpoint["B_layer_state_dict"])
            mesh = None

            print(t)
            scene, mesh = get_scene(
                render_trainer, T_WC[0], None, None, mesh=mesh)
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
            marker_height = 0.3
            color = (1.0, 1.0, 1.0, 1.0)
            marker = visualisation.draw.draw_camera(
                camera, T_WC[0], color=color, marker_height=marker_height
            )
            scene.add_geometry(marker[1])

            scene.camera_transform = T_fix

            png = scene.save_image()
            file_name = 'live_meshes/render_fix_' + f'{t:03}' + '.png'
            with open(file_name, 'wb') as f:
                f.write(png)
                f.close()


if __name__ == "__main__":
    main()
