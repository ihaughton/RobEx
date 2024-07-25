#!/usr/bin/env python
import torch
import trimesh
import numpy as np
import cv2
import json

from RobEx import trajectory_generator
from RobEx.mapping import fc_map, occupancy
from RobEx import visualisation
from RobEx.render import render_rays


def main():
    save = False

    with torch.no_grad():
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        load_path = (
            ""
        )
        with open(load_path + '/config.json') as json_file:
            config = json.load(json_file)

        path = (
            ""
        )
        traj_scene = trajectory_generator.scene.Scene(
            path, visual=True, draw_bounds=False
        )

        dim = 512
        n_bins = config["render"]["n_bins"]
        n_embed_funcs = config["model"]["n_embed_funcs"]
        embedding_size = n_embed_funcs * 6 + 3
        occ_range = [-1., 1.]
        range_dist = occ_range[1] - occ_range[0]

        bounds = traj_scene.bound_scene
        scene_center_np = bounds.centroid
        scene_center = torch.from_numpy(scene_center_np).float().to(device)
        scene_scale_np = bounds.extents / (range_dist * 0.9)
        scene_scale = torch.from_numpy(scene_scale_np).float().to(device)

        pc = occupancy.make_3D_grid(
            occ_range, dim, device,
            translate=scene_center, scale=scene_scale)
        pc = pc.view(-1, 3).to(device)
        voxel_size = np.mean(bounds.extents) / n_bins

        fc_occ_map = fc_map.OccupancyMapTwoStage(embedding_size)
        fc_occ_map.to(device)

        cam_T = np.array(
            [
                [-9.99976999e-01, 6.22870496e-03, -2.68425840e-03, 3.31312255e00],
                [-6.51104082e-03, -7.70737861e-01, 6.37119107e-01, 6.68949851e00],
                [1.89956737e-03, 6.37121929e-01, 7.70760688e-01, 6.65874571e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        for i in range(20, 8000):
            scene = trimesh.Scene()
            # scene = traj_scene.scene
            epoch = i * 100
            print(epoch)

            # Load map NN
            checkpoint_path = (
                load_path + "/checkpoints/"
                "epoch_" + str(epoch) + ".pth"
            )

            checkpoint = torch.load(checkpoint_path)
            fc_occ_map.load_state_dict(checkpoint["model_state_dict"])
            fc_occ_map.eval()

            chunk_size = 700000
            alphas = occupancy.chunk_alphas(
                pc, chunk_size, fc_occ_map, n_embed_funcs)

            occ = render_rays.occupancy_activation(alphas, voxel_size)
            occ = occ.view(dim, dim, dim)
            mat_march = occ.detach().cpu().numpy()

            if i < 2:
                level = 0.01
            else:
                level = 0.5

            mesh = occupancy.marching_cubes(mat_march, level)

            # Transform to [-1, 1] range
            mesh.apply_translation([-0.5, -0.5, -0.5])
            mesh.apply_scale(2)

            # Transform to scene coordinates
            mesh.apply_scale(scene_scale_np)
            mesh.apply_translation(scene_center_np)

            mesh.visual.face_colors = [160, 160, 160, 255]
            scene.add_geometry(mesh)

            if save:
                filename = f'scene_{i:02d}.jpg'
                binary = scene.save_image(resolution=(640 * 2, 480 * 2))
                decoded = cv2.imdecode(np.frombuffer(binary, np.uint8), -1)
                cv2.imwrite(filename, decoded)

            scene.camera_transform = cam_T
            display = {"scene": scene}
            yield display


if __name__ == "__main__":
    scenes = main()

    visualisation.display_scenes.display_scenes(
        scenes, height=int(480 * 2), width=int(640 * 2)
    )
