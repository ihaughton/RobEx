#!/usr/bin/env python
import torch
import numpy as np
import cv2
import imgviz
import json

from RobEx import trajectory_generator
from RobEx.mapping import fc_map, occupancy
from RobEx.render import render_rays


def main():
    save = False

    with torch.no_grad():
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        epoch = 4000
        load_path = (
            ""
        )
        checkpoint_path = (
            load_path + "/checkpoints/"
            "epoch_" + str(epoch) + ".pth"
        )

        with open(load_path + '/config.json') as json_file:
            config = json.load(json_file)

        path = (
            ""
        )
        traj_scene = trajectory_generator.scene.Scene(
            path, visual=True, draw_bounds=False
        )

        dim = 256
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

        fc_occ_map = fc_map.OccupancyMap(embedding_size)
        fc_occ_map.to(device)
        checkpoint = torch.load(checkpoint_path)
        fc_occ_map.load_state_dict(checkpoint["model_state_dict"])
        fc_occ_map.eval()

        chunk_size = 700000
        alphas = occupancy.chunk_alphas(
            pc, chunk_size, fc_occ_map, n_embed_funcs)

        occ = render_rays.occupancy_activation(alphas, voxel_size)
        occ = occ.view(dim, dim, dim)
        mat_march = occ.detach().cpu().numpy()

        for slice_idx in range(dim):
            map_slice = 1. - mat_march[:, :, dim - 1 - slice_idx]
            slice_viz = imgviz.depth2rgb(map_slice, min_value=0., max_value=1.)
            cv2.imshow("slice", slice_viz)
            if save:
                cv2.imwrite("slice_" + str(slice_idx) + ".jpg", slice_viz)
            cv2.waitKey(0)


if __name__ == "__main__":
    scenes = main()
