#!/usr/bin/env python
import torch
import trimesh
import numpy as np
import json
from torchvision import transforms

from RobEx import trajectory_generator
from RobEx.mapping import fc_map, occupancy
from RobEx import visualisation, geometry
from RobEx.datasets.replica_scene import dataset, image_transforms
from RobEx.render import render_rays


def main():
    with torch.no_grad():

        # Set paths -------------------------------------------------------
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        epoch = 2000
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

        traj_file = (
            ""
        )
        ims_file = (
            ""
        )

        # Load map -------------------------------------------------------
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
        checkpoint = torch.load(checkpoint_path)
        fc_occ_map.load_state_dict(checkpoint["model_state_dict"])
        fc_occ_map.eval()

        indices = config["render"]["im_indices"]
        B_cam = len(indices)

        ims = []
        depths = []
        Ts = []

        # Load image dataset  ----------------------------------------------
        scale = 1.0 / (65535.0 * 0.1)
        rgb_transform = transforms.Compose([image_transforms.BGRtoRGB()])
        depth_transform = transforms.Compose(
            [image_transforms.DepthScale(scale)])

        scene_dataset = dataset.ReplicaSceneCache(
            traj_file,
            ims_file,
            rgb_transform=rgb_transform,
            depth_transform=depth_transform,
        )

        for idx in indices:
            sample = scene_dataset[idx]
            ims.append(sample["image"])
            depths.append(sample["depth"])
            Ts.append(sample["T"])

        im_batch = np.array(ims)
        depth_batch_np = np.array(depths)
        T_WC_batch_np = np.array(Ts)
        depth_batch = torch.from_numpy(depth_batch_np).float().to(device)

        H = depth_batch[0].shape[0]
        W = depth_batch[0].shape[1]

        fx, fy = (
            W / 2.0,
            W / 2.0,
        )
        cx, cy = (W - 1.0) / 2.0, (H - 1.0) / 2.0

        # Get map mesh  ----------------------------------------------
        scene = trimesh.Scene()
        scene = traj_scene.scene
        chunk_size = 700000
        alphas = occupancy.chunk_alphas(
            pc, chunk_size, fc_occ_map, n_embed_funcs)

        occ = render_rays.occupancy_activation(alphas, voxel_size)
        occ = occ.view(dim, dim, dim)
        mat_march = occ.detach().cpu().numpy()

        level = 0.5
        mesh = occupancy.marching_cubes(mat_march, level)

        # Transform to [-1, 1] range
        mesh.apply_translation([-0.5, -0.5, -0.5])
        mesh.apply_scale(2)

        # Transform to scene coordinates
        mesh.apply_scale(scene_scale_np)
        mesh.apply_translation(scene_center_np)

        mesh.apply_translation([0, 0, -3])
        scene.add_geometry(mesh)

        # GT depth point cloud -----------------------------------------
        for batch_i in range(B_cam):
            T_WC = T_WC_batch_np[batch_i]
            pcd = geometry.transform.pointcloud_from_depth(
                depth_batch_np[batch_i], fx, fy, cx, cy
            )
            pc_flat = pcd.reshape(-1, 3)
            col = im_batch[batch_i].reshape(-1, 3)
            pc_tri = trimesh.PointCloud(vertices=pc_flat, colors=col)
            translate = np.eye(4)
            translate[:3, 3] = [0, 0, 3]
            scene.add_geometry(pc_tri, transform=translate @ T_WC)

            camera = trimesh.scene.Camera(
                fov=scene.camera.fov,
                resolution=scene.camera.resolution
            )
            marker = visualisation.draw.draw_camera(camera, T_WC)
            scene.add_geometry(marker)

        display = {"scene": scene}
        yield display


if __name__ == "__main__":
    scenes = main()
    tiling = None

    visualisation.display_scenes.display_scenes(
        scenes, height=int(480 * 2), width=int(640 * 2)
    )
