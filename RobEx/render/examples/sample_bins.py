#!/usr/bin/env python
from torchvision import transforms
import numpy as np
import torch
import trimesh

from RobEx import trajectory_generator
from RobEx.datasets.replica_scene import dataset, image_transforms
from RobEx.render import render_rays
from RobEx import visualisation
from RobEx.random_gen import random

if __name__ == "__main__":
    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    traj_file = (
        ""
    )
    ims_file = (
        ""
    )

    path_scene = (
        ""
    )

    scale = 1. / (65535.0 * 0.1)
    rgb_transform = transforms.Compose([image_transforms.BGRtoRGB()])
    depth_transform = transforms.Compose([image_transforms.DepthScale(scale)])

    dataset = dataset.ReplicaSceneCache(
        traj_file, ims_file,
        rgb_transform=rgb_transform,
        depth_transform=depth_transform)


    sample1 = dataset[0]
    sample2 = dataset[10]

    im_batch = np.array([sample1['image'], sample2['image']])
    depth_batch = np.array([sample1['depth'], sample2['depth']])
    T_WC_batch_np = np.array([sample1['T'], sample2['T']])
    depth_batch = torch.from_numpy(depth_batch).float().to(device)
    T_WC_batch = torch.from_numpy(T_WC_batch_np).float().to(device)

    H = depth_batch[0].shape[0]
    W = depth_batch[0].shape[1]
    n_pix = H * W
    fx, fy = W / 2.0, W / 2.0,
    cx, cy = (W - 1.0) / 2.0, (H - 1.0) / 2.0
    B_cam = 2
    B_ray = 1000

    dirs_C = render_rays.ray_dirs_C(
        B_cam, H, W, fx, fy, cx, cy, device, depth_type='z')
    dirs_C = dirs_C.view(B_cam, -1, 3)

    indices_1, indices_2 = random.sample_pixels(B_cam, B_ray, n_pix)
    dirs_C = dirs_C[indices_1, indices_2, :].view(B_cam, B_ray, 3)
    depth_sample = depth_batch.view(B_cam, -1)
    depth_sample = depth_sample[indices_1, indices_2]
    im_sample = im_batch.reshape(B_cam, -1, 3)
    im_sample = im_sample[indices_1, indices_2, :]

    origins, dirs_W = render_rays.origin_dirs_W(T_WC_batch, dirs_C)
    origins = origins.view(-1, 3)
    dirs_W = dirs_W.view(-1, 3)
    n_rays = dirs_W.shape[0]

    max_depth = 6.23
    min_depth = 0.
    n_bins = 16
    z_vals = render_rays.stratified_bins(
        min_depth, max_depth, n_bins, n_rays, device)
    pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])

    traj_scene = trajectory_generator.scene.Scene(path_scene, visual=True)
    scene = traj_scene.scene

    pc_gt = origins + (dirs_W * depth_sample[:, None])
    pc_gt_np = pc_gt.cpu().numpy()
    pc_gt_tri = trimesh.PointCloud(vertices=pc_gt_np, colors=im_sample)
    scene.add_geometry(pc_gt_tri)

    pc_flat = pc.view(-1, 3)
    pc_flat_np = pc_flat.cpu().numpy()
    pc_tri = trimesh.PointCloud(vertices=pc_flat_np)
    scene.add_geometry(pc_tri)

    camera = trimesh.scene.Camera(fov=scene.camera.fov,
                                  resolution=scene.camera.resolution)

    for T_WC in T_WC_batch_np:
        marker = visualisation.draw.draw_camera(camera, T_WC)
        scene.add_geometry(marker)

    scene.show()
