import numpy as np
import torch

# from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import trimesh
import imgviz
import json
import cv2
import copy
import open3d

from RobEx.datasets.cam_recording import dataset, image_transforms
from RobEx.render import render_rays
from RobEx.random_gen import random
from RobEx.mapping import fc_map, occupancy
from RobEx import geometry, visualisation
from RobEx import trajectory_generator
from RobEx.mapping import embedding
from RobEx.label import ilabel


def get_pyramid(depth, levels, kernel_init, do_scaling=True):
    pyramid = []
    pyramid.append(depth)
    for level in range(levels - 1):
        kernel_size = kernel_init * 2 ** (level) + 1
        blur = cv2.GaussianBlur(pyramid[0], (kernel_size, kernel_size), 0)

        if do_scaling:
            binary = np.zeros(pyramid[0].shape)
            binary[pyramid[0] != 0] = 1
            scaling = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)
            scaling[scaling != 0] = 1. / scaling[scaling != 0]
            blur = blur * scaling

        pyramid.insert(0, blur)

    pyramid = np.array(pyramid)

    return pyramid


class FrameData:
    def __init__(self):
        super(FrameData, self).__init__()

        self.im_batch = None
        self.im_batch_np = None
        self.depth_batch_np = None
        self.T_WC_batch_np = None
        self.depth_batch = None
        self.T_WC_batch = None
        self.T_WC_track = None
        self.frame_avg_losses = None


class RenderTrainer(object):
    def __init__(
        self,
        device,
        config_file,
        load_path=None,
        load_epoch=None,
        do_mesh=False,
        incremental=True,
        # grid_dim=512,
        grid_dim=256,
        do_track=False,
        do_color=False,
        do_sem=False
    ):
        super(RenderTrainer, self).__init__()

        self.t_init = 0
        self.dirs_C = None
        self.loss_approx_viz = None
        self.device = device
        self.incremental = incremental
        self.do_color = do_color
        self.do_sem = do_sem
        self.last_is_keyframe = False
        self.has_initialised = False
        self.frame_id = 0
        self.frames_since_add = 0
        self.T_WC = None
        self.do_information = True
        self.do_fine = True
        self.trajectory = None
        self.trajectory_gt = None
        self.poses = None
        self.poses_gt = None
        self.track_optimiser = None
        self.gt_depth_vis = None
        self.kfs_im = None
        self.gt_im_vis = None
        self.pcs_gt_cam = None
        self.kf_indices = []
        self.grid_pc = None
        self.do_hierarchical = False
        self.active_inds = None
        self.batch_size = 0

        if load_path is not None:
            config_folder = load_path
        else:
            config_folder = "."

        with open(config_folder + "/" + config_file) as json_file:
            self.config = json.load(json_file)

        self.frames = FrameData()
        self.w_deltas = []
        self.trans_deltas = []

        self.set_params()
        self.set_cam()

        if self.do_tsdf:
            self.tsdf = open3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4. / 256,
                sdf_trunc=0.04,
                color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

        self.load_data(self.traj_file, self.ims_file, self.scale)
        self.set_directions()
        self.load_networks(do_color=do_color, do_sem=do_sem)
        if load_path is not None:
            self.load_checkpoints(load_path, load_epoch)
        self.fc_occ_map.train()

        self.do_mesh = do_mesh
        if do_mesh:
            self.grid_dim = grid_dim
            self.chunk_size = 300000
            self.occ_range = [-1.0, 1.0]
            self.range_dist = self.occ_range[1] - self.occ_range[0]

            if self.gt_scene:
                traj_scene = trajectory_generator.scene.Scene(
                    self.scene_file, visual=True, draw_bounds=False
                )

                bounds_extents = traj_scene.bound_scene_extents
                bounds_transform = traj_scene.T_extent_to_scene
                self.set_scene_bounds(bounds_extents, bounds_transform)

    def integrate_tsdf(self, im, depth, T_WC, intrinsic):
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            open3d.geometry.Image(im),
            open3d.geometry.Image(depth),
            depth_trunc=self.max_depth,
            depth_scale=1,
            convert_rgb_to_intensity=False,
        )

        T_CW = np.linalg.inv(T_WC)
        self.tsdf.integrate(
            image=rgbd,
            intrinsic=intrinsic,
            extrinsic=T_CW,
        )

    def tsdf_mesh(self, T_WC):
        scene = trimesh.Scene()
        scene.set_camera()
        scene.camera.focal = (self.fx, self.fy)
        scene.camera.resolution = (self.W, self.H)

        mesh = self.tsdf.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        vertex_colors = None
        if mesh.has_vertex_colors:
            vertex_colors = np.asarray(mesh.vertex_colors)
        mesh_tr = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
            vertex_colors=vertex_colors,
        )
        scene.add_geometry(mesh_tr)

        scene.camera_transform = T_WC
        scene.camera_transform = scene.camera_transform @ geometry.transform.to_trimesh()

        return scene

    def set_scene_bounds(self, extents, transform):
        self.bounds_tranform_np = transform
        self.bounds_tranform = (
            torch.from_numpy(self.bounds_tranform_np).float().to(self.device)
        )
        self.scene_scale_np = extents / (self.range_dist * 0.9)
        self.scene_scale = torch.from_numpy(
            self.scene_scale_np).float().to(self.device)

        self.grid_pc = None
        self.grid_pc = occupancy.make_3D_grid(
            self.occ_range,
            self.grid_dim,
            self.device,
            transform=self.bounds_tranform,
            scale=self.scene_scale,
        )
        self.grid_pc = self.grid_pc.view(-1, 3).to(self.device)

    def get_batch(self):
        return self.frames.depth_batch, self.frames.T_WC_batch, self.frames.im_batch

    def step_pyramid(self, iter):
        if self.pyramid_level < -1:
            if iter % self.iters_per_level == 0 and (iter != 0):
                self.pyramid_level += 1

    def load_data(self, traj_file, ims_file, scale):

        rgb_transform = transforms.Compose(
            [image_transforms.BGRtoRGB()])
        depth_transform = transforms.Compose(
            [image_transforms.DepthScale(scale),
             image_transforms.DepthFilter(self.max_depth)])

        if self.dataset_format == "TUM":
            dataset_class = dataset.TUMDataset
            col_ext = None
        elif self.dataset_format == "replica":
            dataset_class = dataset.CamRecording
            col_ext = ".jpg"
        elif self.dataset_format =='nyu':
            col_ext = ".ppm"
            dataset_class = dataset.PpmPgmRecording
        elif self.dataset_format == "synthetic":
            dataset_class = dataset.CamRecording
            col_ext = ".png"
        elif self.dataset_format == "server":
            dataset_class = dataset.CamRecordingServer
            col_ext = ".jpg"

        self.scene_dataset = dataset_class(
            ims_file,
            traj_file,
            rgb_transform=rgb_transform,
            depth_transform=depth_transform,
            col_ext=col_ext
        )

        if "n_random_views" in self.config["dataset"]:
            n_random_views = self.config["dataset"]["n_random_views"]
            if n_random_views > 0:
                n_dataset = len(self.scene_dataset)
                self.indices = np.random.choice(
                    np.arange(0, n_dataset),
                    size=n_random_views,
                    replace=False)

        if self.incremental is False:
            idxs = self.indices
            frame_data = self.get_data(idxs)
            self.add_data(frame_data)

            if self.do_tsdf:
                for i in range(self.batch_size):
                    self.integrate_tsdf(
                        self.frames.im_batch_np[i],
                        self.frames.depth_batch_np[i],
                        self.frames.T_WC_batch_np[i],
                        self.intrinsic_open3d)

    def expand_data(self, batch, data, replace=False):
        cat_fn = np.concatenate
        if torch.is_tensor(data):
            cat_fn = torch.cat

        if batch is None:
            batch = data

        else:
            if replace is False:
                batch = cat_fn((batch, data))
            else:
                batch[-1] = data[0]

        return batch

    def clear_keyframes(self):
        self.frames = FrameData()
        self.w_deltas = []
        self.trans_deltas = []
        self.optim_frames = self.iters_per_kf

    def clear_keyframes_vis(self):
        self.kfs_depth = []
        self.kfs_im = []

    def cut_data(self, frames_limit):
        self.frames.im_batch = self.frames.im_batch[-frames_limit:]
        self.frames.im_batch_np = self.frames.im_batch_np[-frames_limit:]
        self.frames.depth_batch_np = self.frames.depth_batch_np[-frames_limit:]
        self.frames.depth_batch = self.frames.depth_batch[-frames_limit:]
        if self.gt_traj:
            self.frames.T_WC_batch = self.frames.T_WC_batch[-frames_limit:]
            self.frames.T_WC_batch_np = self.frames.T_WC_batch_np[-frames_limit:]

        else:
            self.frames.T_WC_track = self.frames.T_WC_track[-frames_limit:]
            self.w_deltas = self.w_deltas[-(frames_limit - 1):]
            self.trans_deltas = self.trans_deltas[-(frames_limit - 1):]

        self.gt_depth_vis = self.gt_depth_vis[-(frames_limit - 1):]
        self.gt_im_vis = self.gt_im_vis[-(frames_limit - 1):]
        self.pcs_gt_cam = self.pcs_gt_cam[-(frames_limit - 1):]

        self.batch_size = self.frames.depth_batch_np.shape[0]

    def get_data(self, idxs):
        data = FrameData()
        for idx in idxs:
            sample = self.scene_dataset[idx]

            im_np = sample["image"][None, ...]
            depth_np = sample["depth"][None, ...]

            if self.gt_traj:
                T_np = sample["T"][None, ...]
                T = torch.from_numpy(T_np).float().to(self.device)

            depth = torch.from_numpy(depth_np).float().to(self.device)
            im = torch.from_numpy(im_np).float().to(self.device) / 255.

            data.im_batch = self.expand_data(data.im_batch, im)
            data.im_batch_np = self.expand_data(data.im_batch_np, im_np)
            data.depth_batch_np = self.expand_data(
                data.depth_batch_np, depth_np)
            data.depth_batch = self.expand_data(
                data.depth_batch, depth)

            if self.gt_traj:
                data.T_WC_batch_np = self.expand_data(
                    data.T_WC_batch_np, T_np)
                data.T_WC_batch = self.expand_data(data.T_WC_batch, T)

        return data

    def add_data(self, data):
        replace = self.last_is_keyframe is False

        if self.last_is_keyframe:
            self.kf_indices.append(self.frame_id - 1)

        self.frames.im_batch = self.expand_data(
            self.frames.im_batch, data.im_batch, replace)
        self.frames.im_batch_np = self.expand_data(
            self.frames.im_batch_np, data.im_batch_np, replace)
        self.frames.depth_batch_np = self.expand_data(
            self.frames.depth_batch_np,
            data.depth_batch_np, replace
        )
        self.frames.depth_batch = self.expand_data(
            self.frames.depth_batch,
            data.depth_batch, replace
        )
        empty_dist = torch.zeros([1], device=self.device)
        self.frames.frame_avg_losses = self.expand_data(
            self.frames.frame_avg_losses,
            empty_dist, replace
        )

        if self.gt_traj:
            self.frames.T_WC_batch = self.expand_data(
                self.frames.T_WC_batch,
                data.T_WC_batch,
                replace)
            self.frames.T_WC_batch_np = self.expand_data(
                self.frames.T_WC_batch_np,
                data.T_WC_batch_np,
                replace)

            if self.incremental:
                cam_center = self.frames.T_WC_batch_np[-1, :3, 3]
                self.trajectory_gt = self.expand_data(
                    self.trajectory_gt,
                    cam_center[None, ...],
                    False)
                self.poses_gt = self.expand_data(
                    self.poses_gt,
                    self.frames.T_WC_batch_np[-1][None, ...],
                    False)

        self.batch_size = self.frames.depth_batch.shape[0]

    def set_directions(self):
        self.dirs_C = render_rays.ray_dirs_C(
            1,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.device,
            depth_type="z",
        )

        dist = torch.sqrt(
            torch.sum((self.dirs_C[:, 0, 0, :] - self.dirs_C[:, 1, 0, :])**2, -1))
        self.radius = dist * 2 / np.sqrt(12)

        self.dirs_C_vis = render_rays.ray_dirs_C(
            1,
            self.H_vis,
            self.W_vis,
            self.fx_vis,
            self.fy_vis,
            self.cx_vis,
            self.cy_vis,
            self.device,
            depth_type="z",
        )
        dist_vis = torch.sqrt(
            torch.sum((self.dirs_C_vis[:, 0, 0, :] - self.dirs_C_vis[:, 1, 0, :])**2, -1))
        self.radius_vis = dist_vis * 2 / np.sqrt(12)
        self.dirs_C_vis = self.dirs_C_vis.view(1, -1, 3)

        self.dirs_C_vis_up = render_rays.ray_dirs_C(
            1,
            self.H_vis_up,
            self.W_vis_up,
            self.fx_vis_up,
            self.fy_vis_up,
            self.cx_vis_up,
            self.cy_vis_up,
            self.device,
            depth_type="z",
        )
        dist_vis_up = torch.sqrt(
            torch.sum((self.dirs_C_vis_up[:, 0, 0, :] - self.dirs_C_vis_up[:, 1, 0, :])**2, -1))
        self.radius_vis_up = dist_vis_up * 2 / np.sqrt(12)
        self.dirs_C_vis_up = self.dirs_C_vis_up.view(1, -1, 3)

    def set_cam(self):
        self.mh = self.config["camera"]["mh"]
        self.mw = self.config["camera"]["mw"]
        self.H = self.config["camera"]["h"] - 2 * self.mh
        self.W = self.config["camera"]["w"] - 2 * self.mw
        self.n_pix = self.H * self.W
        self.fx = self.config["camera"]["fx"]
        self.fy = self.config["camera"]["fy"]
        self.cx = self.config["camera"]["cx"] - self.mw
        self.cy = self.config["camera"]["cy"] - self.mh

        reduce_factor = self.config["vis"]["im_vis_reduce"]
        self.H_vis = self.H // reduce_factor
        self.W_vis = self.W // reduce_factor
        self.fx_vis = self.fx / reduce_factor
        self.fy_vis = self.fy / reduce_factor
        self.cx_vis = self.cx / reduce_factor
        self.cy_vis = self.cy / reduce_factor

        upscale_factor = 2
        self.H_vis_up = self.H // upscale_factor
        self.W_vis_up = self.W // upscale_factor
        self.fx_vis_up = self.fx / upscale_factor
        self.fy_vis_up = self.fy / upscale_factor
        self.cx_vis_up = self.cx / upscale_factor
        self.cy_vis_up = self.cy / upscale_factor

        if self.do_tsdf:
            self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
                width=self.W,
                height=self.H,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
            )

        self.loss_approx_factor = 8
        w_block = self.W // self.loss_approx_factor
        h_block = self.H // self.loss_approx_factor
        increments_w = torch.arange(
            self.loss_approx_factor, device=self.device) * w_block
        increments_h = torch.arange(
            self.loss_approx_factor, device=self.device) * h_block
        c, r = torch.meshgrid(increments_w, increments_h)
        c, r = c.t(), r.t()
        self.increments_single = torch.stack((r, c), dim=2).view(-1, 2)

    def set_params(self):
        self.ims_file = self.config["dataset"]["ims_file"]
        self.dataset_format = self.config["dataset"]["format"]
        if "scene_file" in self.config["dataset"]:
            self.scene_file = self.config["dataset"]["scene_file"]
        self.traj_file = None
        if "traj_file" in self.config["dataset"]:
            self.traj_file = self.config["dataset"]["traj_file"]
        if "im_indices" in self.config["dataset"]:
            self.indices = self.config["dataset"]["im_indices"]

        self.n_classes = None
        if "semantics" in self.config:
            self.do_hierarchical = bool(
                self.config["semantics"]["do_hierarchical"])
            self.do_names = bool(
                self.config["semantics"]["do_names"])
            if self.do_hierarchical:
                self.n_classes = self.config["semantics"]["h_levels"]
            else:
                self.n_classes = self.config["semantics"]["n_classes"]

            self.label_cam = bool(self.config["semantics"]["label_cam"])

        self.N_epochs = self.config["trainer"]["epochs"]
        self.scale = 1. / self.config["trainer"]["scale"]

        self.noise_std = self.config["model"]["noise_std"]
        self.noise_kf = self.config["model"]["noise_kf"]
        self.noise_frame = self.config["model"]["noise_frame"]
        self.color_scaling = self.config["model"]["color_scaling"]
        self.gt_traj = bool(self.config["model"]["gt_traj"])
        self.gt_scene = bool(self.config["model"]["gt_scene"])
        self.embed_type = self.config["model"]["embed_type"]

        self.n_embed_funcs = None
        self.do_coarse = False
        self.do_mip = False
        if self.embed_type == "gauss":
            self.gauss_std = self.config["model"]["gauss_embed"]["std"]
            self.embed_size = self.config["model"]["gauss_embed"]["size"]
        elif self.embed_type == "axis":
            self.n_embed_funcs = self.config["model"]["axis_embed"]["n_funcs"]
            self.embed_size = self.n_embed_funcs * 6 + 3
            # self.embed_size = 54
        elif self.embed_type == "mip":
            self.n_embed_funcs = self.config["model"]["mip_embed"]["n_funcs"]
            self.embed_size = self.n_embed_funcs * 6
            self.do_coarse = True
            self.do_mip = True

        self.keyframe_select = bool(self.config["model"]["keyframe_select"])
        self.window_size = self.config["model"]["window_size"]
        self.n_rays_is_kf = self.config["model"]["n_rays_is_kf"]
        self.do_tsdf = bool(self.config["model"]["do_tsdf"])
        self.hidden_layers_block = self.config["model"]["hidden_layers_block"]
        self.hidden_feature_size = self.config["model"]["hidden_feature_size"]

        self.learning_rate = self.config["optimizer"]["args"]["lr"]
        self.weight_decay = self.config["optimizer"]["args"]["weight_decay"]
        self.joint_pose_lr = self.config["optimizer"]["args"]["pose_lr"]

        self.n_rays = self.config["render"]["n_rays"]
        self.n_levels = self.config["render"]["pyramid_levels"]
        self.n_bins_fine = self.config["render"]["bins_fine"]
        self.iters_per_level = self.config["render"]["iters_per_level"]
        self.optim_frames = self.iters_per_level * self.n_levels
        self.iters_per_kf = self.config["render"]["iters_per_kf"]
        self.iters_per_frame = self.config["render"]["iters_per_frame"]
        self.kf_dist_th = self.config["render"]["kf_dist_th"]
        self.kf_pixel_ratio = self.config["render"]["kf_pixel_ratio"]
        self.max_depth = self.config["render"]["depth_range"][1]
        self.min_depth = self.config["render"]["depth_range"][0]
        self.n_bins = self.config["render"]["n_bins"]
        self.out_scale = self.n_bins * 1.8
        self.n_levels = self.config["render"]["pyramid_levels"]
        self.kernel_init = self.config["render"]["kernel_init"]

        self.n_rays_track = self.config["track"]["n_rays_track"]
        self.n_bins_track = self.config["track"]["n_bins_track"]
        self.n_bins_fine_track = self.config["track"]["n_bins_fine_track"]
        self.track_lr = self.config["track"]["lr"]
        self.max_track_iters = self.config["track"]["max_iters"]
        self.min_track_iters = self.config["track"]["min_iters"]
        self.delta_th = self.config["track"]["delta_th"]
        self.track_color = bool(self.config["track"]["track_color"])

        self.n_bins_fine_vis = self.config["vis"]["n_bins_fine_vis"]
        self.pyramid_level = -self.n_levels

        self.voxel_size = (self.max_depth - self.min_depth) / (2 * self.n_bins)

    def load_networks(self, do_color=False, do_sem=False):
        self.fc_occ_map = fc_map.OccupancyMap(
            self.embed_size,
            self.out_scale,
            hidden_size=self.hidden_feature_size,
            do_color=do_color,
            do_semantics=do_sem,
            n_classes=self.n_classes,
            hidden_layers_block=self.hidden_layers_block,
        )
        self.fc_occ_map.to(self.device)
        self.fc_occ_map.apply(fc_map.init_weights)

        self.optimiser = optim.AdamW(
            self.fc_occ_map.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.rot_exp = geometry.transform.RotExp.apply

        self.B_layer = None
        if self.embed_type == "gauss":
            self.B_layer = torch.nn.Linear(3, self.embed_size,
                                           bias=True).to(self.device)
            self.B_layer.weight.data.normal_(std=self.gauss_std)
            if self.B_layer.bias is not None:
                self.B_layer.bias.data.uniform_(-np.pi, np.pi)

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()

    def load_checkpoints(self, load_path, load_epoch):
        chechpoint_load_file = (
            load_path + "/checkpoints" "/epoch_" + str(load_epoch) + ".pth"
        )
        checkpoint = torch.load(chechpoint_load_file)
        self.fc_occ_map.load_state_dict(checkpoint["model_state_dict"])
        if self.B_layer is not None:
            self.B_layer.load_state_dict(checkpoint["B_layer_state_dict"])

        self.t_init = checkpoint["epoch"]

    def sample_rays(
        self,
        depth_batch,
        T_WC_batch,
        indices_b,
        indices_h,
        indices_w,
        im_batch=None,
        get_masks=False,
        is_active=False
    ):
        depth_sample = depth_batch[indices_b, indices_h, indices_w].view(-1)
        mask_valid_depth = depth_sample != 0
        depth_sample = depth_sample[mask_valid_depth]
        color_sample = None

        # if no color only render where there is valid depth
        if im_batch is None:
            indices_b = indices_b[mask_valid_depth]
            indices_h = indices_h[mask_valid_depth]
            indices_w = indices_w[mask_valid_depth]

        else:
            color_sample = im_batch[indices_b,
                                    indices_h,
                                    indices_w, :].view(-1, 3)

        T_WC_sample = T_WC_batch[indices_b]
        dirs_C_sample = self.dirs_C[0, indices_h, indices_w, :].view(-1, 3)

        # if there is no color valid depth already masked
        if im_batch is not None:
            indices_b = indices_b[mask_valid_depth]
            indices_h = indices_h[mask_valid_depth]
            indices_w = indices_w[mask_valid_depth]

        masks = None
        if get_masks:
            masks = torch.zeros(depth_batch.shape, device=depth_batch.device)
            masks[indices_b, indices_h, indices_w] = 1

        if is_active:
            self.active_inds = torch.stack([indices_b, indices_h, indices_w])

        return (
            dirs_C_sample,
            depth_sample,
            color_sample,
            T_WC_sample,
            masks,
            mask_valid_depth,
            indices_b,
            indices_h,
            indices_w
        )

    def compute_features(
        self,
        depth_batch,
        T_WC_batch,
        indices_b,
        indices_h,
        indices_w,
        get_masks=True,
        im_batch=None,
        is_active=False,
    ):
        (
            dirs_C_sample,
            depth_sample,
            color_sample,
            T_WC_sample,
            binary_masks,
            mask_valid_depth,
            indices_b,
            indices_h,
            indices_w
        ) = self.sample_rays(
            depth_batch,
            T_WC_batch,
            indices_b,
            indices_h,
            indices_w,
            im_batch=im_batch,
            get_masks=get_masks,
            is_active=is_active
        )


        fc4_features = render_rays.render_features(T_WC_sample,
                                                   self.min_depth,
                                                   self.max_depth,
                                                   self.n_embed_funcs,
                                                   self.n_bins,
                                                   self.n_bins_fine,
                                                   self.fc_occ_map,
                                                   self.B_layer,
                                                   dirs_C=dirs_C_sample,
                                                   grad=False)

        return fc4_features

    def compute_features_2(
        self,
        depth_batch,
        T_WC_batch,
        indices_b,
        indices_h,
        indices_w,
        get_masks=True,
        im_batch=None,
        is_active=False,
    ):
        # convert depth image to xyz image
        xyz_grid = np.indices((self.H, self.W), dtype=np.float32).transpose(1, 2, 0)
        # Move origin to bottom left
        xyz_grid[..., 0] = np.flipud(xyz_grid[..., 0])
        xyz_grid = torch.from_numpy(xyz_grid).to(depth_batch.device)
        z = depth_batch[indices_b[0]] # assuming same batch index for all samples
        x = (xyz_grid[..., 1] - self.cx) * z / self.fx
        y = (xyz_grid[..., 0] - self.cy) * z / self.fy
        xyz_image = torch.stack([x, y, z], dim=-1)

        # compute features for each of the indices
        fc4_features = torch.zeros((len(indices_w), self.hidden_feature_size), dtype=torch.float32)
        for i in range(len(indices_w)):
            # compute positional encoding
            xyz = xyz_image[indices_h[i], indices_w[i], :]
            points_embedding = embedding.positional_encoding(xyz, self.B_layer, num_encoding_functions=self.n_embed_funcs)
            fc4_features[i] = self.fc_occ_map.forward_features_only(points_embedding)
        return fc4_features.detach()

    def render_losses(
        self,
        render_depth,
        render_col,
        render_sem,
        var,
        mask_valid_depth,
        im_batch,
        depth_sample,
        color_sample,
        do_color,
        do_sem,
        do_information,
        loss_importance=None,
        sem_labels=None,
        h_labels=None,
        h_masks=None,
    ):
        if var is not None:
            var = var.detach()

        # if there is color valid render has not been masked
        if im_batch is not None:
            render_depth_masked = render_depth[mask_valid_depth]
            if var is not None:
                var_masked = var[mask_valid_depth]
        else:
            render_depth_masked = render_depth
            if var is not None:
                var_masked = var

        loss = render_rays.render_loss(
            render_depth_masked,
            depth_sample,
            normalise=False)

        if loss_importance is not None:
            loss_normalised = loss * loss_importance[mask_valid_depth]
        else:
            loss_normalised = loss

        if do_information is False:
            var_masked = None
        loss_info = render_rays.reduce_loss(
            loss_normalised, var=var_masked, avg=True)

        loss_col = None
        if do_color:
            loss_col = render_rays.render_loss(
                render_col, color_sample, normalise=False)

            if loss_importance is not None:
                loss_col_normalised = loss_col * loss_importance[..., None]
            else:
                loss_col_normalised = loss_col

            loss = loss + loss_col[mask_valid_depth].sum(-1) / 3.

            loss_col = render_rays.reduce_loss(
                loss_col_normalised, avg=True)

        sem_loss = None
        if do_sem:
            if self.do_hierarchical:
                h_class = torch.sigmoid(render_sem[h_masks])
                sem_loss = self.bce_loss(h_class, h_labels[h_masks])
            else:
                sem_loss = self.ce_loss(render_sem, sem_labels)

        return (
            loss_info,
            loss,
            loss_col,
            sem_loss,
        )

    def render_and_loss(
        self,
        depth_batch,
        T_WC_batch,
        do_information,
        n_bins,
        do_fine,
        n_bins_fine,
        indices_b,
        indices_h,
        indices_w,
        noise_std=None,
        get_masks=True,
        im_batch=None,
        loss_importance=None,
        is_active=False,
        sem_labels=None,
        h_labels=None,
        h_masks=None,
        do_coarse=False
    ):
        (
            dirs_C_sample,
            depth_sample,
            color_sample,
            T_WC_sample,
            binary_masks,
            mask_valid_depth,
            indices_b,
            indices_h,
            indices_w
        ) = self.sample_rays(
            depth_batch,
            T_WC_batch,
            indices_b,
            indices_h,
            indices_w,
            im_batch=im_batch,
            get_masks=get_masks,
            is_active=is_active
        )

        do_color = im_batch is not None
        do_sem = sem_labels is not None
        render_depth, var, render_col, render_sem = render_rays.render_images(
            T_WC_sample,
            self.min_depth,
            self.max_depth,
            self.n_embed_funcs,
            n_bins,
            self.fc_occ_map,
            self.B_layer,
            grad=True,
            dirs_C=dirs_C_sample,
            do_fine=do_fine,
            do_color=do_color,
            do_sem=do_sem,
            n_bins_fine=n_bins_fine,
            noise_std=noise_std,
            render_coarse=do_coarse,
            radius=self.radius,
            do_mip=self.do_mip
        )

        if do_coarse:
            render_depth_f = render_depth[0]
            render_col_f = render_col[0]
            render_sem_f = render_sem[0]
            var_f = var[0]

            (
                loss_info_c,
                _,
                loss_col_c,
                _,
            ) = self.render_losses(
                render_depth[1],
                render_col[1],
                render_sem[1],
                var[1],
                mask_valid_depth,
                im_batch,
                depth_sample,
                color_sample,
                do_color,
                do_sem,
                do_information,
                loss_importance,
                sem_labels,
                h_labels,
                h_masks
            )
        else:
            render_depth_f = render_depth
            render_col_f = render_col
            render_sem_f = render_sem
            var_f = var

        (
            loss_info_f,
            loss,
            loss_col_f,
            sem_loss,
        ) = self.render_losses(
            render_depth_f,
            render_col_f,
            render_sem_f,
            var_f,
            mask_valid_depth,
            im_batch,
            depth_sample,
            color_sample,
            do_color,
            do_sem,
            do_information,
            loss_importance,
            sem_labels,
            h_labels,
            h_masks
        )

        if do_coarse:
            loss_info = 0.1 * loss_info_c + loss_info_f
            loss_col = None
            if do_color:
                loss_col = 0.1 * loss_col_c + loss_col_f
        else:
            loss_info = loss_info_f
            loss_col = loss_col_f

        return (
            loss_info,
            loss,
            binary_masks,
            loss_col,
            sem_loss,
            indices_b,
            indices_h,
            indices_w
        )

    def init_track(self):
        # if self.track_optimiser is None:
        trans_delta = torch.zeros(3, device=self.device)
        w = torch.zeros((1, 3), device=self.device)
        trans_delta.requires_grad_(True)
        w.requires_grad_(True)
        track_optimiser = optim.Adam([w, trans_delta])

        lr = self.track_lr
        set_lr(track_optimiser, lr)

        return w, trans_delta, track_optimiser, lr

    def step_track(
        self,
        w,
        trans_delta,
        depth_batch,
        track_optim,
        do_information,
        n_bins,
        do_fine,
        n_bins_fine,
        im_batch,
        col_scaling=5
    ):
        w.data.fill_(0)
        trans_delta.data.fill_(0)

        R_WC_delta = self.rot_exp(w)
        R_WC = R_WC_delta @ self.T_WC[0, 0:3, 0:3]
        t_WC = trans_delta + self.T_WC[0, 0:3, 3]

        T_WC = torch.eye(4, device=self.device).unsqueeze(0)
        T_WC[0, 0:3, 0:3] = R_WC
        T_WC[0, 0:3, 3] = t_WC

        indices_b, indices_h, indices_w = random.sample_pixels(
            self.n_rays_track,
            1,
            depth_batch.shape[1],
            depth_batch.shape[2],
            device=self.device,
        )

        loss_info, _, _, l_col_random, _, _, _, _ = self.render_and_loss(
            depth_batch,
            T_WC,
            do_information,
            n_bins,
            do_fine,
            n_bins_fine,
            indices_b,
            indices_h,
            indices_w,
            get_masks=False,
            im_batch=im_batch,
            do_coarse=self.do_coarse
        )

        loss = loss_info
        if l_col_random is not None:
            loss = loss + l_col_random * col_scaling

        loss.backward()
        track_optim.step()
        track_optim.zero_grad(set_to_none=True)

        # update
        R_WC_delta = self.rot_exp(w)
        self.T_WC[0, 0:3, 0:3] = (R_WC_delta @ self.T_WC[0, 0:3, 0:3]).detach()
        self.T_WC[0, 0:3, 3] = (trans_delta + self.T_WC[0, 0:3, 3]).detach()

    def init_trajectory(self, T_WC):
        T_WC_np = T_WC.cpu().numpy()
        cam_center = T_WC_np[0, :3, 3]
        self.trajectory = cam_center[None, ...]
        self.poses = T_WC_np

    def init_pose_vars(self, T_WC_track_init):
        trans_delta = torch.zeros((3), device=self.device)
        w = torch.zeros((3), device=self.device)
        self.trans_deltas.append(trans_delta)
        self.w_deltas.append(w)

        self.frames.T_WC_track = T_WC_track_init
        self.T_WC = self.frames.T_WC_track.clone()

    def track(self,
              depth_gt,
              do_information,
              do_fine,
              depth_gt_np,
              im_batch,
              do_vis=False):

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        for param in self.fc_occ_map.parameters():
            param.requires_grad = False

        w, trans_delta, track_optimiser, lr = self.init_track()

        for track_iter in range(self.max_track_iters):
            self.step_track(
                w,
                trans_delta,
                depth_gt,
                track_optimiser,
                do_information,
                self.n_bins_track,
                do_fine,
                self.n_bins_fine_track,
                im_batch
            )

            angle_delta_norm = torch.norm(w)
            trans_delta_norm = torch.norm(trans_delta)

            if (
                angle_delta_norm < self.delta_th
                and trans_delta_norm < self.delta_th
                and track_iter > self.min_track_iters
            ):
                break

        torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        track_time = start.elapsed_time(end)

        if False:
            view_depths_track, _, _, _ = render_vis(
                1,
                self,
                self.T_WC,
                do_fine,
                do_var=False,
                do_color=False,
            )
            depth_gt_np_resize = imgviz.resize(
                depth_gt_np,
                width=self.W_vis,
                height=self.H_vis,
                interpolation="nearest")[None, ...]
            track_viz = image_vis(1, depth_gt_np_resize, view_depths_track,)
            track_viz = cv2.cvtColor(track_viz, cv2.COLOR_RGB2BGR)
            track_viz = imgviz.resize(track_viz, width=200)
            cv2.imshow("track_viz", track_viz)
            cv2.waitKey(1)

        T_WC_np = self.T_WC.cpu().numpy()
        cam_center = T_WC_np[0, :3, 3]
        self.trajectory = np.concatenate(
            (self.trajectory, cam_center[None, ...]), 0)
        self.poses = np.concatenate(
            (self.poses, T_WC_np), 0)

        for param in self.fc_occ_map.parameters():
            param.requires_grad = True

        return track_time

    def init_joint_poses(self, T_WC_batch, idxs=None):
        for i in range(len(self.w_deltas)):
            self.w_deltas[i].data.fill_(0)
            self.trans_deltas[i].data.fill_(0)

        w_deltas_batch = torch.stack(self.w_deltas)
        trans_deltas_batch = torch.stack(self.trans_deltas)

        if idxs:
            w_deltas_batch = w_deltas_batch[idxs, ...]
            trans_deltas_batch = trans_deltas_batch[idxs, ...]

        R_WC_delta = self.rot_exp(w_deltas_batch)
        R_WC = R_WC_delta @ T_WC_batch[:, 0:3, 0:3]
        t_WC = trans_deltas_batch + T_WC_batch[:, 0:3, 3]

        T_WCs = (
            torch.eye(4, device=self.device)
            .unsqueeze(0)
            .repeat(T_WC_batch.shape[0], 1, 1)
        )
        T_WCs[:, 0:3, 0:3] = R_WC
        T_WCs[:, 0:3, 3] = t_WC

        return T_WCs

    def update_joint_poses(self, T_WC_batch, idxs):
        w_deltas_batch = torch.stack(self.w_deltas)
        trans_deltas_batch = torch.stack(self.trans_deltas)

        if idxs:
            w_deltas_batch = w_deltas_batch[idxs, ...]
            trans_deltas_batch = trans_deltas_batch[idxs, ...]

        R_WC_delta = self.rot_exp(w_deltas_batch)
        T_WC_batch[:, 0:3, 0:3] = (
            R_WC_delta @ T_WC_batch[:, 0:3, 0:3]).detach()

        T_WC_batch[:, 0:3, 3] = (
            trans_deltas_batch + T_WC_batch[:, 0:3, 3]).detach()

        self.T_WC[0] = T_WC_batch[-1]

    def select_keyframes(self):
        limit = self.batch_size - 2
        loss_dist = self.frames.frame_avg_losses[:-2] / \
            self.frames.frame_avg_losses[:-2].sum()
        loss_dist_np = loss_dist.cpu().numpy()
        rand_ints = np.random.choice(
            np.arange(0, limit),
            size=self.window_size - 2,
            replace=False,
            p=loss_dist_np)

        last = self.batch_size - 1
        idxs = [*rand_ints, last - 1, last]

        return idxs

    def step_sem(self,
                 depth_batch,
                 T_WC_batch,
                 do_information,
                 do_fine,
                 indices_b,
                 indices_h,
                 indices_w,
                 labels,
                 h_labels,
                 h_masks,
                 im_batch=None,
                 color_scaling=5.0,
                 sem_scaling=8.0):

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        (
            l_depth_info,
            _,
            _,
            l_col,
            l_sem,
            _, _, _
        ) = self.render_and_loss(
            depth_batch,
            T_WC_batch,
            do_information,
            self.n_bins,
            do_fine,
            self.n_bins_fine,
            indices_b,
            indices_h,
            indices_w,
            noise_std=self.noise_std,
            get_masks=False,
            im_batch=im_batch,
            sem_labels=labels,
            h_labels=h_labels,
            h_masks=h_masks
        )

        l_total = l_depth_info + l_sem * sem_scaling
        if im_batch is not None:
            l_total = l_total + l_col * color_scaling

        ## HYPER_PARAM
        n_sem_pixels = indices_b.shape[0]
        weight = float(2*n_sem_pixels) / (self.n_rays * 1)
        print("weight: ", weight)

        # set weight to one
        l_total = l_total * weight
        l_total.backward()
        self.optimiser.step()
        self.optimiser.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        step_time = start.elapsed_time(end)

        return step_time

    def step(
        self,
        depth_batch,
        T_WC_batch,
        do_information,
        do_fine,
        do_active=True,
        im_batch=None,
        color_scaling=5.0,
    ):

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if self.keyframe_select and self.batch_size > self.window_size and self.incremental:
            idxs = self.select_keyframes()
            self.win_idxs = idxs

            depth_batch = depth_batch[idxs, ...]
            T_WC_select = T_WC_batch[idxs, ...]
            if im_batch is not None:
                im_batch = im_batch[idxs, ...]
            n_batch = len(idxs)
        else:
            idxs = None
            n_batch = self.batch_size
            T_WC_select = T_WC_batch

        indices_b, indices_h, indices_w = random.sample_pixels(
            self.n_rays,
            n_batch,
            depth_batch.shape[1],
            depth_batch.shape[2],
            device=self.device,
        )

        if self.w_deltas:
            T_WCs = self.init_joint_poses(T_WC_select, idxs)
        else:
            T_WCs = T_WC_select

        (
            l_depth_info_random,
            l_depth_random,
            binary_masks,
            l_col_random,
            _,
            indices_b,
            indices_h,
            indices_w
        ) = self.render_and_loss(
            depth_batch,
            T_WCs,
            do_information,
            self.n_bins,
            do_fine,
            self.n_bins_fine,
            indices_b,
            indices_h,
            indices_w,
            noise_std=self.noise_std,
            im_batch=im_batch,
            do_coarse=self.do_coarse
        )

        l_random = l_depth_info_random
        if im_batch is not None:
            l_random = l_random + l_col_random * color_scaling

        l_random.backward()
        self.optimiser.step()

        self.optimiser.zero_grad(set_to_none=True)

        total_loss = l_depth_random.mean()

        if self.w_deltas:
            self.update_joint_poses(T_WC_select, idxs)
            if idxs is not None:
                T_WC_batch[idxs] = T_WC_select

        full_loss = torch.zeros(
            depth_batch.shape, device=depth_batch.device)
        full_loss[indices_b, indices_h,
                  indices_w] = l_depth_random.detach()

        loss_approx = render_rays.loss_approx(
            full_loss, binary_masks, self.W, self.H,
            factor=self.loss_approx_factor
        )
        factor = loss_approx.shape[1]
        frame_sum = loss_approx.sum(dim=(1, 2))
        frame_avg_loss = frame_sum / (factor * factor)

        if idxs:
            self.frames.frame_avg_losses[idxs] = frame_avg_loss
        else:
            self.frames.frame_avg_losses = frame_avg_loss

        if do_active:
            (
                indices_b,
                indices_h,
                indices_w,
                loss_importance
            ) = random.active_sample(
                loss_approx, frame_sum,
                self.W, self.H,
                n_batch, self.n_rays,
                self.increments_single
            )

            (
                l_depth_info_active,
                _,
                binary_masks_active,
                l_col_active,
                _,
                _, _, _
            ) = self.render_and_loss(
                depth_batch,
                T_WC_select,
                do_information,
                self.n_bins,
                do_fine,
                self.n_bins_fine,
                indices_b,
                indices_h,
                indices_w,
                noise_std=self.noise_std,
                get_masks=False,
                im_batch=im_batch,
                loss_importance=loss_importance,
                is_active=True
            )
            l_active = l_depth_info_active
            if im_batch is not None:
                l_active = l_active + l_col_active * color_scaling

            l_active.backward()
            self.optimiser.step()
            self.optimiser.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        step_time = start.elapsed_time(end)

        return total_loss, step_time

    def add_track_variable(self):
        trans_delta = torch.zeros((3), device=self.device)
        w = torch.zeros((3), device=self.device)

        w.requires_grad_(True)
        trans_delta.requires_grad_(True)

        self.optimiser.add_param_group({"params": w, "lr": self.joint_pose_lr})
        self.optimiser.add_param_group(
            {"params": trans_delta, "lr": self.joint_pose_lr}
        )

        self.trans_deltas.append(trans_delta)
        self.w_deltas.append(w)

    def add_track_pose(self, T_WC):
        if self.last_is_keyframe:
            self.frames.T_WC_track = torch.cat((self.frames.T_WC_track, T_WC))
            self.add_track_variable()
        else:
            self.frames.T_WC_track[-1] = T_WC[0]

    def add_track_variable_noopt(self):
        trans_delta = torch.zeros((3), device=self.device)
        w = torch.zeros((3), device=self.device)
        self.trans_deltas.append(trans_delta)
        self.w_deltas.append(w)


    def add_track_pose_noopt(self, T_WC):
        if self.last_is_keyframe:
            self.frames.T_WC_track = torch.cat((self.frames.T_WC_track, T_WC))
            self.add_track_variable_noopt()
        else:
            self.frames.T_WC_track[-1] = T_WC[0]

    def is_keyframe(self, T_WC, depth_gt, do_fine, n_bins_fine):
        indices_b, indices_h, indices_w = random.sample_pixels(
            self.n_rays_is_kf,
            1,
            depth_gt.shape[1],
            depth_gt.shape[2],
            device=self.device,
        )

        (
            dirs_C_sample,
            depth_sample,
            _,
            T_WC_sample,
            _,
            _,
            indices_b,
            indices_h,
            indices_w
        ) = self.sample_rays(
            depth_gt,
            T_WC,
            indices_b,
            indices_h,
            indices_w
        )

        render_depth, _, _, _ = render_rays.render_images(
            T_WC_sample,
            self.min_depth,
            self.max_depth,
            self.n_embed_funcs,
            self.n_bins,
            self.frozen_fc_occ_map,
            self.frozen_B_layer,
            grad=False,
            dirs_C=dirs_C_sample,
            do_fine=do_fine,
            do_color=False,
            do_var=False,
            n_bins_fine=n_bins_fine,
            radius=self.radius,
            do_mip=self.do_mip
        )
        loss = render_rays.render_loss(
            render_depth,
            depth_sample,
            normalise=True)
        below_th = loss < self.kf_dist_th
        size_loss = below_th.shape[0]
        below_th_prop = below_th.sum().float() / size_loss
        is_keyframe = below_th_prop.item() < self.kf_pixel_ratio

        return is_keyframe

    def update_vis_vars(self, depth_batch_np, im_batch_np):
        if self.gt_depth_vis is None:
            updates = depth_batch_np.shape[0]
        else:
            diff_size = depth_batch_np.shape[0] - \
                self.gt_depth_vis.shape[0]
            updates = diff_size + 1

        for i in range(updates, 0, -1):
            prev_depth_gt = depth_batch_np[-i]
            prev_im_gt = im_batch_np[-i]
            prev_depth_gt_resize = imgviz.resize(
                prev_depth_gt, width=self.W_vis,
                height=self.H_vis,
                interpolation="nearest")[None, ...]
            prev_im_gt_resize = imgviz.resize(
                prev_im_gt, width=self.W_vis,
                height=self.H_vis)[None, ...]

            replace = False
            if i == updates:
                replace = True

            if self.gt_scene is False:
                pcs_gt_cam = backproject_pcs(prev_depth_gt_resize,
                                             self.fx_vis,
                                             self.fy_vis,
                                             self.cx_vis,
                                             self.cy_vis)
                self.pcs_gt_cam = self.expand_data(
                    self.pcs_gt_cam, pcs_gt_cam, replace=replace)

            self.gt_depth_vis = self.expand_data(
                self.gt_depth_vis,
                prev_depth_gt_resize,
                replace=replace)
            self.gt_im_vis = self.expand_data(
                self.gt_im_vis,
                prev_im_gt_resize,
                replace=replace)

    def slice_occupancy(self,
                        dim_x=1024,
                        range_x=[-4, 4],
                        range_y=[-2.3, 2.3],
                        trans=[3, 1.1],
                        height=-1.22):

        ratio = range_y[1] / float(range_x[1])
        dim_y = int(dim_x * ratio)

        t_x = torch.linspace(
            range_x[0], range_x[1], steps=dim_x, device=self.device)
        t_y = torch.linspace(
            range_y[0], range_y[1], steps=dim_y, device=self.device)

        grid = torch.meshgrid(t_x, t_y)
        heights = torch.zeros(grid[0].shape, device=self.device) + height

        slice_occ = torch.cat((grid[0][..., None] + trans[0],
                               grid[1][..., None] + trans[1],
                               heights[..., None]
                               ), dim=2)

        points_embedding = embedding.positional_encoding(
            slice_occ,
            self.B_layer,
            num_encoding_functions=self.n_embed_funcs
        )
        with torch.set_grad_enabled(False):
            alphas, _, _ = self.fc_occ_map(
                points_embedding,
                do_alpha=True,
                do_color=False)
        delta = torch.tensor([0.005], device=self.device)
        occ = 1 - render_rays.occupancy_activation(alphas, delta)
        occ_image = occ[:, :, 0].cpu().detach().numpy()
        alpha_image = alphas[:, :, 0].cpu().detach().numpy()
        occ_viz = imgviz.depth2rgb(occ_image, min_value=0, max_value=1.)
        alpha_viz = imgviz.depth2rgb(alpha_image)
        cv2.imshow("occ_viz", occ_viz)
        cv2.imshow("alpha_viz", alpha_viz)
        cv2.waitKey(1)

        return occ_viz

    def draw_3D(
        self,
        show_pc,
        show_mesh,
        level_marching,
        view_depths,
        view_cols,
        depth_gt,
        T_WC_batch_np,
        draw_cameras=False,
    ):
        scene = trimesh.Scene()
        scene.set_camera()
        scene.camera.focal = (self.fx, self.fy)
        scene.camera.resolution = (self.W, self.H)

        if draw_cameras:
            draw_cams(self.batch_size, T_WC_batch_np, scene)

            if self.trajectory is not None:
                visualisation.draw.draw_trajectory(
                    self.trajectory, scene, color=(1.0, 1.0, 0.0)
                )
            if self.trajectory_gt is not None:
                visualisation.draw.draw_trajectory(
                    self.trajectory_gt, scene, color=(1.0, 0.0, 1.0)
                )

        if show_pc:
            pcs_cam = backproject_pcs(view_depths,
                                      self.fx_vis,
                                      self.fy_vis,
                                      self.cx_vis,
                                      self.cy_vis)
            draw_pc(
                self.batch_size,
                pcs_cam,
                T_WC_batch_np,
                view_cols,
                scene
            )

        if show_mesh:
            if self.gt_scene is False:
                pc = draw_pc(
                    self.batch_size,
                    self.pcs_gt_cam,
                    T_WC_batch_np
                )
                self.transform, self.extents = trimesh.bounds.oriented_bounds(
                    pc)
                self.transform = np.linalg.inv(self.transform)
                self.set_scene_bounds(self.extents, self.transform)

            with torch.set_grad_enabled(False):
                alphas, _, _ = occupancy.chunks(
                    self.grid_pc,
                    self.chunk_size,
                    self.fc_occ_map,
                    self.n_embed_funcs,
                    self.B_layer,
                )

                occ = render_rays.occupancy_activation(alphas, self.voxel_size)
                dim = self.grid_dim
                occ = occ.view(dim, dim, dim)

            fc_occ_map = None
            if self.do_color:
                fc_occ_map = self.fc_occ_map

            occ_mesh = draw_mesh(
                occ,
                level_marching,
                self.scene_scale_np,
                self.bounds_tranform_np,
                self.chunk_size,
                self.B_layer,
                self.device,
                fc_occ_map,
                self.n_embed_funcs,
            )
            scene.add_geometry(occ_mesh)

        view_idx = -1
        scene.camera_transform = T_WC_batch_np[view_idx]
        scene.camera_transform = scene.camera_transform @ geometry.transform.to_trimesh()
        scene.camera_transform = (
            scene.camera_transform
            @ trimesh.transformations.translation_matrix([0, 0, 0.1])
        )

        return scene

    def add_frame(self, frame_data, frames_limit=None):
        if self.last_is_keyframe:
            self.frozen_fc_occ_map = copy.deepcopy(self.fc_occ_map)
            self.frozen_B_layer = None
            if self.B_layer is not None:
                self.frozen_B_layer = copy.deepcopy(self.B_layer)

        self.add_data(frame_data)
        if frames_limit:
            self.cut_data(frames_limit)

        self.frames_since_add = 0

    def check_keyframe(self, T_WC):
        new_frame = False

        if self.last_is_keyframe:
            new_frame = True
            self.optim_frames = self.iters_per_frame
            self.noise_std = self.noise_frame

        else:
            depth_gt = self.frames.depth_batch[-1].unsqueeze(0)

            self.last_is_keyframe = self.is_keyframe(
                T_WC, depth_gt, self.do_fine, self.n_bins_fine
            )

            if self.last_is_keyframe:
                self.optim_frames = self.iters_per_kf
                self.noise_std = self.noise_kf
            else:
                new_frame = True
                self.optim_frames = self.iters_per_frame
                self.noise_std = self.noise_frame

        return new_frame


def render_vis_up(depth,
                  T_WC,
                  trainer,
                  origins_dirs,
                  do_color=False,
                  do_sem=False,
                  do_hierarchical=False,
                  radius=None,
                  do_mip=False
                  ):
    depth_up = torch.nn.functional.interpolate(
        depth[None, ...],
        size=[trainer.H_vis_up, trainer.W_vis_up],
        mode='bilinear', align_corners=True
    )

    depth_up = depth_up.view(-1)
    z_vals_limits = torch.stack((depth_up - 0.0085,
                                 depth_up - 0.0025,
                                 depth_up + 0.0025,
                                 depth_up + 0.0085), dim=1)

    view_depth_up, _, view_color_up, view_sem_up = render_rays.render_images(
        T_WC,
        trainer.min_depth,
        trainer.max_depth,
        trainer.n_embed_funcs,
        trainer.n_bins,
        trainer.fc_occ_map,
        trainer.B_layer,
        H=trainer.H_vis_up,
        W=trainer.W_vis_up,
        fx=trainer.fx_vis_up,
        fy=trainer.fy_vis_up,
        cx=trainer.cx_vis_up,
        cy=trainer.cy_vis_up,
        grad=False,
        do_fine=False,
        do_var=False,
        do_color=do_color,
        do_sem=do_sem,
        n_bins_fine=trainer.n_bins_fine_vis,
        z_vals_limits=z_vals_limits,
        normalise_weights=True,
        do_sem_activation=True,
        do_hierarchical=do_hierarchical,
        radius=radius,
        origins_dirs=origins_dirs,
        do_mip=do_mip
    )
    if do_sem:
        view_sem_up = view_sem_up.view(
            trainer.H_vis_up, trainer.W_vis_up, trainer.n_classes)

    if do_color:
        view_color_up = view_color_up.view(
            trainer.H_vis_up, trainer.W_vis_up, 3).cpu().numpy()

    return view_depth_up, view_color_up, view_sem_up


def render_vis(
    batch_size,
    trainer,
    T_WC_batch,
    do_fine,
    do_var=True,
    do_color=False,
    do_sem=False,
    do_hierarchical=False,
    radius=None,
    do_mip=False
):
    view_depths = []
    if do_var:
        view_vars = []
    else:
        view_vars = None

    if do_color:
        view_cols = []
    else:
        view_cols = None

    if do_sem:
        view_sems = []
    else:
        view_sems = None

    for batch_i in range(batch_size):
        T_WC = T_WC_batch[batch_i].unsqueeze(0)

        view_depth, var, view_color, view_sem = render_rays.render_images(
            T_WC,
            trainer.min_depth,
            trainer.max_depth,
            trainer.n_embed_funcs,
            trainer.n_bins,
            trainer.fc_occ_map,
            trainer.B_layer,
            H=trainer.H_vis,
            W=trainer.W_vis,
            fx=trainer.fx_vis,
            fy=trainer.fy_vis,
            cx=trainer.cx_vis,
            cy=trainer.cy_vis,
            grad=False,
            dirs_C=trainer.dirs_C_vis,
            do_fine=do_fine,
            do_var=do_var,
            do_color=do_color,
            do_sem=do_sem,
            n_bins_fine=trainer.n_bins_fine_vis,
            do_sem_activation=True,
            do_hierarchical=do_hierarchical,
            radius=radius,
            do_mip=do_mip
        )

        view_depth = view_depth.view(trainer.H_vis, trainer.W_vis)
        view_depths.append(view_depth)

        if do_var:
            view_var = var.view(trainer.H_vis, trainer.W_vis)
            view_vars.append(view_var)

        if do_color:
            view_color = view_color.view(trainer.H_vis, trainer.W_vis, 3)
            view_cols.append(view_color)

        if do_sem:
            view_sem = view_sem.view(
                trainer.H_vis, trainer.W_vis, trainer.n_classes)
            view_sems.append(view_sem)

    view_depths = torch.stack(view_depths)

    if do_var:
        view_vars = torch.stack(view_vars)
        view_vars = view_vars.cpu().numpy()

    if do_color:
        view_cols = torch.stack(view_cols)
        view_cols = (view_cols.cpu().numpy() * 255).astype(np.uint8)

    if do_sem:
        view_sems = torch.stack(view_sems)

    return view_depths, view_vars, view_cols, view_sems


def save_trajectory(traj, dir_name, file_name, format="replica", timestamps=None):
    traj_file = open(dir_name + file_name, "w")

    if format == "replica":
        for T_WC in traj:
            np.savetxt(traj_file, T_WC[:3, :].reshape([1, 12]), fmt="%f")
    elif format == "TUM":
        for idx, T_WC in enumerate(traj):
            quat = trimesh.transformations.quaternion_from_matrix(T_WC[:3, :3])
            quat = np.roll(quat, -1)
            trans = T_WC[:3, 3]
            time = timestamps[idx]

            traj_file.write('{} '.format(time))
            np.savetxt(traj_file, trans.reshape([1, 3]), fmt="%f", newline=" ")
            np.savetxt(traj_file, quat.reshape([1, 4]), fmt="%f",)

    traj_file.close()


def image_vis(
    batch_size,
    gt_depth_ims,
    view_depths,
    view_vars=None,
    im_batch_np=None,
    view_cols=None,
):
    views = []
    for batch_i in range(batch_size):
        depth = view_depths[batch_i]
        depth_viz = imgviz.depth2rgb(depth)

        if view_vars is not None:
            var = view_vars[batch_i]
            var_viz = imgviz.depth2rgb(var)

        if view_cols is not None:
            view_col = view_cols[batch_i]

        if im_batch_np is not None:
            im = im_batch_np[batch_i]

        gt = gt_depth_ims[batch_i]
        gt_viz = imgviz.depth2rgb(gt)

        loss = np.abs(gt - depth)
        loss[gt == 0] = 0
        loss_viz = imgviz.depth2rgb(loss)

        visualisations = [gt_viz, depth_viz, loss_viz]
        if view_vars is not None:
            visualisations.append(var_viz)
        if im_batch_np is not None:
            visualisations.append(im)
        if view_cols is not None:
            visualisations.append(view_col)
        viz = np.vstack(visualisations)
        views.append(viz)

    viz = np.hstack(views)
    return viz


def live_vis_up(surface_normals,
                diffuse,
                sem_vis,
                view_color,
                label_vis,
                depth,
                rgb,
                entropy=None,
                count=None,
                ):
    surface_normals = (-surface_normals + 1.) / 2.

    diffuse = np.repeat(diffuse[:, :, np.newaxis], 3, axis=2)
    diffuse_im = diffuse
    specular_im = (diffuse**40)
    phong_col = view_color * 0.7 + diffuse_im * 0.15 + specular_im * 0.15
    phong_col = np.clip(phong_col, 0., 1.)
    surface_normals = np.clip(surface_normals, 0., 1.)
    phong_col = (phong_col * 255).astype(np.uint8)
    surface_normals = (surface_normals * 255).astype(np.uint8)

    if sem_vis is not None:
        sem_vis = sem_vis[0] / 255.
        phong_sem = sem_vis * 0.6 + diffuse_im * 0.2 + specular_im * 0.2
        phong_sem = np.clip(phong_sem, 0., 1.)
        phong_sem = (phong_sem * 255).astype(np.uint8)

    depth[depth == 0] = np.nan
    depth_vis = imgviz.depth2rgb(depth)

    padv = np.zeros([depth_vis.shape[0], 5, 3]).astype(np.uint8)
    vis1 = np.hstack([depth_vis, padv, rgb])
    vis2 = np.hstack([surface_normals, padv, phong_col])
    pad = np.zeros([5, vis1.shape[1], 3]).astype(np.uint8)
    vis_up = np.vstack([vis1, pad, vis2])
    if sem_vis is not None:
        vis3 = np.hstack([phong_sem, padv, label_vis])
        vis_up = np.vstack([vis_up, pad, vis3])

    if count is not None:
        im_file = "/home/data/screen_capture/" + "depth_vis_" + str(count) + ".png"
        cv2.imwrite(im_file, cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB))
        im_file = "/home/data/screen_capture/" + "rgb_" + str(count) + ".png"
        cv2.imwrite(im_file, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        im_file = "/home/data/screen_capture/" + "surface_normals_" + str(count) + ".png"
        cv2.imwrite(im_file, cv2.cvtColor(surface_normals, cv2.COLOR_BGR2RGB))
        im_file = "/home/data/screen_capture/" + "phong_col_" + str(count) + ".png"
        cv2.imwrite(im_file, cv2.cvtColor(phong_col, cv2.COLOR_BGR2RGB))
        im_file = "/home/data/screen_capture/" + "phong_sem_" + str(count) + ".png"
        cv2.imwrite(im_file, cv2.cvtColor(phong_sem, cv2.COLOR_BGR2RGB))
        im_file = "/home/data/screen_capture/" + "entropy_" + str(count) + ".png"
        cv2.imwrite(im_file, cv2.cvtColor(entropy, cv2.COLOR_BGR2RGB))
        im_file = "/home/data/screen_capture/" + "label_vis_" + str(count) + ".png"
        cv2.imwrite(im_file, cv2.cvtColor(label_vis, cv2.COLOR_BGR2RGB))

    vis_up = cv2.cvtColor(vis_up, cv2.COLOR_RGB2BGR)

    return vis_up


def live_vis(
    gt,
    depth,
    var=None,
    im_batch_np=None,
    view_cols=None,
    sem_ims=None,
    entropies=None,
    min_depth=None,
    max_depth=None
):

    depth_viz = imgviz.depth2rgb(
        depth, min_value=min_depth, max_value=max_depth)
    var_viz = imgviz.depth2rgb(var)
    gt[gt == 0] = np.nan
    gt_viz = imgviz.depth2rgb(
        gt, min_value=min_depth, max_value=max_depth)

    loss = np.abs(gt - depth)
    loss[gt == 0] = 0
    loss_viz = imgviz.depth2rgb(loss)

    gt_viz = np.hstack([im_batch_np, gt_viz])
    render_viz = np.hstack([view_cols, depth_viz])
    res_viz = np.hstack([var_viz, loss_viz])

    W = res_viz.shape[1]
    margin = np.zeros([5, W, 3], dtype=np.uint8) * 255
    viz = np.vstack([margin, gt_viz, margin, render_viz,
                     margin, res_viz])
    if sem_ims is not None:
        for entropy, sem_im in zip(entropies, sem_ims):
            entropy_viz = imgviz.depth2rgb(entropy)
            sem_viz = np.hstack([sem_im.astype(np.uint8), entropy_viz])
            viz = np.vstack([viz, margin, sem_viz])

    return viz


def backproject_pcs(depths, fx, fy, cx, cy):
    pcs = []
    batch_size = depths.shape[0]
    for batch_i in range(batch_size):
        pcd = geometry.transform.pointcloud_from_depth(
            depths[batch_i], fx, fy, cx, cy
        )
        pc_flat = pcd.reshape(-1, 3)
        pcs.append(pc_flat)

    pcs = np.stack(pcs, axis=0)
    return pcs


def draw_pc(batch_size,
            pcs_cam,
            T_WC_batch_np,
            im_batch=None,
            scene=None):

    pcs_w = []
    for batch_i in range(batch_size):
        T_WC = T_WC_batch_np[batch_i]
        pc_cam = pcs_cam[batch_i]

        col = None
        if im_batch is not None:
            img = im_batch[batch_i]
            col = img.reshape(-1, 3)

        pc_tri = trimesh.PointCloud(vertices=pc_cam, colors=col)
        pc_tri.apply_transform(T_WC)
        pcs_w.append(pc_tri.vertices)

        if scene is not None:
            scene.add_geometry(pc_tri)

    pcs_w = np.concatenate(pcs_w, axis=0)
    return pcs_w


def draw_cams(batch_size, T_WC_batch_np, scene):
    color = (0.0, 1.0, 0.0, 0.8)
    for batch_i in range(batch_size):
        T_WC = T_WC_batch_np[batch_i]

        camera = trimesh.scene.Camera(
            fov=scene.camera.fov, resolution=scene.camera.resolution
        )
        marker_height = 0.2
        if batch_i == batch_size - 1:
            color = (1.0, 1.0, 1.0, 1.0)
            marker_height = 0.5

        marker = visualisation.draw.draw_camera(
            camera, T_WC, color=color, marker_height=marker_height
        )
        scene.add_geometry(marker[1])


def draw_mesh(
    occ, level, scale, transform, chunk_size, B_layer, device,
    occ_map=None, n_embed_funcs=None,
    do_sem=False, do_hierarchical=False,
    colormap=None, h_level=0
):
    mat_march = occ.detach().cpu().numpy()
    mesh = occupancy.marching_cubes(mat_march, level)

    # Transform to [-1, 1] range
    mesh.apply_translation([-0.5, -0.5, -0.5])
    mesh.apply_scale(2)

    # Transform to scene coordinates
    mesh.apply_scale(scale)
    mesh.apply_transform(transform)

    if occ_map is not None:
        vertices = np.array(mesh.vertices)
        vertices = torch.from_numpy(vertices).float().to(device)

        with torch.set_grad_enabled(False):
            _, color, sems = occupancy.chunks(
                vertices,
                chunk_size,
                occ_map,
                n_embed_funcs,
                B_layer,
                do_alpha=False,
                do_color=True,
                do_sem=do_sem,
                do_hierarchical=do_hierarchical
            )

        mesh_color = color * 255
        vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors

        mesh_sem = None
        if do_sem:
            mesh_sem = mesh.copy()
            if do_hierarchical:
                binary = (torch.round(sems).long())
                mesh_color = ilabel.map_color_hierarchcial(
                    sems, colormap, binary, h_level)[0]

            else:
                sems = sems.squeeze(0)
                max_ind = torch.argmax(sems, axis=1)
                one_hot = torch.nn.functional.one_hot(
                    max_ind, num_classes=sems.shape[1])
                mesh_color = render_rays.map_color(one_hot, colormap)
            vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
            mesh_sem.visual.vertex_colors = vertex_colors

    else:
        mesh.visual.face_colors = [160, 160, 160, 255]


    return mesh, mesh_sem


def set_lr(optimiser, lr):
    for g in optimiser.param_groups:
        g["lr"] = lr
