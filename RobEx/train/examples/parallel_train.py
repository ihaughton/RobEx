#!/usr/bin/env python3
import torch

# from torch.utils.data import DataLoader
import cv2
import argparse
import queue
import open3d as o3d
import numpy as np
import imgviz
import trimesh
import timeit
import random
import copy
import pickle
import os
from datetime import datetime
from scipy import ndimage
from scipy import linalg
from scipy.spatial.transform import Rotation


from RobEx.train import trainer
from RobEx import visualisation
from RobEx.render import render_rays
from RobEx.mapping import occupancy
from RobEx.kinect_recorder import reader
from RobEx.label.ilabel import iLabel
from RobEx.label.ilabel import binary_to_decimal
from RobEx.visualisation import automatic_query

from RobEx.ros.ros_bridge import RosBridge
from RobEx.franka_control.imap_task_interface import ImapTaskInterface
from mercury.geometry.coordinate import Coordinate
import mercury
import pybullet_planning as pp

import time

# name = 'init_stairs'
torch.backends.cuda.matmul.allow_tf32 = False
TOP_DOWN = True

ROS = True

def init_gpu(gpu):
    tensor0 = torch.tensor([0.]).to(gpu)


def mask_from_inds(inds, H, W, H_vis, W_vis):
    inds_mask = inds[0] == inds[0, -1]
    h_inds = inds[1, inds_mask]
    w_inds = inds[2, inds_mask]
    mask = np.zeros([H, W])
    mask[h_inds, w_inds] = 1

    mask = ndimage.binary_dilation(
        mask, iterations=6)
    mask = (mask * 255).astype(np.uint8)

    return mask


def get_latest_queue(q):
    message = None
    while(True):
        try:
            message_latest = q.get(block=False)
            if message is not None:
                del message
            message = message_latest

        except queue.Empty:
            break

    return message


def tracking(track_to_map_IDT,
             map_to_track_params,
             track_to_vis_T_WC,
             wait_map_exit,
             gpu_for_process,
             config_file,
             load_path,
             load_epoch,
             show_mesh,
             incremental,
             do_track,
             do_color,
             do_sem,
             live=False,
             robot_start=None,
             sim=False,
             rosbag=False):
    print('track: starting')
    device = gpu_for_process["track"]
    render_trainer = trainer.RenderTrainer(
        device,
        config_file,
        load_path=load_path,
        load_epoch=load_epoch,
        do_mesh=show_mesh,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
        do_sem=do_sem
    )
    init_gpu(gpu_for_process["track"])
    init_gpu(gpu_for_process["map"])
    render_trainer.optimiser = None
    num_frames_in_recording = 0
    if render_trainer.config["camera"]["sensor"] == "Azure":
        sensor = "Azure"
        if live:
            config_file = render_trainer.config["kinect"]["config_file"]
            kinect_config = o3d.io.read_azure_kinect_sensor_config(config_file)
            if not ROS:
                data_reader = reader.KinectReaderLight(
                    kinect_config, 0,
                    w=render_trainer.config["camera"]["w"],
                    h=render_trainer.config["camera"]["h"],
                    fx=render_trainer.config["camera"]["fx"],
                    fy=render_trainer.config["camera"]["fy"],
                    cx=render_trainer.config["camera"]["cx"],
                    cy=render_trainer.config["camera"]["cy"],
                    k1=render_trainer.config["camera"]["k1"],
                    k2=render_trainer.config["camera"]["k2"],
                    k3=render_trainer.config["camera"]["k3"],
                    k4=render_trainer.config["camera"]["k4"],
                    k5=render_trainer.config["camera"]["k5"],
                    k6=render_trainer.config["camera"]["k6"],
                    p1=render_trainer.config["camera"]["p1"],
                    p2=render_trainer.config["camera"]["p2"]
                )
        else:
            num_frames_in_recording = len(render_trainer.scene_dataset)
    elif render_trainer.config["camera"]["sensor"] == "Realsense":
        sensor = "Realsense"                                                )

    ros_bridge = RosBridge(init_node=True, sim=sim, name="ros_bridge_track", config=render_trainer.config)

    idx = 0
    start = False
    pause = False
    track_times = None

    while(True):
        # read data---------------------------------------------------------
        if live:

            depth_np, im_np, pose_np = ros_bridge.get_camera_data()

            if pose_np is None:
                print("Error, pose is None!")
                continue

            im_np = render_trainer.scene_dataset.rgb_transform(im_np)
            depth_np = render_trainer.scene_dataset.depth_transform(depth_np)

            im_np = im_np[None, ...]
            depth_np = depth_np[None, ...]

        else:
            if pause is False:
                sample = render_trainer.scene_dataset[idx]
                im_np = sample["image"][None, ...]
                depth_np = sample["depth"][None, ...]

                if render_trainer.gt_traj:
                    T_np = sample["T"][None, ...]
                    T = torch.from_numpy(T_np).float().to(
                        render_trainer.device)

        rgb_vis = imgviz.resize(im_np[0], width=400)
        rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB)

        if render_trainer.label_cam:
            mid_y = (rgb_vis.shape[0] - 1) // 2
            mid_x = (rgb_vis.shape[1] - 1) // 2
            cv2.circle(rgb_vis, (mid_x, mid_y), 3, [0, 0, 0], -1)


        ## Pause imap if camera is below z value -- 0.3m
        T_camera = ros_bridge.robot_model.camera_link.worldcoords().T()
        z_camera = T_camera[2,3]

        if not pause and z_camera < 0.4:
            pause = not pause
        if pause and z_camera > 0.4:
            pause = not pause

        cv2.imshow("rgb_vis", rgb_vis)
        key = cv2.waitKey(1)

        key_ = ros_bridge.get_imap_key()
        if key_ is not None:
            key = key_

        if robot_start is not None:
            r_start = get_latest_queue(robot_start)
            if r_start == True:
                key = 115
            elif r_start == False:
                key = 112

        if key == 115:
            start = True
            print("Starting imap...")
        elif key == 112:
            pause = not pause
            print("Pausing imap...")
        elif key == 113:
            break
        if start is False:
            continue

        depth = torch.from_numpy(depth_np).float().to(
            render_trainer.device)
        im = torch.from_numpy(im_np).float().to(
            render_trainer.device) / 255.

        if pause is False:
            # track ---------------------------------------------------------
            if idx == 0:
                if render_trainer.gt_traj:
                    render_trainer.T_WC = T
                else:
                    render_trainer.T_WC = torch.eye(
                        4, device=render_trainer.device).unsqueeze(0)

                render_trainer.init_trajectory(render_trainer.T_WC)

            else:
                im_track = None
                if render_trainer.do_color and render_trainer.track_color:
                    im_track = im

            if ROS:
                render_trainer.T_WC = torch.from_numpy(pose_np).unsqueeze(0).float().to(render_trainer.device)

            # send data to mapping -------------------------------------------
            if idx % 5 == 0:
                IDT = (im, depth, render_trainer.T_WC)
                track_to_map_IDT.put(IDT)

            # send pose to vis -----------------------------------------------
            try:
                track_to_vis_T_WC.put((render_trainer.T_WC,
                                       depth_np, im_np), block=False)
            except queue.Full:
                pass

        # get NN params --------------------------------------------------
        # block before receiving first NN params
        if idx == 0:
            params = map_to_track_params.get()
        else:
            params = get_latest_queue(map_to_track_params)

        if params is not None:
            state_dict, B_layer_dict = params
            render_trainer.fc_occ_map.load_state_dict(state_dict)
            del state_dict
            if B_layer_dict:
                render_trainer.B_layer.load_state_dict(B_layer_dict)
                del B_layer_dict

        if pause is False:
            idx += 1
            if not live and idx == num_frames_in_recording - 1:
                idx = 1

    track_to_map_IDT.put("finish")
    wait_map_exit.wait()
    # release
    params = get_latest_queue(map_to_track_params)
    del params
    del render_trainer
    print("finish track")

def mapping(track_to_map_IDT,
            map_to_track_params,
            map_to_vis_params,
            map_to_vis_kf_depth,
            map_to_vis_active,
            vis_to_map_save_idx,
            vis_to_map_labels,
            vis_to_map_rois,
            wait_map_exit,
            wait_vis_exit,
            gpu_for_process,
            config_file,
            load_path,
            load_epoch,
            show_mesh,
            incremental,
            do_track,
            do_color,
            do_sem,
            sim=False):
    print('map: starting')
    device = gpu_for_process["map"]
    render_trainer = trainer.RenderTrainer(
        device,
        config_file,
        load_path=load_path,
        load_epoch=load_epoch,
        do_mesh=show_mesh,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
        do_sem=do_sem
    )
    init_gpu(gpu_for_process["track"])
    init_gpu(gpu_for_process["map"])
    iters = 0
    read_data = True
    sem_inds_b = None
    sem_freq = 1
    clear_keyframes = False

    step_times = None

    #Keep record of saved rois for naming purposes
    roi_count = 0
    while(True):
        finish_optim = (
            render_trainer.frames_since_add >= render_trainer.optim_frames
        )

        action = get_latest_queue(vis_to_map_actions)
        if action is not None:
            if action == "clear_keyframes":
                clear_keyframes = True
            del action
        if clear_keyframes:
            render_trainer.clear_keyframes()
            read_data = True
            clear_keyframes = False
            sem_inds_b = None

        if finish_optim and read_data is False:
            render_trainer.has_initialised = True
            T_WC = render_trainer.frames.T_WC_track[-1]
            T_WC = T_WC.unsqueeze(0)
            read_data = render_trainer.check_keyframe(T_WC)

            # send kf to vis
            if read_data is False and render_trainer.last_is_keyframe:
                map_to_vis_kf_depth.put(
                    (render_trainer.frames.depth_batch[-1],
                     render_trainer.frames.im_batch[-1],
                        T_WC)
                )

        # get/add data ------------------------------------------------------
        if read_data:
            IDT = get_latest_queue(track_to_map_IDT)

        if IDT is not None:
            if IDT == "finish":
                break

            im, depth, T_WC = IDT
            im = im.to(render_trainer.device)
            depth = depth.to(render_trainer.device)
            T_WC = T_WC.to(render_trainer.device)
            IDT = None
            read_data = False
            # print("added---------------------------------------------------------")

            render_trainer.frame_id += 1
            data = trainer.FrameData()
            data.im_batch = im.clone()
            data.depth_batch = depth.clone()

            # add to remove empty pose when self.gt_traj is True in add_frame()
            data.T_WC_batch = T_WC.clone()
            data.T_WC_batch_np = T_WC.cpu().numpy()
            data.im_batch_np = im.cpu().numpy()
            data.depth_batch_np = depth.cpu().numpy()

            render_trainer.add_frame(data)
            del im
            del depth

            if render_trainer.batch_size == 1:
                render_trainer.init_pose_vars(T_WC.clone())
                render_trainer.last_is_keyframe = True

                # send first kf to vis
                map_to_vis_kf_depth.put(
                    (render_trainer.frames.depth_batch[-1],
                        render_trainer.frames.im_batch[-1],
                     render_trainer.frames.T_WC_track[-1].unsqueeze(0))
                )
            else:
                # render_trainer.add_track_pose(T_WC.clone()) ## stop optimisation of camera poses
                render_trainer.add_track_pose_noopt(T_WC.clone())
                render_trainer.last_is_keyframe = False

            del T_WC

        if render_trainer.frames.depth_batch is not None:
            # build pyramid ---------------------------------------------------
            if iters == 0:
                pyramid_batch = None
                for idx in range(render_trainer.batch_size):
                    pyramid_np = trainer.get_pyramid(
                        render_trainer.frames.depth_batch[idx].cpu().numpy(),
                        render_trainer.n_levels,
                        render_trainer.kernel_init)[None, ...]
                    pyramid = torch.from_numpy(
                        pyramid_np).float().to(render_trainer.device)
                    pyramid_batch = render_trainer.expand_data(
                        pyramid_batch, pyramid)

            if render_trainer.has_initialised is False:
                depth_batch = pyramid_batch[
                    :, render_trainer.pyramid_level, :, :]

            else:
                depth_batch = render_trainer.frames.depth_batch
                pyramid_batch = None

            # joint optim step ------------------------------------------------
            render_trainer.win_idxs = None

            im_batch = None
            if render_trainer.do_color:
                im_batch = render_trainer.frames.im_batch
            loss, step_time = render_trainer.step(
                depth_batch,
                render_trainer.frames.T_WC_track,
                render_trainer.do_information,
                render_trainer.do_fine,
                do_active=True,
                im_batch=im_batch,
                color_scaling=render_trainer.color_scaling
            )

            # read semantic labels --------------------------------------------
            if render_trainer.do_sem:
                labels = get_latest_queue(vis_to_map_labels)
                if labels is not None:
                    sem_steps_click = 0
                    sem_freq = 1
                    sem_inds_b = torch.tensor(
                        labels[0], device=render_trainer.device)
                    sem_inds_h = torch.tensor(
                        labels[1], device=render_trainer.device)
                    sem_inds_w = torch.tensor(
                        labels[2], device=render_trainer.device)
                    sem_classes = torch.tensor(
                        labels[3], device=render_trainer.device)
                    h_labels = torch.tensor(
                        labels[4], device=render_trainer.device,
                        dtype=torch.float
                    )
                    h_masks = torch.tensor(
                        labels[5], device=render_trainer.device)

                    del labels

                # step semantic optim -----------------------------------------
                if sem_inds_b is not None and iters % sem_freq == 0:
                    im_batch = None
                    if do_color:
                        im_batch = render_trainer.frames.im_batch

                    time_sem = render_trainer.step_sem(
                        depth_batch,
                        render_trainer.frames.T_WC_track,
                        render_trainer.do_information,
                        render_trainer.do_fine,
                        sem_inds_b,
                        sem_inds_h,
                        sem_inds_w,
                        sem_classes,
                        h_labels,
                        h_masks,
                        im_batch=im_batch,
                        color_scaling=render_trainer.color_scaling
                    )
                    sem_steps_click += 1
                    if sem_steps_click == 40:
                        sem_freq = 15
                    # print("time_sem: ", time_sem)

                # Save embeddings if roi selected
                rois = get_latest_queue(vis_to_map_rois)
                if rois is not None:
                    b = rois[0][0]
                    x = rois[1]
                    y = rois[2]
                    w = rois[3]
                    h = rois[4]

                    num_rois = len(x)
                    num_points_per_region = 100
                    object_names = ['c1', 's1', 'c2', 's2', 'c3', 's3', 'c4', 's4']
                    for roi_idx in range(num_rois):
                        # sample points
                        x_range = np.arange(x[roi_idx], x[roi_idx] + w[roi_idx])
                        y_range = np.arange(y[roi_idx], y[roi_idx] + h[roi_idx])
                        indices_w, indices_h = np.meshgrid(x_range, y_range, indexing='xy')
                        indices_w = indices_w.reshape(-1)
                        indices_h = indices_h.reshape(-1)
                        # sample 100 random points from roi
                        x_points = np.random.randint(0, indices_w.shape[0], (num_points_per_region,))
                        y_points = np.random.randint(0, indices_h.shape[0], (num_points_per_region,))
                        indices_w = indices_w[x_points]
                        indices_h = indices_h[y_points]
                        indices_b = indices_w.shape[0] * [b]

                        inds_b = torch.tensor(
                            indices_b, dtype=torch.long, device=render_trainer.device)
                        inds_h = torch.tensor(
                            indices_h.tolist(), dtype=torch.long, device=render_trainer.device)
                        inds_w = torch.tensor(
                            indices_w.tolist(), dtype=torch.long, device=render_trainer.device)
                        feats = render_trainer.compute_features_2(depth_batch,
                                                                render_trainer.frames.T_WC_track,
                                                                inds_b,
                                                                inds_h,
                                                                inds_w,
                                                                im_batch=im_batch
                                                                )

                        # flatten
                        feats = feats.cpu().numpy()
                        # save features and corresponding region
                        filename = f"./temp/{object_names[roi_idx]}.npy"
                        np.save(filename, feats)
                        roi_count += 1

            render_trainer.step_pyramid(iters)


            # send NN params to vis -------------------------------------------
            if iters % 5 == 0 and iters != 0:
                try:
                    map_to_vis_active.put(
                        render_trainer.active_inds, block=False)
                except queue.Full:
                    pass

                state_dict = render_trainer.fc_occ_map.state_dict()

                B_layer_dict = None
                if render_trainer.B_layer:
                    B_layer_dict = render_trainer.B_layer.state_dict()

                try:
                    map_to_vis_params.put(
                        (state_dict,
                         B_layer_dict,
                         render_trainer.frames.T_WC_track),
                        block=False
                    )
                except queue.Full:
                    pass

            # send NN params to track -----------------------------------------
            if render_trainer.has_initialised:
                if render_trainer.frames_since_add % 5 == 0 and render_trainer.frames_since_add != 0:
                    state_dict = render_trainer.fc_occ_map.state_dict()
                    B_layer_dict = None
                    if render_trainer.B_layer:
                        B_layer_dict = render_trainer.B_layer.state_dict()

                    map_to_track_params.put(
                        (state_dict,
                         B_layer_dict)
                    )
                    # print("sent params to track")

            iters += 1
            render_trainer.frames_since_add += 1

    map_to_vis_params.put("finish")
    wait_vis_exit.wait()
    # release
    params = get_latest_queue(vis_to_map_save_idx)
    del params
    params = get_latest_queue(vis_to_map_labels)
    del params
    params = get_latest_queue(track_to_map_IDT)
    del params
    del render_trainer
    wait_map_exit.set()

    print("finish map")


def vis(map_to_vis_params,
        track_to_vis_T_WC,
        map_to_vis_kf_depth,
        map_to_vis_active,
        vis_to_map_save_idx,
        vis_to_map_labels,
        vis_to_map_rois,
        wait_vis_exit,
        render_trainer,
        view_freq=1,
        use_scribble=False,
        save_res=False,
        save_dir='live_save/',
        robot_target=None,
        robot_label=None,
        debug_vis=True,
        do_vis_up=True,
        collision_mesh=None,
        sim=False,
        ):

    print('vis: starting')

    # init vars---------------------------------------------------------------
    do_render = True
    do_kf_vis = False
    render_mesh = True
    T_WC_np = None
    update_kfs = False
    do_mesh = False
    do_sem_mesh = False
    toggle_mesh = False
    mask_class = False
    follow = True
    colormap = None
    h_level = None
    param_counter = 0
    render_trainer.kfs_im = []
    render_trainer.kfs_depth = []
    render_trainer.frames.depth_batch_np = []
    render_trainer.frames.im_batch_np = []

    ros_bridge = RosBridge(init_node=True, sim=sim, name="ros_bridge_vis", config=render_trainer.config)

    mesh_geo = None
    mesh_frame = None
    if render_trainer.do_mesh:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=int(render_trainer.W),
                          height=int(render_trainer.H),
                          left=600, top=50)
        view_ctl = vis.get_view_control()
        cam = view_ctl.convert_to_pinhole_camera_parameters()
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            render_trainer.W, render_trainer.H,
            render_trainer.fx, render_trainer.fy,
            render_trainer.cx, render_trainer.cy)

    if render_trainer.do_sem:
        query_selector = automatic_query.AnnotationSelector(
            render_trainer.W, render_trainer.H,
            render_trainer.W_vis, render_trainer.H_vis,
            query_strategy="entropy", split_mode="pixel", pixel_mode="rand",
            top_n_percent=-99, grid_num=16, n_pixels_sampled=1) #n_pixels_sampled=10)
        ilabel = iLabel(render_trainer.do_hierarchical,
                        render_trainer.do_names,
                        render_trainer.label_cam,
                        render_trainer.device,
                        render_trainer.n_classes,
                        query_selector,
                        use_scribble=use_scribble,
                        n_points_per_scribble=render_trainer.config["semantics"]["n_points_per_scribble"],
                        bootstrap=render_trainer.config["semantics"]["bootstrap"],
                        bootstrap_dir=render_trainer.config["dataset"]["bootstrap_dir"],
                        )

    save_count = 0
    frame_count = 0
    save_vis_up_count = 0

    render_trainer.ilabel = ilabel
    render_trainer.save_count = save_count

    if debug_vis:
        cv2.namedWindow('iMAP_debug', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("iMAP_debug", 800, 100)

    cv2.namedWindow('keyframes', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("keyframes", 200, 950)
    if do_vis_up:
        cv2.namedWindow('iMAP', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("iMAP", 100, 100)

    force_exit = False
    if save_res:
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_path = save_dir + time_str
        os.mkdir(save_path)
        os.mkdir(save_path + "/network")
        os.mkdir(save_path + "/poses")
        os.mkdir(save_path + "/keyframes")
        os.mkdir(save_path + "/semantics")
        os.mkdir(save_path + "/live_ims")
        if render_trainer.do_hierarchical:
            os.mkdir(save_path + "/trees")

    while(True):
        if force_exit:
            break
        # add reveived keyframes-----------------------------------------------
        while(True):
            if force_exit:
                break
            try:
                kf_depth, kf_im, T_WC = map_to_vis_kf_depth.get(block=False)

                if kf_depth is not None:

                    T_WC_np = T_WC.cpu().numpy()
                    ros_bridge.create_and_publish_pointcloud( [[]],
                        depth=kf_depth.cpu().numpy(),
                        camera_pose=T_WC.cpu().numpy(),
                        intrinsics=[render_trainer.fx, render_trainer.fy, render_trainer.cx, render_trainer.cy],
                    )

                    # add pose
                    render_trainer.frames.T_WC_batch = render_trainer.expand_data(
                        render_trainer.frames.T_WC_batch,
                        T_WC.to(render_trainer.device),
                        False)
                    del T_WC
                    render_trainer.batch_size = render_trainer.frames.T_WC_batch.shape[0]

                    # add depth
                    kf_depth_np = kf_depth.cpu().numpy()
                    render_trainer.frames.depth_batch_np.append(np.copy(kf_depth_np))

                    kf_depth_np_vis = imgviz.depth2rgb(kf_depth_np)
                    kf_depth_np_vis = cv2.cvtColor(
                        kf_depth_np_vis, cv2.COLOR_BGR2RGB)
                    del kf_depth
                    kf_depth_resize = imgviz.resize(
                        kf_depth_np,
                        width=render_trainer.W_vis,
                        height=render_trainer.H_vis,
                        interpolation="nearest")
                    render_trainer.kfs_depth.append(kf_depth_resize)

                    # add rgb
                    kf_im_np = kf_im.cpu().numpy()
                    render_trainer.frames.im_batch_np.append(np.copy(kf_im_np))

                    del kf_im
                    kf_im_resize = imgviz.resize(
                        kf_im_np,
                        width=int(render_trainer.W_vis), #* 1.5),
                        height=int(render_trainer.H_vis), #* 1.5),
                        interpolation="nearest")
                    kf_im_resize = (kf_im_resize * 255).astype(np.uint8)
                    render_trainer.kfs_im.append(kf_im_resize)

                    # add to semantics
                    if render_trainer.do_sem:
                        label_kf = (kf_im_np * 255).astype(np.uint8)
                        ilabel.add_keyframe(label_kf, frame_count)

                    # add pointcloud
                    if render_trainer.gt_scene is False and render_trainer.do_mesh:
                        pc_gt_cam = trainer.backproject_pcs(
                            kf_depth_resize[None, ...],
                            render_trainer.fx_vis,
                            render_trainer.fy_vis,
                            render_trainer.cx_vis,
                            render_trainer.cy_vis)
                        render_trainer.pcs_gt_cam = render_trainer.expand_data(
                            render_trainer.pcs_gt_cam,
                            pc_gt_cam,
                            replace=False)

                    update_kfs = True
                    frame_count += 1

            except queue.Empty:
                break

        if update_kfs:
            # draw keyframes------ --------------------------------------------
            kfs_im = np.hstack(render_trainer.kfs_im)
            kfs_vis = cv2.cvtColor(kfs_im, cv2.COLOR_BGR2RGB)
            cv2.imshow("keyframes", kfs_vis)
            cv2.waitKey(1)
            if save_res:
                save_pad = f'{save_count:05}'
                file_name = save_path + '/keyframes/keyframe_' + save_pad + '.png'
                label_kf = cv2.cvtColor(label_kf, cv2.COLOR_BGR2RGB)
                cv2.imwrite(file_name, label_kf)

            update_kfs = False

            # recompute scene bounds ------------------------------------------
            if render_trainer.gt_scene is False and render_trainer.do_mesh:
                pc_batch_size = render_trainer.pcs_gt_cam.shape[0]
                pc = trainer.draw_pc(
                    pc_batch_size,
                    render_trainer.pcs_gt_cam,
                    render_trainer.frames.T_WC_batch.cpu().numpy()
                )
                transform, extents = trimesh.bounds.oriented_bounds(pc)
                transform = np.linalg.inv(transform)
                render_trainer.set_scene_bounds(extents, transform)

                if save_res:
                    np.save(save_path + "/bounds_T.npy", transform)
                    np.save(save_path + "/bounds_extents.npy", extents)

        # update NN params -------------------------------------------------
        params = get_latest_queue(map_to_vis_params)
        if params is not None:
            if params == "finish":
                break
            state_dict, B_layer_dict, T_WC_track = params
            render_trainer.fc_occ_map.load_state_dict(state_dict)
            del state_dict
            if B_layer_dict:
                render_trainer.B_layer.load_state_dict(B_layer_dict)
                del B_layer_dict

            render_trainer.frames.T_WC_track = T_WC_track.to(
                render_trainer.device)
            del T_WC_track

            param_counter += 1

        # compute scene mesh -------------------------------------------------
        if render_trainer.grid_pc is not None and do_mesh == True:
            with torch.set_grad_enabled(False):
                alphas, _, sems = occupancy.chunks(
                    render_trainer.grid_pc,
                    render_trainer.chunk_size,
                    render_trainer.fc_occ_map,
                    render_trainer.n_embed_funcs,
                    render_trainer.B_layer,
                    do_sem=mask_class,
                    do_hierarchical=render_trainer.do_hierarchical,
                    to_cpu=mask_class
                )

                occ = render_rays.occupancy_activation(
                    alphas, render_trainer.voxel_size)

                if mask_class:
                    class_idx = ilabel.mouse_callback_params["class"]
                    if render_trainer.do_hierarchical:
                        binary = (torch.round(sems).long())
                        decimal = binary_to_decimal(
                            binary, ilabel.h_level + 1).squeeze(0)
                        decimal += 2**(ilabel.h_level + 1) - 2
                        occ[decimal != class_idx] = 0.
                    else:
                        occ_labels = torch.argmax(sems, dim=1)
                        occ[occ_labels != class_idx] = 0.

                dim = render_trainer.grid_dim
                occ = occ.view(dim, dim, dim)

            fc_occ_map = None
            if render_trainer.do_color:
                fc_occ_map = render_trainer.fc_occ_map

            if render_trainer.do_sem:
                colormap = ilabel.colormap
                h_level = ilabel.h_level

            do_sem = render_trainer.do_sem and (mask_class is False)
            mesh, mesh_sem = trainer.draw_mesh(
                occ,
                0.4,
                render_trainer.scene_scale_np,
                render_trainer.bounds_tranform_np,
                render_trainer.chunk_size,
                render_trainer.B_layer,
                render_trainer.device,
                fc_occ_map,
                render_trainer.n_embed_funcs,
                do_sem=do_sem,
                do_hierarchical=render_trainer.do_hierarchical,
                colormap=colormap,
                h_level=h_level
            )

            if collision_mesh is not None:
                mesh_wrt_world = mesh.copy()
                collision_mesh.put(mesh_wrt_world, block=False)

            mesh = visualisation.draw.trimesh_to_open3d(mesh)
            if mesh_sem is not None:
                mesh_sem = visualisation.draw.trimesh_to_open3d(mesh_sem)

            if mesh_geo is not None:
                mesh_geo.clear()
                reset_bound = False
            else:
                reset_bound = True

            if do_sem and do_sem_mesh:
                mesh_geo = copy.deepcopy(mesh_sem)
            else:
                mesh_geo = copy.deepcopy(mesh)
            vis.add_geometry(mesh_geo, reset_bounding_box=reset_bound)
            T_WC_np = render_trainer.T_WC[0].cpu().numpy()
            np.save('./scene.npy', T_WC_np)
            print("Mesh saved")
            do_mesh = False
            print("do_mesh: ", do_mesh)

        if toggle_mesh and do_sem and mesh_geo is not None:
            mesh_geo.clear()

            if do_sem_mesh:
                mesh_geo = copy.deepcopy(mesh_sem)
            else:
                mesh_geo = copy.deepcopy(mesh)
            vis.add_geometry(mesh_geo, reset_bounding_box=False)

            toggle_mesh = False

        # set 3D scene view ---------------------------------------------------
        if render_trainer.T_WC is not None and render_trainer.do_mesh:
            if follow:
                T_CW_np = render_trainer.T_WC[0].inverse().cpu().numpy()
                cam = view_ctl.convert_to_pinhole_camera_parameters()
                cam.extrinsic = T_CW_np
                view_ctl.convert_from_pinhole_camera_parameters(cam)
                if mesh_frame is not None:
                    mesh_frame.clear()

        if render_mesh and render_trainer.do_mesh:
            vis.poll_events()
            vis.update_renderer()

        # get latest frame and pose ------------------------------------------
        latest_frame = get_latest_queue(track_to_vis_T_WC)
        if latest_frame is not None:
            T_WC, depth_latest_np, im_latest_np = latest_frame
            render_trainer.T_WC = T_WC.clone().to(render_trainer.device)
            del T_WC

            render_trainer.depth_resize = imgviz.resize(
                depth_latest_np[0],
                width=render_trainer.W_vis,
                height=render_trainer.H_vis,
                interpolation="nearest")
            render_trainer.depth_resize_up = imgviz.resize(
                depth_latest_np[0],
                width=render_trainer.W_vis_up,
                height=render_trainer.H_vis_up,
                interpolation="nearest")
            del depth_latest_np

            render_trainer.im_resize = imgviz.resize(
                im_latest_np[0],
                width=render_trainer.W_vis,
                height=render_trainer.H_vis)
            render_trainer.im_resize_up = imgviz.resize(
                im_latest_np[0],
                width=render_trainer.W_vis_up,
                height=render_trainer.H_vis_up)
            del im_latest_np

        # get active samples and compute mask ---------------------------------
        active_inds = get_latest_queue(map_to_vis_active)
        if active_inds is not None:
            render_trainer.active_inds = active_inds.to(render_trainer.device)
            del active_inds

        mask = None
        mask_up = None
        mask_vis = None
        if render_trainer.active_inds is not None:
            inds = render_trainer.active_inds.cpu().numpy()
            mask = mask_from_inds(
                inds,
                render_trainer.H, render_trainer.W,
                render_trainer.H_vis, render_trainer.W_vis
            )
            if debug_vis:
                mask_vis = imgviz.resize(
                    mask,
                    width=render_trainer.W_vis,
                    height=render_trainer.H_vis).astype(bool)
            if do_vis_up:
                mask_up = imgviz.resize(
                    mask,
                    width=render_trainer.W_vis_up,
                    height=render_trainer.H_vis_up).astype(bool)

        # render live view ----------------------------------------------------
        if render_trainer.T_WC is not None and param_counter % view_freq == 0:
            if do_render:
                render_pose = render_trainer.T_WC
                view_depths, view_vars, view_cols, view_sems = trainer.render_vis(
                    1,
                    render_trainer,
                    render_pose,
                    render_trainer.do_fine,
                    do_var=debug_vis,
                    do_color=(render_trainer.do_color and debug_vis),
                    do_sem=(render_trainer.do_sem and debug_vis),
                    do_hierarchical=render_trainer.do_hierarchical,
                    radius=render_trainer.radius_vis,
                    do_mip=render_trainer.do_mip
                )
                if do_vis_up:
                    origins_dirs = render_rays.origin_dirs_W(
                        render_pose, render_trainer.dirs_C_vis_up)
                    view_depth_up, view_color_up, sem_pred_up = trainer.render_vis_up(
                        view_depths,
                        render_pose,
                        render_trainer,
                        origins_dirs,
                        do_color=render_trainer.do_color,
                        do_sem=render_trainer.do_sem,
                        do_hierarchical=render_trainer.do_hierarchical,
                        radius=render_trainer.radius_vis_up,
                        do_mip=render_trainer.do_mip
                    )

                    sem_vis_up = None
                    label_vis_up = None
                    if sem_pred_up is not None:
                        sem_vis_up, label_vis_up, entropy, _ = ilabel.get_vis_sem(
                            sem_pred_up,
                            render_trainer.im_resize_up,
                            get_entropy=True)

                    surface_normals, diffuse = render_rays.render_normals(
                        render_pose,
                        view_depth_up,
                        render_trainer.fc_occ_map,
                        render_trainer.dirs_C_vis_up,
                        render_trainer.n_embed_funcs,
                        origins_dirs,
                        render_trainer.B_layer,
                        noise_std=None,
                        radius=render_trainer.radius_vis_up,
                        do_mip=render_trainer.do_mip

                    )
                    surface_normals = surface_normals.view(
                        render_trainer.H_vis_up,
                        render_trainer.W_vis_up, 3).detach().cpu().numpy()

                    diffuse = diffuse.view(
                        render_trainer.H_vis_up,
                        render_trainer.W_vis_up).detach().cpu().numpy()

                    vis_rgb = render_trainer.im_resize_up

                    entropy = entropy[0]
                    entropy_vis = imgviz.depth2rgb(entropy)

                    if mask_up is not None:
                        vis_rgb = render_trainer.im_resize_up.copy()
                        vis_rgb[mask_up, :] = [123, 0, 0]
                    vis_up = trainer.live_vis_up(
                        surface_normals,
                        diffuse,
                        sem_vis_up,
                        view_color_up,
                        label_vis_up,
                        render_trainer.depth_resize_up,
                        vis_rgb,
                        entropy=entropy_vis,
                        count=save_vis_up_count,
                    )
                    cv2.imshow("iMAP", vis_up)

                    vis_up_file = "/home/data/screen_capture/" + "vis_up_" + str(save_vis_up_count) + ".png"
                    cv2.imwrite(vis_up_file, vis_up)
                    save_vis_up_count += 1

                view_depths = view_depths.cpu().numpy()
                render_trainer.latest_depth = view_depths[0]
                if debug_vis:
                    sem_vis = None
                    label_vis = None
                    entropy = None
                    if view_sems is not None:
                        sem_pred = view_sems[0]
                        sem_vis, label_vis, entropy, _ = ilabel.get_vis_sem(
                            sem_pred, render_trainer.im_resize)

                    if False:
                        col_view = label_vis
                    else:
                        col_view = render_trainer.im_resize

                    if view_cols is not None:
                        col_render = view_cols[0]
                    else:
                        col_render = col_view.copy()
                    if mask_vis is not None:
                        col_render[mask_vis, :] = [123, 0, 0]

                    if render_trainer.label_cam:
                        mid_y = (col_view.shape[0] - 1) // 2
                        mid_x = (col_view.shape[1] - 1) // 2
                        cv2.circle(col_view, (mid_x, mid_y),
                                   2, [255, 255, 255], -1)

                    viz = trainer.live_vis(
                        render_trainer.depth_resize,
                        view_depths[0],
                        view_vars[0],
                        view_cols=col_render,
                        im_batch_np=col_view,
                        sem_ims=sem_vis,
                        entropies=entropy,
                        min_depth=render_trainer.min_depth,
                        max_depth=render_trainer.max_depth
                    )

                    viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)

            # render view and keyboard interface ---------------------------
            if debug_vis:
                cv2.imshow("iMAP_debug", viz)

            key = cv2.waitKey(1)
            key_ = ros_bridge.get_imap_key()
            if key_ is not None:
                key = key_ -2000

            if save_res:
                save_pad = f'{save_count:05}'
                cv2.imwrite(save_path + '/live_ims/viz_' +
                            save_pad + '.png', viz)
                if render_trainer.do_hierarchical:
                    cv2.imwrite(save_path + '/trees/tree_' +
                                save_pad + '.png', ilabel.tree_im)

            param_counter += 1

            if collision_mesh is not None:
                do_mesh = get_latest_queue(collision_mesh)

            if key == 109:
                # m key to generate/update mesh
                do_mesh = True
            elif key == 102:
                # f key to follow camera in 3D vis
                follow = not follow
            elif key == 114:
                # r key to render network
                do_render = not do_render
            elif key == 107:
                # k key to render keyframe
                do_kf_vis = not do_kf_vis
            elif key == 111:
                # o key to render mesh
                render_mesh = not render_mesh
            elif key == 27:
                # ESC key to exit and save mesh to file (useful for running in headless mode)
                # update mesh
                print("Updating mesh before exit")
                with torch.set_grad_enabled(False):
                    alphas, _, sems = occupancy.chunks(
                        render_trainer.grid_pc,
                        render_trainer.chunk_size,
                        render_trainer.fc_occ_map,
                        render_trainer.n_embed_funcs,
                        render_trainer.B_layer,
                        do_sem=False
                    )
                    occ = render_rays.occupancy_activation(
                        alphas, render_trainer.voxel_size)
                    dim = render_trainer.grid_dim
                    occ = occ.view(dim, dim, dim)

                fc_occ_map = None
                if render_trainer.do_color:
                    fc_occ_map = render_trainer.fc_occ_map

                if render_trainer.do_sem:
                    colormap = ilabel.colormap
                    h_level = ilabel.h_level

                occ_mesh = trainer.draw_mesh(
                    occ,
                    0.45,
                    render_trainer.scene_scale_np,
                    render_trainer.bounds_tranform_np,
                    render_trainer.chunk_size,
                    render_trainer.B_layer,
                    render_trainer.device,
                    fc_occ_map,
                    render_trainer.n_embed_funcs,
                    do_sem=render_trainer.do_sem,
                    do_hierarchical=render_trainer.do_hierarchical,
                    colormap=colormap,
                    h_level=h_level
                )

                mesh = visualisation.draw.trimesh_to_open3d(occ_mesh)
                if mesh_geo is not None:
                    mesh_geo.clear()
                mesh_geo = mesh

                T_WC_np = render_trainer.T_WC[0].cpu().numpy()
                np.save('./scene.npy', T_WC_np)
                trimesh.exchange.export.export_mesh(occ_mesh, 'scene.ply')
                print("Mesh saved")
                force_exit = True
            elif key == 98:
                # b key to mask selcted class in mesh
                mask_class = not mask_class
            elif key == 116:
                # t key to toggle mesh color (semantics or rgb)
                do_sem_mesh = not do_sem_mesh
                toggle_mesh = True
            elif key == 122:
                # z key to clear keyframes
                vis_to_map_actions.put("clear_keyframes")
                render_trainer.clear_keyframes_vis()
                ilabel.clear_keyframes()

            # interactive label ------------------------------------------
            if render_trainer.do_sem and do_render:
                labelling_done = False
                while not labelling_done:
                    pixel_clicked, labelling_done = ilabel.label(
                        render_trainer.batch_size,
                        key,
                        vis_to_map_labels,
                        render_trainer,
                        do_kf_vis=do_kf_vis,
                        vis_to_map_rois=vis_to_map_rois,
                        robot_target=robot_target,
                        robot_label=robot_label,
                    )
                # print("save_res: ", save_res)

                if save_res:
                    if pixel_clicked:
                        np.save(save_path + "/semantics/ind_b_" + save_pad,
                                ilabel.mouse_callback_params["indices_b"])
                        np.save(save_path + "/semantics/ind_h_" + save_pad,
                                ilabel.mouse_callback_params["indices_h"])
                        np.save(save_path + "/semantics/ind_w_" + save_pad,
                                ilabel.mouse_callback_params["indices_w"])
                        np.save(save_path + "/semantics/class" + save_pad,
                                ilabel.mouse_callback_params["classes"])

            if save_res:
                np.save(save_path + "/poses/T_" + save_pad + ".npy",
                        render_trainer.T_WC.cpu().numpy())

                if render_trainer.frames.T_WC_track is not None:
                    np.save(save_path + "/poses/kf_Ts_" + save_pad + ".npy",
                            render_trainer.frames.T_WC_track.cpu().numpy())
                kf_select = ilabel.mouse_callback_params["batch_label"]
                class_select = ilabel.mouse_callback_params["class"]
                h_level = ilabel.h_level
                class_kf = np.array([class_select, kf_select, h_level])
                np.save(save_path + "/semantics/class_kf_" + save_pad + ".npy",
                        class_kf)
                with open(save_path + "/semantics/names_" + save_pad + ".txt", "wb") as fp:
                    pickle.dump(ilabel.label_names, fp)

            if save_res:
                chechpoint_file = save_path + "/network/epoch_" + save_pad + ".pth"
                B_layer_dict = render_trainer.B_layer.state_dict()
                torch.save(
                    {
                        "epoch": save_count,
                        "model_state_dict": render_trainer.fc_occ_map.state_dict(),
                        "B_layer_state_dict": B_layer_dict,
                    },
                    chechpoint_file,
                )

            # vis_to_map_save_idx.put(save_count)
            save_count += 1
            render_trainer.save_count += 1

    # release
    params = get_latest_queue(map_to_vis_params)
    del params
    params = get_latest_queue(track_to_vis_T_WC)
    del params
    params = get_latest_queue(map_to_vis_kf_depth)
    del params
    params = get_latest_queue(map_to_vis_active)
    del params
    del render_trainer
    wait_vis_exit.set()
    print("finish vis")

def robot(robot_query_target_T_WC,
        robot_query_label,
        robot_start,
        collision_mesh,
        sim,
        rosbag):

    if sim:
        TIME_SCALE_NEAR = 1.
        TIME_SCALE_FAR = 1.
        TIME_SCALE = 1. #0.5
    else:
        TIME_SCALE_NEAR = 7
        TIME_SCALE_FAR = 8
        TIME_SCALE = TIME_SCALE_NEAR

    print("Starting pybullet simulation...")
    task_interface = ImapTaskInterface(sim=sim)

    mesh = None
    n_label = 0
    if rosbag:
        while True:
            task_interface.real2robot()

            key = task_interface.ros_bridge.get_imap_key()
            if key is not None:
                if key == (108+2000):
                    collision_mesh.put(True, block=False)
                    time.sleep(1.)

                    while type(mesh) != trimesh.base.Trimesh: ## Wait for mesh to be returned
                        mesh = get_latest_queue(collision_mesh)
                        time.sleep(0.1)

                    collision_scene = task_interface.add_collision_mesh(mesh)

                    task_interface.ros_bridge.pause_rosbag(True)
                    while get_latest_queue(robot_query_target_T_WC) is None:
                        time.sleep(0.1)
                    task_interface.ros_bridge.pause_rosbag(False)

                    continue

            imap_label = task_interface.ros_bridge.get_imap_label()
            if imap_label is not None:
                print("imap_label: ", imap_label)
                if imap_label[0] == -99:
                    imap_label[0] = -99
                elif imap_label[0] == 0: ##Ignore table plane
                    imap_label[0] = -99
                else:
                    imap_label[0] = int(imap_label[0] + 49)

                    robot_query_label.put(imap_label, block=False)

                    task_interface.ros_bridge.pause_rosbag(True)
                    while get_latest_queue(robot_query_target_T_WC) is None:
                        time.sleep(0.1)
                    task_interface.ros_bridge.pause_rosbag(False)

                n_label += 1
            time.sleep(0.01)

    task_interface.real2robot()
    task_interface.reset_pose(time_scale=TIME_SCALE)

    eye = task_interface.pi.robot_model.camera_link.worldcoords().T()[:3,3]
    target = [0.5, -0.05, 0.]
    task_interface.look_at_target(target=target, eye=eye, time_scale=TIME_SCALE)

    j_home = task_interface.pi.getj()
    print("j_home: ", j_home)

    ## Start imap
    print("starting ilabel...")
    robot_start.put(True, block=False)
    task_interface.ros_bridge.set_imap_key(115) ## s to start i label
    time.sleep(5.)

    eye = task_interface.pi.robot_model.camera_link.worldcoords().T()
    c_eye = Coordinate().from_matrix(eye)
    c_eye.translate([0.,0.,0.3], wrt="local")
    task_interface.look_at_target(target=target, eye=c_eye.position, time_scale=TIME_SCALE)

    translation = np.eye(4, dtype=float)
    translation[:3,3] = [0.5, 0., 0.]
    rot = Rotation.from_euler('z', 180, degrees=True)
    rotation = np.eye(4, dtype=float)
    rotation[:3,:3] = rot.as_matrix()
    transform = np.matmul(translation, rotation)

    task_interface.real2robot()
    task_interface.movejs([j_home], time_scale=TIME_SCALE, retry=True)

    target_poses = None
    mesh = None
    collision_scene = None
    # n_random = 3
    n_random = 0
    n_target = 0
    while True:
        if target_poses is None:
            target_poses = get_latest_queue(robot_query_target_T_WC)
        else:

            if mesh is None:
                task_interface.ros_bridge.set_imap_key(108+2000) ## l to create mesh
                collision_mesh.put(True, block=False)
                time.sleep(1.)

                while type(mesh) != trimesh.base.Trimesh: ## Wait for mesh to be returned
                    mesh = get_latest_queue(collision_mesh)
                    time.sleep(0.1)

                collision_scene = task_interface.add_collision_mesh(mesh)

            imap_paused = False
            for n_pose in range(0, len(target_poses)):
                target_pose = target_poses[n_pose]
                if type(target_pose) is list:

                    target_pose = target_pose[0]

                    if TOP_DOWN:
                        target_pose_ = np.eye(4, dtype=float)
                        target_pose_[:3,3] = target_pose[:3,3]
                    else:
                        target_pose_ = target_pose
                    c_target = Coordinate().from_matrix(target_pose_)
                    c_target.rotate([0.,np.pi,0.], wrt="local")
                    time.sleep(4.)

                    task_interface.ros_bridge.broadcast_tf_frame(np.copy(c_target.matrix), "target_"+str(n_target), "map")
                    task_interface.ros_bridge.publish_label((list(target_pose[:3,3]) + [0]))
                    label = [int(0+49), n_pose]
                    robot_query_label.put(label, block=False)
                    time.sleep(1.2)
                    target_poses = get_latest_queue(robot_query_target_T_WC)
                    break


                if TOP_DOWN:
                    target_pose_ = np.eye(4, dtype=float)
                    target_pose_[:3,3] = target_pose[:3,3]
                else:
                    target_pose_ = target_pose
                c_target = Coordinate().from_matrix(target_pose_)
                c_target.rotate([0.,np.pi,0.], wrt="local")
                task_interface.ros_bridge.broadcast_tf_frame(np.copy(c_target.matrix), "target_"+str(n_target), "map")


                if target_pose[0,3] < 0.5:
                    TIME_SCALE = TIME_SCALE_NEAR
                else:
                    TIME_SCALE = TIME_SCALE_FAR

                task_interface.pi.setj(j_home)

                if collision_scene is None:
                    obstacles = task_interface._env.object_ids + [task_interface._env.plane, task_interface._env.wall]
                else:
                    obstacles = [collision_scene, task_interface._env.plane, task_interface._env.wall]
                time_start = time.time()

                n_ik_attempts = 9
                pause_offset = -0.25
                c_pause_imap = c_target.copy()
                c_pause_imap.translate([0.,0.,pause_offset], wrt="local") ## Offset for pausing imap

                j_pause_imap = task_interface.pi.solve_ik(
                    c_pause_imap.pose,
                    move_target=task_interface.pi.robot_model.tipLink,
                    n_init=n_ik_attempts, # first attempt is from current state --> use setj if neccessary. Subsequent are random states.
                    thre=None,
                    rthre=[1., 1., 360],
                    rotation_axis='z',
                    validate=True,
                )

                if j_pause_imap is None:
                    c_target.rotate([0.,np.pi,0.], wrt="local") ## Offset for grasping
                    c_pause_imap = c_target.copy()
                    c_pause_imap.translate([0.,0.,pause_offset], wrt="local") ## Offset for pausing imap

                    j_pause_imap = task_interface.pi.solve_ik(
                        c_pause_imap.pose,
                        move_target=task_interface.pi.robot_model.tipLink,
                        n_init=n_ik_attempts, # first attempt is from current state --> use setj if neccessary. Subsequent are random states.
                        thre=None,
                        rthre=[1., 1., 360],
                        rotation_axis='z',
                        validate=True,
                    )

                    if j_pause_imap is None:
                        label = [-99, n_pose]
                        if n_pose == len(target_poses)-1:
                            task_interface.ros_bridge.publish_label((list(target_pose[:3,3]) + [-99]))
                            robot_query_label.put(label, block=False)
                            time.sleep(1.2)
                            target_poses = get_latest_queue(robot_query_target_T_WC)
                        continue

                offset = -0.04
                js_approach = task_interface.move_to_classify(c_target, offset=offset, time_scale=TIME_SCALE, obstacles=obstacles)
                if js_approach is None:
                    label = [-99, n_pose]
                    if n_pose == len(target_poses)-1:
                        task_interface.ros_bridge.publish_label((list(target_pose[:3,3]) + [-99]))
                        robot_query_label.put(label, block=False)
                        time.sleep(1.2)
                        target_poses = get_latest_queue(robot_query_target_T_WC)
                    continue

                time_end= time.time()
                print("Planning took: {0:.3f} secs".format((time_end-time_start)))

                task_interface.real2robot()

                task_interface.movejs(js_approach, time_scale=TIME_SCALE, retry=True)

                label = task_interface.classify(max_dz=-1.1*offset, time_scale=4*TIME_SCALE)
                task_interface.ros_bridge.publish_label((list(target_pose[:3,3]) + [label]))

                if label is None:
                    label = [-99, n_pose]
                else:
                    label = [label + 49, n_pose]

                robot_query_label.put(label, block=False)

                js = task_interface.pi.planj(
                    j_home,
                    obstacles=obstacles,
                    min_distances=None,
                    min_distances_start_goal=None,
                )
                if js is None:
                    js = task_interface.pi.planj(
                        j_pause_imap,
                        obstacles=obstacles,
                        min_distances=None,
                        min_distances_start_goal=None,
                    )
                    if js is None:
                        task_interface.movejs([j_pause_imap], time_scale=TIME_SCALE, retry=True) ## reverse motion back to pause position
                    else:
                        task_interface.movejs(js, time_scale=TIME_SCALE, retry=True)
                    task_interface.movejs([j_home], time_scale=TIME_SCALE, wait=False, retry=False) ## reverse motion back to home
                else:
                    task_interface.movejs(js, time_scale=TIME_SCALE, wait=False, retry=False)

                ## Query for next point to interact
                target_poses = get_latest_queue(robot_query_target_T_WC)

                break

    print("finish robot")

def train(
    save_ims,
    show_freq,
    config_file,
    scene_draw_freq=200,
    checkpoint_freq=None,
    show_pc=False,
    show_mesh=True,
    save_path=None,
    load_path=None,
    load_epoch=None,
    incremental=True,
    do_active=True,
    do_track=True,
    do_color=True,
    do_sem=False,
    live=False,
    use_scribble=False,
    save_res=False,
    save_dir='save_live/',
    debug_vis=True,
    do_vis_up=True,
    sim=False,
    rosbag=False,
):
    n_gpus = torch.cuda.device_count()
    gpu_for_process = {}
    # print("n_gpus ", n_gpus)

    if n_gpus >= 3:
        gpu_for_process["track"] = "cuda:0"
        gpu_for_process["vis"] = "cuda:1"
        gpu_for_process["map"] = "cuda:2"
        view_freq = 1
    elif n_gpus == 2:
        gpu_for_process["track"] = "cuda:0"
        gpu_for_process["vis"] = "cuda:1"
        gpu_for_process["map"] = "cuda:0"
        view_freq = 1
    else:
        gpu_for_process["track"] = "cuda:0"
        gpu_for_process["vis"] = "cuda:0"
        gpu_for_process["map"] = "cuda:0"
        view_freq = 20

    # init trainer-------------------------------------------------------------
    render_trainer = trainer.RenderTrainer(
        gpu_for_process["vis"],
        config_file,
        load_path=load_path,
        load_epoch=load_epoch,
        do_mesh=show_mesh,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
        do_sem=do_sem
    )
    init_gpu(gpu_for_process["track"])
    init_gpu(gpu_for_process["map"])
    init_gpu(gpu_for_process["vis"])
    render_trainer.optimiser = None
    render_trainer.fc_occ_map.eval()
    if render_trainer.B_layer:
        render_trainer.B_layer.eval()

    torch.multiprocessing.set_start_method('spawn')
    track_to_map_IDT = torch.multiprocessing.Queue()
    map_to_track_params = torch.multiprocessing.Queue()
    map_to_vis_params = torch.multiprocessing.Queue(maxsize=1)
    track_to_vis_T_WC = torch.multiprocessing.Queue(maxsize=1)
    map_to_vis_active = torch.multiprocessing.Queue(maxsize=1)
    map_to_vis_kf_depth = torch.multiprocessing.Queue()
    vis_to_map_save_idx = torch.multiprocessing.Queue()
    vis_to_map_labels = torch.multiprocessing.Queue()
    vis_to_map_rois = torch.multiprocessing.Queue()

    wait_map_exit = torch.multiprocessing.Event()
    wait_vis_exit = torch.multiprocessing.Event()

    ## phys-imap
    robot_start = torch.multiprocessing.Queue(maxsize=1)
    robot_query_target_T_WC = torch.multiprocessing.Queue(maxsize=1)
    robot_query_label = torch.multiprocessing.Queue(maxsize=1)
    collision_mesh = torch.multiprocessing.Queue(maxsize=1)

    track_p = torch.multiprocessing.Process(
        target=tracking, args=(
            (track_to_map_IDT),
            (map_to_track_params),
            (track_to_vis_T_WC),
            (wait_map_exit),
            gpu_for_process,
            (config_file),
            (load_path),
            (load_epoch),
            (show_mesh),
            (incremental),
            (do_track),
            (do_color),
            (do_sem),
            (live),
            (robot_start),
            (sim),
            (rosbag),
        ))
    map_p = torch.multiprocessing.Process(
        target=mapping, args=(
            (track_to_map_IDT),
            (map_to_track_params),
            (map_to_vis_params),
            (map_to_vis_kf_depth),
            (map_to_vis_active),
            (vis_to_map_save_idx),
            (vis_to_map_labels),
            (vis_to_map_rois),
            (wait_map_exit),
            (wait_vis_exit),
            gpu_for_process,
            (config_file),
            (load_path),
            (load_epoch),
            (show_mesh),
            (incremental),
            (do_track),
            (do_color),
            (do_sem),
            (sim),
        ))

    robot_p = torch.multiprocessing.Process(
        target=robot, args=(
            (robot_query_target_T_WC),
            (robot_query_label),
            (robot_start),
            (collision_mesh),
            (sim),
            (rosbag),
        )
    )

    track_p.start()
    map_p.start()
    robot_p.start()

    vis(map_to_vis_params,
        track_to_vis_T_WC,
        map_to_vis_kf_depth,
        map_to_vis_active,
        vis_to_map_save_idx,
        vis_to_map_labels,
        vis_to_map_rois,
        wait_vis_exit,
        render_trainer,
        view_freq=view_freq,
        use_scribble=use_scribble,
        save_res=save_res,
        save_dir=save_dir,
        robot_target=robot_query_target_T_WC,
        robot_label=robot_query_label,
        debug_vis=debug_vis,
        do_vis_up=do_vis_up,
        collision_mesh=collision_mesh,
        sim=sim,
    )

    robot_p.terminate()
    map_p.terminate()
    track_p.terminate()
    track_p.join()
    map_p.join()
    robot_p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Scene SLAM.')
    parser.add_argument('--config', type=str, help='input json config')
    parser.add_argument('-ni',
                        '--no_incremental',
                        action='store_false',
                        help='disable incremental SLAM option')
    parser.add_argument('-na',
                        '--no_active',
                        action='store_false',
                        help='disable active sampling option')
    parser.add_argument('-nt',
                        '--no_track',
                        action='store_false',
                        help='disable camera tracking')
    parser.add_argument('-nc',
                        '--no_color',
                        action='store_false',
                        help='disable color optimisation')
    parser.add_argument('-l',
                        '--live',
                        action='store_true',
                        help='live camera operation')
    parser.add_argument('-s',
                        '--do_sem',
                        action='store_true',
                        help='do semantics')
    parser.add_argument('--use_scribble',
                        action='store_true',
                        help='Use freehand scribble to label objects')
    parser.add_argument('-m',
                        '--do_mesh',
                        action='store_true',
                        help='do meshing')
    parser.add_argument('-nd',
                        '--no_debug',
                        action='store_false',
                        help='disable debug visualisation')
    parser.add_argument('-nn',
                        '--no_normal',
                        action='store_false',
                        help='disable normal visualisation')
    parser.add_argument('-sr',
                        '--save',
                        action='store_true',
                        help='save results')
    parser.add_argument('--save_dir',
                        type=str,
                        default='save_live/',
                        help='Directory where live results get saved.')
    parser.add_argument('--sim',
                        action='store_true',
                        help='Running in simulation.')
    parser.add_argument('--rosbag',
                        action='store_true',
                        help='data replayed from rosbag')
    args = parser.parse_args()

    config_file = args.config
    incremental = args.no_incremental
    do_active = args.no_active
    do_track = args.no_track
    do_color = args.no_color
    live = args.live
    do_sem = args.do_sem
    use_scribble = args.use_scribble
    save = args.save
    save_dir = args.save_dir
    show_mesh = args.do_mesh
    debug_vis = args.no_debug
    do_vis_up = args.no_normal
    sim = args.sim
    rosbag = args.rosbag

    load = False
    show_pc = False
    save_ims = False
    show_freq = 100
    scene_draw_freq = 200
    checkpoint_freq = 100

    train(
        save_ims,
        show_freq,
        config_file,
        scene_draw_freq=scene_draw_freq,
        checkpoint_freq=checkpoint_freq,
        show_pc=show_pc,
        show_mesh=show_mesh,
        incremental=incremental,
        do_active=do_active,
        do_track=do_track,
        do_color=do_color,
        do_sem=do_sem,
        live=live,
        use_scribble=use_scribble,
        save_res=save,
        save_dir=save_dir,
        debug_vis=debug_vis,
        do_vis_up=do_vis_up,
        sim=sim,
        rosbag=rosbag,
    )
