#!/usr/bin/env python
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
from scipy import ndimage

from RobEx.train import trainer
from RobEx import visualisation
from RobEx.render import render_rays
from RobEx.mapping import occupancy
from RobEx.kinect_recorder import reader
from RobEx.gui import utils


def mask_from_inds(inds, H, W, H_vis, W_vis):
    inds_mask = inds[0] == inds[0, -1]
    h_inds = inds[1, inds_mask]
    w_inds = inds[2, inds_mask]
    mask = np.zeros([H, W])
    mask[h_inds, w_inds] = 1

    mask = ndimage.binary_dilation(
        mask, iterations=9)
    mask = (mask * 255).astype(np.uint8)

    mask = imgviz.resize(
        mask, width=W_vis, height=H_vis).astype(bool)

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
             track_to_vis_request_new_kf,
             track_to_vis_save_signal,
             vis_to_track_segmentation,
             track_to_map_labels,
             device,
             config_file,
             show_mesh,
             incremental,
             do_track,
             do_color):

    print('track: starting')
    render_trainer = trainer.RenderTrainer(
        device,
        config_file,
        load_path=None,
        load_epoch=None,
        do_mesh=show_mesh,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
        do_sem=True
    )
    render_trainer.optimiser = None
    sensor = None
    data_reader = None

    if render_trainer.config["camera"]["sensor"] == 'Azure':
        sensor = 'Azure'

        data_reader = reader.DataReaderServer(
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
            p2=render_trainer.config["camera"]["p2"],
            port=8485
        )

    elif render_trainer.config["camera"]["sensor"] == 'Realsense':
        sensor = 'Realsense'

        data_reader = reader.DataReaderServer(
            w=render_trainer.config["camera"]["w"],
            h=render_trainer.config["camera"]["h"],
            fx=render_trainer.config["camera"]["fx"],
            fy=render_trainer.config["camera"]["fy"],
            cx=render_trainer.config["camera"]["cx"],
            cy=render_trainer.config["camera"]["cy"],
            k1=0,
            k2=0,
            k3=0,
            k4=0,
            k5=0,
            k6=0,
            p1=0,
            p2=0,
            port=8485
        )

    idx = 0
    pause = False
    track_times = None

    while (True):
        # read data---------------------------------------------------------
        if sensor == 'Azure':
            data = data_reader.get(
                mw=render_trainer.mw, mh=render_trainer.mh)
        elif sensor == 'Realsense':
            data = data_reader.get()

        if data is None:
            break

        depth_np, im_np, mouse_callback_params, keyframe_idx, request_new_kf, save_mesh_flag = data

        track_to_vis_save_signal.put((save_mesh_flag))

        # if new mouse_callback_params then push to track_to_map_labels
        if mouse_callback_params is not None:
            indices_b = [keyframe_idx] * len(mouse_callback_params["indices_h"])
            track_to_map_labels.put(
                (indices_b,
                 mouse_callback_params["indices_h"],
                 mouse_callback_params["indices_w"],
                 mouse_callback_params["classes"],
                 mouse_callback_params["h_labels"],
                 mouse_callback_params["h_masks"])
            )
        track_to_vis_request_new_kf.put((request_new_kf))

        im_np = render_trainer.scene_dataset.rgb_transform(im_np)
        depth_np = render_trainer.scene_dataset.depth_transform(depth_np)
        im_np = im_np[None, ...]
        depth_np = depth_np[None, ...]

        depth = torch.from_numpy(depth_np).float().to(
            render_trainer.device)
        im = torch.from_numpy(im_np).float().to(
            render_trainer.device) / 255.

        if pause is False:
            # track ---------------------------------------------------------
            if idx == 0:
                render_trainer.T_WC = torch.eye(
                    4, device=render_trainer.device).unsqueeze(0)
                render_trainer.init_trajectory(render_trainer.T_WC)
            else:
                im_track = None
                if render_trainer.do_color and render_trainer.track_color:
                    im_track = im
                track_time = render_trainer.track(depth,
                                                  render_trainer.do_information,
                                                  render_trainer.do_fine,
                                                  depth_np,
                                                  im_track)

                if track_times is None:
                    track_times = np.array([track_time])
                else:
                    track_times = np.append(track_times, track_time)

            # send data to mapping -------------------------------------------
            if idx % 10 == 0:
                IDT = (im, depth, render_trainer.T_WC)
                track_to_map_IDT.put(IDT)

            # send pose to vis -----------------------------------------------
            try:
                track_to_vis_T_WC.put((render_trainer.T_WC,
                                       depth_np, im_np), block=False)

            except queue.Full:
                pass

            # Send results back to client
        results = get_latest_queue(vis_to_track_segmentation)
        if results is not None:
            # send result back to client
            send_success = data_reader.send(results)
            if not send_success:
                del results
                break

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
            render_trainer.B_layer.load_state_dict(B_layer_dict)
            del B_layer_dict

        if pause is False:
            idx += 1
            if idx == len(render_trainer.scene_dataset):
                idx = 0

    track_to_map_IDT.put("finish")
    print("finish track")


def mapping(track_to_map_IDT,
            map_to_track_params,
            map_to_vis_params,
            map_to_vis_kf_depth,
            map_to_vis_active,
            map_to_vis_kill,
            track_to_map_labels,
            device,
            config_file,
            show_mesh,
            incremental,
            do_track,
            do_color):

    print('map: starting')
    render_trainer = trainer.RenderTrainer(
        device,
        config_file,
        load_path=None,
        load_epoch=None,
        do_mesh=show_mesh,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
        do_sem=True
    )
    iters = 0
    read_data = True
    sem_inds_b = None
    sem_freq = 1

    while(True):
        # check if keyframe --------------------------------------------------
        finish_optim = (
            render_trainer.frames_since_add == render_trainer.optim_frames
        )

        if finish_optim:
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
                del IDT
                break

            im, depth, T_WC = IDT
            im = im.to(render_trainer.device)
            depth = depth.to(render_trainer.device)
            T_WC = T_WC.to(render_trainer.device)
            IDT = None
            read_data = False

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
                render_trainer.add_track_pose(T_WC.clone())
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
            loss, step_time = render_trainer.step(
                depth_batch,
                render_trainer.frames.T_WC_track,
                render_trainer.do_information,
                render_trainer.do_fine,
                do_active=True,
                im_batch=render_trainer.frames.im_batch,
                color_scaling=render_trainer.color_scaling
            )

            # read semantic labels --------------------------------------------
            labels = get_latest_queue(track_to_map_labels)
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
                    im_batch=render_trainer.frames.im_batch,
                    color_scaling=render_trainer.color_scaling
                )
                sem_steps_click += 1
                if sem_steps_click == 60:
                    sem_freq = 15

            render_trainer.step_pyramid(iters)

            # send NN params to vis -------------------------------------------
            if iters % 5 == 0 and iters != 0:
                try:
                    map_to_vis_active.put(
                        render_trainer.active_inds, block=False)
                except queue.Full:
                    pass

                state_dict = render_trainer.fc_occ_map.state_dict()
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
                if iters % 5 == 0:
                    state_dict = render_trainer.fc_occ_map.state_dict()
                    B_layer_dict = render_trainer.B_layer.state_dict()

                    map_to_track_params.put(
                        (state_dict,
                         B_layer_dict)
                    )

            iters += 1
            render_trainer.frames_since_add += 1

    print("finish map")

    # send terminate signal to vis
    map_to_vis_kill.put((True))

def vis(map_to_vis_params,
        track_to_vis_T_WC,
        track_to_vis_request_new_kf,
        track_to_vis_save_signal,
        vis_to_track_segmentation,
        map_to_vis_kf_depth,
        map_to_vis_active,
        map_to_vis_kill,
        render_trainer,
        view_freq=1,
        use_scribble=True,
        save_dir='live_save/'
        ):

    print('vis: starting')

    # init vars---------------------------------------------------------------
    do_render = True
    do_kf_vis = False
    render_mesh = True
    T_WC_np = None
    update_kfs = False
    do_mesh = False
    follow = True
    h_level = None
    param_counter = 0
    render_trainer.kfs_im = []
    render_trainer.kfs_depth = []

    mesh_geo = None
    mesh_frame = None
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(render_trainer.W),
                      height=int(render_trainer.H),
                      left=600, top=50,
                      visible=False)
    view_ctl = vis.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        render_trainer.W, render_trainer.H,
        render_trainer.fx, render_trainer.fy,
        render_trainer.cx, render_trainer.cy)

    # generate colourmap
    colormap = torch.tensor(
        [[255, 0, 0],
         [0, 255, 0],
         [0, 0, 255],
         [255, 255, 0],
         [240, 163, 255],
         [0, 117, 220],
         [153, 63, 0],
         [76, 0, 92],
         [25, 25, 25],
         [0, 92, 49],
         [43, 206, 72],
         [255, 204, 153],
         [128, 128, 128],
         [148, 255, 181],
         [143, 124, 0],
         [157, 204, 0],
         [194, 0, 136],
         [0, 51, 128],
         [255, 164, 5],
         [255, 168, 187],
         [66, 102, 0],
         [255, 0, 16],
         [94, 241, 242],
         [0, 153, 143],
         [224, 255, 102],
         [116, 10, 255],
         [255, 255, 128],
         [255, 80, 5]],
        device=render_trainer.device
    )

    colormap_np = colormap.cpu().numpy().astype(np.uint8)
    colormap_np = colormap_np.reshape(-1, 3)

    # keep track of keyframes
    label_kf_store = []
    last_keyframe_sent = -1
    keyframe_idx = 0

    latest_image = None
    force_kill = False
    while(True):
        if force_kill:
            break
        # add reveived keyframes-----------------------------------------------
        while(True):
            if force_kill:
                break
            try:
                try:
                    exit_signal = map_to_vis_kill.get(block=False, timeout=None)
                    if exit_signal:
                        force_kill = True
                        break
                except queue.Empty:
                    pass
                save_mesh = track_to_vis_save_signal.get(block=False, timeout=None)
                if save_mesh:
                    del save_mesh
                    print("Saving mesh")
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
                        # occ_labels = torch.argmax(sems, dim=1)
                        # occ[occ_labels != 0] = 0.
                        dim = render_trainer.grid_dim
                        occ = occ.view(dim, dim, dim)

                    fc_occ_map = None
                    if render_trainer.do_color:
                        fc_occ_map = render_trainer.fc_occ_map

                    occ_mesh = trainer.draw_mesh(
                        occ,
                        0.45,
                        render_trainer.scene_scale_np,
                        render_trainer.bounds_tranform_np,
                        render_trainer.chunk_size,
                        render_trainer.B_layer,
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
                    np.save(save_dir + 'scene.npy', T_WC_np)
                    trimesh.exchange.export.export_mesh(occ_mesh, save_dir + 'scene.ply')
                    print("Mesh saved")
                    do_mesh = False

                # Get callback params
                kf_depth, kf_im, T_WC = map_to_vis_kf_depth.get(block=False)
                if kf_depth is not None:
                    # add pose
                    render_trainer.frames.T_WC_batch = render_trainer.expand_data(
                        render_trainer.frames.T_WC_batch,
                        T_WC.to(render_trainer.device),
                        False)
                    del T_WC
                    render_trainer.batch_size = render_trainer.frames.T_WC_batch.shape[0]

                    # add depth
                    kf_depth_np = kf_depth.cpu().numpy()
                    del kf_depth
                    kf_depth_resize = imgviz.resize(
                        kf_depth_np,
                        width=render_trainer.W_vis,
                        height=render_trainer.H_vis,
                        interpolation="nearest")
                    render_trainer.kfs_depth.append(kf_depth_resize)

                    # add rgb
                    kf_im_np = kf_im.cpu().numpy()
                    del kf_im

                    label_kf = (kf_im_np * 255).astype(np.uint8)
                    # Update kf store
                    label_kf_store.append(cv2.cvtColor(label_kf, cv2.COLOR_BGR2RGB))

                    kf_im_resize = imgviz.resize(
                        kf_im_np,
                        width=render_trainer.W_vis,
                        height=render_trainer.H_vis,
                        interpolation="nearest")

                    kf_im_resize = (kf_im_resize * 255).astype(np.uint8)
                    render_trainer.kfs_im.append(kf_im_resize)

                    # add pointcloud
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

            except queue.Empty:
                break

        if update_kfs:
            update_kfs = False

            # recompute scene bounds ------------------------------------------
            pc = trainer.draw_pc(
                render_trainer.batch_size,
                render_trainer.pcs_gt_cam,
                render_trainer.frames.T_WC_batch.cpu().numpy()
            )
            transform, extents = trimesh.bounds.oriented_bounds(pc)
            transform = np.linalg.inv(transform)
            render_trainer.set_scene_bounds(extents, transform)

        # update NN params -------------------------------------------------
        params = get_latest_queue(map_to_vis_params)
        if params is not None:
            # print("got params")
            state_dict, B_layer_dict, T_WC_track = params
            render_trainer.fc_occ_map.load_state_dict(state_dict)
            del state_dict
            render_trainer.B_layer.load_state_dict(B_layer_dict)
            del B_layer_dict

            render_trainer.frames.T_WC_track = T_WC_track.to(
                render_trainer.device)
            del T_WC_track

            param_counter += 1

        # compute scene mesh -------------------------------------------------
        if render_trainer.grid_pc is not None and do_mesh:

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

            occ_mesh = trainer.draw_mesh(
                occ,
                0.45,
                render_trainer.scene_scale_np,
                render_trainer.bounds_tranform_np,
                render_trainer.chunk_size,
                render_trainer.B_layer,
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
                reset_bound = False
            else:
                reset_bound = True

            mesh_geo = mesh
            vis.add_geometry(mesh_geo, reset_bounding_box=reset_bound)
            do_mesh = False

        # set 3D scene view ---------------------------------------------------
        if render_trainer.T_WC is not None:
            if follow:
                T_CW_np = render_trainer.T_WC[0].inverse().cpu().numpy()
                cam = view_ctl.convert_to_pinhole_camera_parameters()
                cam.extrinsic = T_CW_np
                view_ctl.convert_from_pinhole_camera_parameters(cam)
                if mesh_frame is not None:
                    mesh_frame.clear()

            else:
                T_WC_np = render_trainer.T_WC[0].cpu().numpy()
                if mesh_frame is not None:
                    mesh_frame.clear()
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.2)
                mesh_frame.transform(T_WC_np)
                vis.add_geometry(mesh_frame, reset_bounding_box=False)

        if render_mesh:
            vis.poll_events()
            vis.update_renderer()

        # get latest frame and pose ------------------------------------------
        latest_frame = get_latest_queue(track_to_vis_T_WC)
        if latest_frame is not None:
            T_WC, depth_latest_np, im_latest_np = latest_frame
            render_trainer.T_WC = T_WC.clone().to(render_trainer.device)
            del T_WC
            del latest_frame

            render_trainer.depth_resize = imgviz.resize(
                depth_latest_np[0],
                width=render_trainer.W_vis,
                height=render_trainer.H_vis,
                interpolation="nearest")
            del depth_latest_np

            render_trainer.im_resize = imgviz.resize(
                im_latest_np[0],
                width=render_trainer.W_vis,
                height=render_trainer.H_vis)

            latest_image = np.copy(im_latest_np[0])
            del im_latest_np

        # get active samples and compute mask ---------------------------------
        active_inds = get_latest_queue(map_to_vis_active)
        if active_inds is not None:
            render_trainer.active_inds = active_inds
            del active_inds

        mask = None

        # render live view ----------------------------------------------------
        if render_trainer.T_WC is not None and param_counter % view_freq == 0:
            if do_render:
                view_depths, view_vars, view_cols, view_sems = trainer.render_vis(
                    1,
                    render_trainer,
                    render_trainer.T_WC,
                    render_trainer.do_fine,
                    do_var=True,
                    do_color=render_trainer.do_color,
                    do_sem=render_trainer.do_sem,
                    do_hierarchical=render_trainer.do_hierarchical
                )


                if mask is not None:
                    col = view_cols[0]
                    col[mask, :] = [123, 0, 0]


                # Add result to queue
                try:
                    # render_trainer
                    # send segmentation result to vis if in server mode
                    if latest_image is not None and len(label_kf_store) > 0:
                        try:
                            label_im = torch.argmax(view_sems[0], axis=2).cpu().numpy().astype(np.uint8)

                            # Convert sem_pred to mask
                            sem_mask = cv2.resize(label_im, (latest_image.shape[1], latest_image.shape[0]), cv2.INTER_NEAREST)
                            seg_image = utils.visualise_segmentation(latest_image, sem_mask, colormap_np)
                            request_new_kf = track_to_vis_request_new_kf.get(block=False, timeout=None)

                            if request_new_kf:
                                keyframe_idx = len(label_kf_store) - 1
                                del request_new_kf

                            if keyframe_idx != last_keyframe_sent:
                                last_keyframe_sent = keyframe_idx
                                seg_result = np.concatenate((seg_image, np.copy(label_kf_store[keyframe_idx])), axis=-1)
                                vis_to_track_segmentation.put(np.copy(seg_result))
                            else:
                                vis_to_track_segmentation.put(np.copy(seg_image))

                            latest_image = None
                        except queue.Empty:
                            print("keyframe idx is empty")
                            pass

                except queue.Full:
                    print("queue is full")
                    pass

    print("Vis finished")


def train(
        device,
        config_file,
        show_mesh=True,
        incremental=True,
        do_track=True,
        do_color=True,
        multi_gpu=False,
        use_scribble=False,
        save_dir='./save_live/'
):
    if multi_gpu:
        render_device = "cuda:1"
        main_device = device
        view_freq = 1
    else:
        main_device = device
        render_device = device
        view_freq = 1

    # init trainer-------------------------------------------------------------
    render_trainer = trainer.RenderTrainer(
        render_device,
        config_file,
        load_path=None,
        load_epoch=None,
        do_mesh=show_mesh,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
        do_sem=True
    )
    render_trainer.optimiser = None
    render_trainer.fc_occ_map.eval()
    render_trainer.B_layer.eval()

    torch.multiprocessing.set_start_method('spawn')
    track_to_map_IDT = torch.multiprocessing.Queue()
    map_to_track_params = torch.multiprocessing.Queue()
    map_to_vis_params = torch.multiprocessing.Queue(maxsize=1)
    track_to_vis_T_WC = torch.multiprocessing.Queue(maxsize=1)
    map_to_vis_active = torch.multiprocessing.Queue(maxsize=1)
    vis_to_track_segmentation = torch.multiprocessing.Queue(maxsize=1)
    track_to_vis_save_signal = torch.multiprocessing.Queue(maxsize=1)
    map_to_vis_kf_depth = torch.multiprocessing.Queue()
    track_to_map_labels = torch.multiprocessing.Queue()
    track_to_vis_request_new_kf = torch.multiprocessing.Queue()
    map_to_vis_kill = torch.multiprocessing.Queue(maxsize=1)


    track_p = torch.multiprocessing.Process(
        target=tracking, args=(
            (track_to_map_IDT),
            (map_to_track_params),
            (track_to_vis_T_WC),
            (track_to_vis_request_new_kf),
            (track_to_vis_save_signal),
            (vis_to_track_segmentation),
            (track_to_map_labels),
            (main_device),
            (config_file),
            (show_mesh),
            (incremental),
            (do_track),
            (do_color)
        ))


    map_p = torch.multiprocessing.Process(
        target=mapping, args=(
            (track_to_map_IDT),
            (map_to_track_params),
            (map_to_vis_params),
            (map_to_vis_kf_depth),
            (map_to_vis_active),
            (map_to_vis_kill),
            (track_to_map_labels),
            (main_device),
            (config_file),
            (show_mesh),
            (incremental),
            (do_track),
            (do_color),
        ))

    track_p.start()
    map_p.start()
    vis(map_to_vis_params,
        track_to_vis_T_WC,
        track_to_vis_request_new_kf,
        track_to_vis_save_signal,
        vis_to_track_segmentation,
        map_to_vis_kf_depth,
        map_to_vis_active,
        map_to_vis_kill,
        render_trainer,
        view_freq=view_freq,
        use_scribble=use_scribble,
        save_dir=save_dir)

    print ("Terminating threads")

    track_to_map_IDT.close()
    map_to_track_params.close()
    map_to_vis_params.close()
    track_to_vis_T_WC.close()
    map_to_vis_active.close()
    vis_to_track_segmentation.close()
    track_to_vis_save_signal.close()
    map_to_vis_kf_depth.close()
    track_to_map_labels.close()
    track_to_vis_request_new_kf.close()

    map_p.terminate()
    track_p.terminate()
    track_p.join()
    map_p.join()
    print ("Threads terminated")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

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
    parser.add_argument('-sg',
                        '--single_gpu',
                        action='store_false',
                        help='single GPU operation')
    parser.add_argument('--use_scribble',
                        action='store_true',
                        help='Use freehand scribble to label objects')
    parser.add_argument('--save_dir',
                        type=str,
                        default='save_live/',
                        help='Directory where live results get saved.')
    parser.add_argument('-d',
                        '--device',
                        required=True,
                        type=str,
                        default='cuda:0',
                        help='Device on which to run main GUI visualisation thread.')
    args = parser.parse_args()

    config_file = args.config
    incremental = args.no_incremental
    do_active = args.no_active
    do_track = args.no_track
    do_color = args.no_color
    multi_gpu = args.single_gpu
    use_scribble = args.use_scribble
    save_dir = args.save_dir
    device = args.device

    show_mesh = True

    train(
        device,
        config_file,
        show_mesh=show_mesh,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
        multi_gpu=multi_gpu,
        use_scribble=use_scribble,
        save_dir=save_dir
    )
