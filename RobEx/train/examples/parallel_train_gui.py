#!/usr/bin/env python
import torch
import sys
# from torch.utils.data import DataLoader
import cv2
import configargparse
import queue
import open3d as o3d
import numpy as np
import imgviz
import trimesh
from scipy import ndimage

from RobEx.train import trainer
from RobEx import visualisation
from RobEx.render import render_rays
from RobEx.mapping import occupancy
from RobEx.kinect_recorder import reader
from RobEx.gui import utils

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QObject

pg.setConfigOptions(imageAxisOrder='row-major')

#--------------------__Globals-----------------------
parser = configargparse.ArgParser()
parser.add_argument('--gui_config', required=False, is_config_file=True, default='./gui_configs/gui_config.txt', help='config file path')
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
parser.add_argument('-d',
                    '--device',
                    required=True,
                    type=str,
                    help='Device on which to run main GUI visualisation thread.')
parser.add_argument('-sen',
                    '--sensor',
                    required=True,
                    type=str,
                    help='Type of sensor: [Azure, Realsense, None (for sim data)]')
parser.add_argument('-cl',
                    '--num_classes',
                    required=True,
                    type=int,
                    help='Number of semantic classes.')
parser.add_argument('-pts',
                    '--points_per_scribble',
                    required=True,
                    type=int,
                    help='Number of points to sample from each scribble.')

args = parser.parse_args()

# Mouse parameters
mouse_callback_params = {}
mouse_callback_params["indices_w"] = []
mouse_callback_params["indices_h"] = []
mouse_callback_params["indices_b"] = []
mouse_callback_params["h_labels"] = []
mouse_callback_params["h_masks"] = []
mouse_callback_params["classes"] = []
mouse_callback_params["class"] = 0
mouse_callback_params["batch_label"] = 0

#Scribble specific
mouse_callback_params["drawing"] = False
mouse_callback_params["new_scribble"] = False
mouse_callback_params["request_new_kf"] = False
mouse_callback_params["n_points_per_scribble"] = 12  # Number of points to sample from each scribble

#Scribble params
scribble_params = {}
# Scribble parameters
scribble_params["indices_b"] = []
scribble_params["indices_w"] = []
scribble_params["indices_h"] = []
scribble_params["class"] = []

request_new_keyframe = False
request_previous_keyframe = False
force_exit = False
#----------------------------------------------------


def tracking(track_to_map_IDT,
             map_to_track_params,
             track_to_vis_T_WC,
             device,
             config_file,
             load_path,
             load_epoch,
             show_mesh,
             incremental,
             do_track,
             do_color,
             do_sem,
             live=False):
    print('track: starting')

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

    render_trainer.optimiser = None
    sensor = None

    if render_trainer.config["camera"]["sensor"] == 'Azure':
        sensor = 'Azure'
        if live:
            cfg = render_trainer.config["kinect"]["config_file"]
            kinect_config = o3d.io.read_azure_kinect_sensor_config(cfg)
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
                p2=render_trainer.config["camera"]["p2"],
                undistort=False
            )
    elif render_trainer.config["camera"]["sensor"] == 'Realsense':
        sensor = 'Realsense'
        if live:
            data_reader = reader.RealsenseReaderLight(
                w=render_trainer.config["camera"]["w"],
                h=render_trainer.config["camera"]["h"],
                align_depth_to_color=True
            )

    idx = 0
    start = True
    pause = False
    track_times = None

    while (True):
        # read data---------------------------------------------------------
        if live:
            data = data_reader.get(
                mw=render_trainer.mw, mh=render_trainer.mh)
            if data is None:
                continue
            depth_np, im_np = data

            im_np = render_trainer.scene_dataset.rgb_transform(im_np)
            depth_np = render_trainer.scene_dataset.depth_transform(depth_np)
            im_np = im_np[None, ...]
            depth_np = depth_np[None, ...]
        else:
            if pause is False:
                sample = render_trainer.scene_dataset[idx]
                im_np = sample["image"][None, ...]
                depth_np = sample["depth"][None, ...]

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

        # print("frame ", idx)
        if pause is False:
            idx += 1

    track_to_map_IDT.put("finish")
    print("finish track")


def mapping(track_to_map_IDT,
            map_to_track_params,
            map_to_vis_params,
            map_to_vis_kf_depth,
            map_to_vis_active,
            vis_to_map_save_idx,
            vis_to_map_labels,
            device,
            config_file,
            load_path,
            load_epoch,
            show_mesh,
            incremental,
            do_track,
            do_color,
            do_sem):
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
        do_sem=do_sem
    )
    iters = 0
    read_data = True
    sem_inds_b = None
    sem_freq = 1

    while (True):
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

            # add to remove empty pose when gt_traj is True in add_frame()
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
                    # print("time_sem: ", time_sem)

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
                    # print("sent params to track")

            iters += 1
            render_trainer.frames_since_add += 1

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


class Train(QObject):
    # setup emitters
    changeSegImage = pyqtSignal(QtGui.QImage)
    changeKeyframe = pyqtSignal(QtGui.QImage)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    config_file = args.config
    incremental = args.no_incremental
    do_active = args.no_active
    do_track = args.no_track
    do_color = args.no_color
    multi_gpu = args.single_gpu
    live = args.live
    do_sem = args.do_sem
    use_scribble = args.use_scribble
    show_mesh = True
    main_device = args.device

    if multi_gpu:
        device = "cuda:1"
        view_freq = 1
    else:
        device = main_device
        view_freq = 1

    colormap = torch.tensor(
        [[255, 0, 0],
         [0, 255, 0],
         [0, 0, 255],
         [255, 255, 0],
         [0, 255, 255],
         [255, 0, 255],
         [255, 128, 0],
         [101, 67, 33],
         [230, 0, 126],
         [120, 120, 120],
         [235, 150, 135],
         [0, 128, 0]],
        device=device
    )

    colormap_np = colormap.cpu().numpy().astype(np.uint8)
    colormap_np = colormap_np.reshape(-1, 3)
    h_level = 0

    # init trainer-------------------------------------------------------------
    render_trainer = trainer.RenderTrainer(
        device,
        config_file,
        load_path=None,
        load_epoch=None,
        do_mesh=True,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
        do_sem=do_sem
    )
    render_trainer.optimiser = None
    render_trainer.fc_occ_map.eval()
    render_trainer.B_layer.eval()

    im_width = render_trainer.config["camera"]["w"] - 2 * render_trainer.config["camera"]["mw"]
    im_height = render_trainer.config["camera"]["h"] - 2 * render_trainer.config["camera"]["mh"]

    # Main thread
    def vis(self):
        global mouse_callback_params, request_new_keyframe, force_exit, request_previous_keyframe

        print('vis: starting')

        ctx = torch.multiprocessing.get_context('spawn')

        self.track_to_map_IDT = ctx.Queue()
        self.map_to_track_params = ctx.Queue()
        self.map_to_vis_params = ctx.Queue(maxsize=1)
        self.track_to_vis_T_WC = ctx.Queue(maxsize=1)
        self.map_to_vis_active = ctx.Queue(maxsize=1)
        self.map_to_vis_kf_depth = ctx.Queue()
        self.vis_to_map_save_idx = ctx.Queue()
        self.vis_to_map_labels = ctx.Queue()

        track_p = ctx.Process(
            target=tracking, args=(
                (self.track_to_map_IDT),
                (self.map_to_track_params),
                (self.track_to_vis_T_WC),
                (self.main_device),
                (self.config_file),
                (None),
                (None),
                (self.show_mesh),
                (self.incremental),
                (self.do_track),
                (self.do_color),
                (self.do_sem),
                (self.live)
            ))

        map_p = ctx.Process(
            target=mapping, args=(
                (self.track_to_map_IDT),
                (self.map_to_track_params),
                (self.map_to_vis_params),
                (self.map_to_vis_kf_depth),
                (self.map_to_vis_active),
                (self.vis_to_map_save_idx),
                (self.vis_to_map_labels),
                (self.main_device),
                (self.config_file),
                (None),
                (None),
                (self.show_mesh),
                (self.incremental),
                (self.do_track),
                (self.do_color),
                (self.do_sem),
            ))

        # start processes
        track_p.start()
        map_p.start()

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
        self.render_trainer.kfs_im = []
        self.render_trainer.kfs_depth = []

        mesh_geo = None
        mesh_frame = None
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=int(self.render_trainer.W),
                          height=int(self.render_trainer.H),
                          left=600, top=50,
                          visible=False)
        view_ctl = vis.get_view_control()
        cam = view_ctl.convert_to_pinhole_camera_parameters()
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.render_trainer.W, self.render_trainer.H,
            self.render_trainer.fx, self.render_trainer.fy,
            self.render_trainer.cx, self.render_trainer.cy)

        latest_image = None
        label_kf_store = []
        keyframe_set = False
        while(True):
            if force_exit:
                break
            # add reveived keyframes-----------------------------------------------
            while(True):
                if force_exit:
                    break
                try:
                    kf_depth, kf_im, T_WC = self.map_to_vis_kf_depth.get(block=False)
                    if kf_depth is not None:
                        # add pose
                        self.render_trainer.frames.T_WC_batch = self.render_trainer.expand_data(
                            self.render_trainer.frames.T_WC_batch,
                            T_WC.to(self.render_trainer.device),
                            False)
                        del T_WC
                        self.render_trainer.batch_size = self.render_trainer.frames.T_WC_batch.shape[0]

                        # add depth
                        kf_depth_np = kf_depth.cpu().numpy()
                        del kf_depth
                        kf_depth_resize = imgviz.resize(
                            kf_depth_np,
                            width=self.render_trainer.W_vis,
                            height=self.render_trainer.H_vis,
                            interpolation="nearest")
                        self.render_trainer.kfs_depth.append(kf_depth_resize)

                        # add rgb
                        kf_im_np = kf_im.cpu().numpy()
                        del kf_im

                        kf_im_resize = imgviz.resize(
                            kf_im_np,
                            width=self.render_trainer.W_vis,
                            height=self.render_trainer.H_vis,
                            interpolation="nearest")

                        kf_im_resize = (kf_im_resize * 255).astype(np.uint8)
                        self.render_trainer.kfs_im.append(kf_im_resize)

                        # add to semantics
                        label_kf = (kf_im_np * 255).astype(np.uint8)
                        label_kf_store.append(label_kf)


                        # add pointcloud
                        pc_gt_cam = trainer.backproject_pcs(
                            kf_depth_resize[None, ...],
                            self.render_trainer.fx_vis,
                            self.render_trainer.fy_vis,
                            self.render_trainer.cx_vis,
                            self.render_trainer.cy_vis)
                        self.render_trainer.pcs_gt_cam = self.render_trainer.expand_data(
                            self.render_trainer.pcs_gt_cam,
                            pc_gt_cam,
                            replace=False)
                        update_kfs = True

                except queue.Empty:
                    break

            if request_new_keyframe or (len(label_kf_store) > 0 and not keyframe_set):
                keyframe_set = True
                request_new_keyframe = False
                batch_label = mouse_callback_params["batch_label"] + 1
                if batch_label < len(label_kf_store):
                    mouse_callback_params["batch_label"] = batch_label

                # emit new keyframe to gui
                qtKeyframe = QtGui.QImage(label_kf_store[mouse_callback_params["batch_label"]].data, label_kf_store[mouse_callback_params["batch_label"]].shape[1], label_kf_store[mouse_callback_params["batch_label"]].shape[0], label_kf_store[mouse_callback_params["batch_label"]].strides[0],
                                          QtGui.QImage.Format_RGB888)
                keyframe_emit = qtKeyframe.scaled(self.im_width, self.im_height, Qt.KeepAspectRatio)
                self.changeKeyframe.emit(keyframe_emit)
            if request_previous_keyframe and keyframe_set and len(label_kf_store) > 0:
                request_previous_keyframe = False
                batch_label = mouse_callback_params["batch_label"] - 1
                if batch_label >= 0:
                    mouse_callback_params["batch_label"] = batch_label
                # emit new keyframe to gui
                qtKeyframe = QtGui.QImage(label_kf_store[mouse_callback_params["batch_label"]].data, label_kf_store[mouse_callback_params["batch_label"]].shape[1], label_kf_store[mouse_callback_params["batch_label"]].shape[0], label_kf_store[mouse_callback_params["batch_label"]].strides[0],
                                          QtGui.QImage.Format_RGB888)
                keyframe_emit = qtKeyframe.scaled(self.im_width, self.im_height, Qt.KeepAspectRatio)
                self.changeKeyframe.emit(keyframe_emit)

            if update_kfs:

                update_kfs = False

                # recompute scene bounds ------------------------------------------
                pc = trainer.draw_pc(
                    self.render_trainer.batch_size,
                    self.render_trainer.pcs_gt_cam,
                    self.render_trainer.frames.T_WC_batch.cpu().numpy()
                )
                transform, extents = trimesh.bounds.oriented_bounds(pc)
                transform = np.linalg.inv(transform)
                self.render_trainer.set_scene_bounds(extents, transform)

            # update NN params -------------------------------------------------
            params = get_latest_queue(self.map_to_vis_params)
            if params is not None:
                # print("got params")
                state_dict, B_layer_dict, T_WC_track = params
                self.render_trainer.fc_occ_map.load_state_dict(state_dict)
                del state_dict
                self.render_trainer.B_layer.load_state_dict(B_layer_dict)
                del B_layer_dict

                self.render_trainer.frames.T_WC_track = T_WC_track.to(
                    self.render_trainer.device)
                del T_WC_track

                param_counter += 1

            # compute scene mesh -------------------------------------------------
            if self.render_trainer.grid_pc is not None and do_mesh:

                with torch.set_grad_enabled(False):
                    alphas, _, sems = occupancy.chunks(
                        self.render_trainer.grid_pc,
                        self.render_trainer.chunk_size,
                        self.render_trainer.fc_occ_map,
                        self.render_trainer.n_embed_funcs,
                        self.render_trainer.B_layer,
                        do_sem=False
                    )

                    occ = render_rays.occupancy_activation(
                        alphas, self.render_trainer.voxel_size)

                    dim = self.render_trainer.grid_dim
                    occ = occ.view(dim, dim, dim)

                fc_occ_map = None
                if self.render_trainer.do_color:
                    fc_occ_map = self.render_trainer.fc_occ_map

                if self.render_trainer.do_sem:
                    colormap = self.colormap
                    h_level =self.h_level

                occ_mesh = trainer.draw_mesh(
                    occ,
                    0.45,
                    self.render_trainer.scene_scale_np,
                    self.render_trainer.bounds_tranform_np,
                    self.render_trainer.chunk_size,
                    self.render_trainer.B_layer,
                    fc_occ_map,
                    self.render_trainer.n_embed_funcs,
                    do_sem=self.render_trainer.do_sem,
                    do_hierarchical=self.render_trainer.do_hierarchical,
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
            if self.render_trainer.T_WC is not None:
                if follow:
                    T_CW_np = self.render_trainer.T_WC[0].inverse().cpu().numpy()
                    cam = view_ctl.convert_to_pinhole_camera_parameters()
                    cam.extrinsic = T_CW_np
                    view_ctl.convert_from_pinhole_camera_parameters(cam)
                    if mesh_frame is not None:
                        mesh_frame.clear()

                else:
                    T_WC_np = self.render_trainer.T_WC[0].cpu().numpy()
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
            latest_frame = get_latest_queue(self.track_to_vis_T_WC)
            if latest_frame is not None:
                T_WC, depth_latest_np, im_latest_np = latest_frame
                self.render_trainer.T_WC = T_WC.clone().to(self.render_trainer.device)
                del T_WC

                self.render_trainer.depth_resize = imgviz.resize(
                    depth_latest_np[0],
                    width=self.render_trainer.W_vis,
                    height=self.render_trainer.H_vis,
                    interpolation="nearest")
                del depth_latest_np

                self.render_trainer.im_resize = imgviz.resize(
                    im_latest_np[0],
                    width=self.render_trainer.W_vis,
                    height=self.render_trainer.H_vis)

                latest_image = np.copy(im_latest_np[0])
                del im_latest_np

            # get active samples and compute mask ---------------------------------
            active_inds = get_latest_queue(self.map_to_vis_active)
            if active_inds is not None:
                self.render_trainer.active_inds = active_inds
                del active_inds

            mask = None
            if self.render_trainer.active_inds is not None:
                inds = self.render_trainer.active_inds.cpu().numpy()

            # render live view ----------------------------------------------------
            if self.render_trainer.T_WC is not None and param_counter % self.view_freq == 0:
                if do_render:
                    view_depths, view_vars, view_cols, view_sems = trainer.render_vis(
                        1,
                        self.render_trainer,
                        self.render_trainer.T_WC,
                        self.render_trainer.do_fine,
                        do_var=True,
                        do_color=self.render_trainer.do_color,
                        do_sem=self.render_trainer.do_sem,
                        do_hierarchical=self.render_trainer.do_hierarchical
                    )

                    # send segmentation result to vis if in server mode
                    if latest_image is not None and len(label_kf_store) > 0:
                        label_im = torch.argmax(view_sems[0], axis=2).cpu().numpy().astype(np.uint8)
                        sem_mask = cv2.resize(label_im, (self.im_width, self.im_height), cv2.INTER_NEAREST)
                        seg_image = utils.visualise_segmentation(latest_image, sem_mask, self.colormap_np)
                        # Emit seg vis
                        qtCurrentframe = QtGui.QImage(seg_image.data, seg_image.shape[1], seg_image.shape[0], seg_image.strides[0],
                                                      QtGui.QImage.Format_RGB888)
                        seg_emit = qtCurrentframe.scaled(self.im_width, self.im_height, Qt.KeepAspectRatio)
                        self.changeSegImage.emit(seg_emit)
                        # request_new_kf = track_to_vis_request_new_kf.get(block=False, timeout=None)

                param_counter += 1

                # interactive label ------------------------------------------
                # Check if new scribble is available
                if self.render_trainer.do_sem and do_render and mouse_callback_params["drawing"]:
                    while not mouse_callback_params["new_scribble"]:
                        continue

                    mouse_callback_params["new_scribble"] = False

                    self.vis_to_map_labels.put(
                        (mouse_callback_params["indices_b"],
                         mouse_callback_params["indices_h"],
                         mouse_callback_params["indices_w"],
                         mouse_callback_params["classes"],
                         mouse_callback_params["h_labels"],
                         mouse_callback_params["h_masks"])
                    )



#------------------------------------------ GUI -----------------------------------------------------------------
# GUI
class IGui(QtGui.QDialog):

    def __init__(self, device, num_classes, sensor_type, num_points_per_scribble):
        global mouse_callback_params

        mouse_callback_params["n_points_per_scribble"] = num_points_per_scribble

        super(IGui, self).__init__()
        if sensor_type == 'Realsense':
            self.window_size = (640, 480)
        elif sensor_type == 'Azure':
            self.window_size = (1200, 680)
        elif sensor_type == 'None':
            self.window_size = (1200, 600)

        self.initUI()
        self.setup_thread()
        self.num_classes = num_classes
        self.sensor_type = sensor_type
        self.device = device

        # set colour pallette. Same as iMAP
        self.colormap = torch.tensor(
            [[255, 0, 0],
             [0, 255, 0],
             [0, 0, 255],
             [255, 255, 0],
             [0, 255, 255],
             [255, 0, 255],
             [255, 128, 0],
             [101, 67, 33],
             [230, 0, 126],
             [120, 120, 120],
             [235, 150, 135],
             [0, 128, 0]],
            device=self.device
        )

        colormap_np = self.colormap.cpu().numpy().astype(np.uint8)
        self.colormap_np = colormap_np.reshape(-1, 3)
        self.h_level = 0

        self.current_keyframe = None
        self.keyframe_idx = -1
        self.save_mesh_flag = False
        self.client_closing_flag = False

        self.n_clicks = 0
        self.batch_label = 0
        self.h_level = 0

    def initUI(self):
        self.originalPalette = QtGui.QApplication.palette()
        self.main_tab = QtGui.QWidget()
        self.main_tab.layout = QtGui.QHBoxLayout()

        # add class buttons
        buttons_layout = QtGui.QHBoxLayout()
        self.class_one_button = QtGui.QPushButton('1')
        self.class_one_button.setStyleSheet("background-color:rgb(255, 0, 0); border: native")
        buttons_layout.addWidget(self.class_one_button)

        self.class_two_button = QtGui.QPushButton('2')
        self.class_two_button.setStyleSheet("background-color:rgb(0, 255, 0); border: none")
        buttons_layout.addWidget(self.class_two_button)

        self.class_three_button = QtGui.QPushButton('3')
        self.class_three_button.setStyleSheet("background-color:rgb(0, 0, 255); border: none")
        buttons_layout.addWidget(self.class_three_button)

        self.class_four_button = QtGui.QPushButton('4')
        self.class_four_button.setStyleSheet("background-color:rgb(255, 255, 0); border: none")
        buttons_layout.addWidget(self.class_four_button)

        self.class_five_button = QtGui.QPushButton('5')
        self.class_five_button.setStyleSheet("background-color:rgb(0, 255, 255); border: none")
        buttons_layout.addWidget(self.class_five_button)

        # add nearest keyframe button
        misc_buttons_layout = QtGui.QVBoxLayout()
        keyframe_buttons_layout = QtGui.QHBoxLayout()
        self.next_keyframe_button = QtGui.QPushButton('Next KF')
        self.previous_keyframe_button = QtGui.QPushButton('Previous KF')
        keyframe_buttons_layout.addWidget(self.next_keyframe_button)
        keyframe_buttons_layout.addWidget(self.previous_keyframe_button)
        misc_buttons_layout.addLayout(keyframe_buttons_layout)

        self.save_button = QtGui.QPushButton('Save Mesh')
        misc_buttons_layout.addWidget(self.save_button, alignment=QtCore.Qt.AlignTop)

        # image layout: rgb image and keyframe image
        self.image_widget = pg.GraphicsLayoutWidget()
        self.keyframe_widget = pg.GraphicsLayoutWidget()
        self.current_view_box = self.image_widget.addViewBox(lockAspect=True)
        self.current_view_box.invertY(True)
        self.current_view_box.setMouseEnabled(x=False, y=False)
        self.current_view_image = pg.ImageItem(np.zeros([self.window_size[0], self.window_size[1], 3], dtype=np.int32))
        self.current_view_box.addItem(self.current_view_image)
        self.keyframe_view_box = self.keyframe_widget.addViewBox(lockAspect=True)
        self.keyframe_view_box.invertY(True)
        self.keyframe_view_box.setMouseEnabled(x=False, y=False)
        self.keyframe_image = pg.ImageItem(np.zeros([self.window_size[0], self.window_size[1], 3], dtype=np.int32))
        self.keyframe_view_box.addItem(self.keyframe_image)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addLayout(buttons_layout, 0, 1)
        mainLayout.addWidget(self.image_widget, 1, 0)
        mainLayout.addWidget(self.keyframe_widget, 1, 1)
        mainLayout.addLayout(misc_buttons_layout, 1, 2)
        self.setLayout(mainLayout)

        self.show()

    def mouse_press_event(self, event):
        global mouse_callback_params, scribble_params

        drawing = mouse_callback_params["drawing"]
        scribble_exists = mouse_callback_params["new_scribble"]

        # Don't allow new annotation until previous one has been sent to server
        if event.button() == Qt.LeftButton and not drawing and not scribble_exists:
            mouse_callback_params['drawing'] = True
            mouse_callback_params['new_scribble'] = False
            pos = event.pos()
            pos = self.keyframe_image.mapFromScene(pos)
            scribble_params["indices_h"].append(int(pos.y()))
            scribble_params["indices_w"].append(int(pos.x()))
            scribble_params["indices_b"].append(mouse_callback_params["batch_label"])

    def mouse_move_event(self, event):
        global mouse_callback_params, scribble_params

        if (event.buttons() and Qt.LeftButton) and mouse_callback_params['drawing']:
            sem_class = mouse_callback_params["class"]
            colour = self.colormap_np[sem_class]
            keyframe_image = np.copy(self.keyframe_image.image)
            pos = event.pos()
            pos = self.keyframe_image.mapFromScene(pos)
            cv2.line(keyframe_image, (scribble_params["indices_w"][-1], scribble_params["indices_h"][-1]), (int(pos.x()), int(pos.y())), colour.astype(np.uint8).tolist(), 5)
            # update position
            scribble_params["indices_h"].append(int(pos.y()))
            scribble_params["indices_w"].append(int(pos.x()))
            scribble_params["indices_b"].append(mouse_callback_params["batch_label"])
            self.keyframe_image.setImage(keyframe_image)

    def mouse_release_event(self, event):
        global mouse_callback_params, scribble_params

        if mouse_callback_params['drawing']:
            mouse_callback_params['drawing'] = False
            mouse_callback_params['new_scribble'] = True

            scribble_length = len(scribble_params["indices_h"])
            if scribble_length > mouse_callback_params["n_points_per_scribble"]:
                sample_idx = np.linspace(0, scribble_length - 1, mouse_callback_params["n_points_per_scribble"], dtype=int)
                mouse_callback_params["indices_h"].extend([scribble_params["indices_h"][i] for i in sample_idx[:mouse_callback_params["n_points_per_scribble"]]])
                mouse_callback_params["indices_w"].extend([scribble_params["indices_w"][i] for i in sample_idx[:mouse_callback_params["n_points_per_scribble"]]])
                mouse_callback_params["indices_b"].extend([scribble_params["indices_b"][i] for i in sample_idx[:mouse_callback_params["n_points_per_scribble"]]])

                mouse_callback_params["classes"].extend(mouse_callback_params["n_points_per_scribble"] * [mouse_callback_params["class"]])
            else:
                # Otherwsie use all points in scribble
                mouse_callback_params["indices_h"].extend(scribble_params["indices_h"])
                mouse_callback_params["indices_w"].extend(scribble_params["indices_w"])
                mouse_callback_params["indices_b"].extend(scribble_params["indices_b"])
                mouse_callback_params["classes"].extend(scribble_length * [mouse_callback_params["class"]])

            # reset temp scribble params
            scribble_params["indices_w"] = []
            scribble_params["indices_h"] = []
            scribble_params["indices_b"] = []
            mouse_callback_params["drawing"] = False

    def QImageToNP(self, image):
        # Convert a QImage to a numpy array
        image = image.convertToFormat(QtGui.QImage.Format_RGB888)
        width = image.width()
        height = image.height()
        ptr = image.constBits()
        return np.frombuffer(ptr.asstring(image.byteCount()), dtype=np.uint8).reshape(height, width, 3)

    @pyqtSlot(QtGui.QImage)
    def set_current_image(self, image):
        np_image = self.QImageToNP(image)
        self.current_view_image.setImage(np_image)

    @pyqtSlot(QtGui.QImage)
    def set_keyframe(self, image):
        np_image = self.QImageToNP(image)
        self.keyframe_image.setImage(np_image)

    @pyqtSlot()
    def request_new_keyframe(self):
        global request_new_keyframe
        request_new_keyframe = True

    @pyqtSlot()
    def request_previous_keyframe(self):
        global request_previous_keyframe
        request_previous_keyframe = True

    @pyqtSlot()
    def save_mesh(self):
        self.save_mesh_flag = True

    @pyqtSlot()
    def set_class(self, idx: int):
        """
        Sets the class index.

        :param class_idx:
        :return:
        """

        if idx == mouse_callback_params["class"]:
            return

        assert (idx < self.num_classes)

        previous_class = mouse_callback_params["class"]

        mouse_callback_params["class"] = idx

        # Reset appearance of previous class
        if previous_class == 0:
            self.class_one_button.setStyleSheet("background-color:rgb(255, 0, 0); border: none")
        if previous_class == 1:
            self.class_two_button.setStyleSheet("background-color:rgb(0, 255, 0); border: none")
        if previous_class == 2:
            self.class_three_button.setStyleSheet("background-color:rgb(0, 0, 255); border: none")
        if previous_class == 3:
            self.class_four_button.setStyleSheet("background-color:rgb(255, 255, 0); border: none")
        if previous_class == 4:
            self.class_five_button.setStyleSheet("background-color:rgb(0, 255, 255); border: none")

        # highlight button of current class
        if idx == 0:
            self.class_one_button.setStyleSheet("background-color:rgb(255, 0, 0); border: native")
        if idx == 1:
            self.class_two_button.setStyleSheet("background-color:rgb(0, 255, 0); border: native")
        if idx == 2:
            self.class_three_button.setStyleSheet("background-color:rgb(0, 0, 255); border: native")
        if idx == 3:
            self.class_four_button.setStyleSheet("background-color:rgb(255, 255, 0); border: native")
        if idx == 4:
            self.class_five_button.setStyleSheet("background-color:rgb(0, 255, 255); border: native")

    def keyPressEvent(self, event):
        global force_exit
        if event.key() == QtCore.Qt.Key_Q:
            print("Closing client")
            self.client_closing_flag = True
            # send exit flag
        event.accept()
        force_exit = True
        self.close()

    def setup_thread(self):
        self.thread = QThread()
        self.worker = Train()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.vis)

        self.keyframe_widget.mousePressEvent = self.mouse_press_event
        self.keyframe_widget.mouseMoveEvent = self.mouse_move_event
        self.keyframe_widget.mouseReleaseEvent = self.mouse_release_event

        self.worker.changeSegImage.connect(self.set_current_image)
        self.worker.changeKeyframe.connect(self.set_keyframe)

        # Connect buttons
        self.class_one_button.clicked.connect(lambda: self.set_class(0))
        self.class_two_button.clicked.connect(lambda: self.set_class(1))
        self.class_three_button.clicked.connect(lambda: self.set_class(2))
        self.class_four_button.clicked.connect(lambda: self.set_class(3))
        self.class_five_button.clicked.connect(lambda: self.set_class(4))

        self.next_keyframe_button.clicked.connect(self.request_new_keyframe)
        self.previous_keyframe_button.clicked.connect(self.request_previous_keyframe)
        self.save_button.clicked.connect(self.save_mesh)
        self.thread.start()

    def close_client(self):
        print("Closing client")
        self.close()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = IGui(args.device, args.num_classes, args.sensor, args.points_per_scribble)
    # ex.show()
    sys.exit(app.exec_())
