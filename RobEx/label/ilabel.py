#!/usr/bin/env python
import torch

from pytorch3d.transforms import matrix_to_quaternion
import cv2
import numpy as np
import imgviz
import random
import glob

from RobEx.train import trainer
from RobEx.render import render_rays
from RobEx import visualisation
import pdb

import matplotlib.pyplot as plt

import mercury
import queue
import time
from RobEx import geometry

from . import raster_point
from . import edge_detection

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

def sample_points_from_bounding_box(bounding_box, num_points):
    """
    Samples num_points random points from the given bounding box [x, y, w, h]
    If num_points = 1 then sample central point
    """
    coord_set = set()
    if num_points == 1:
        centre_x = int(bounding_box[0] + bounding_box[2] / 2.)
        centre_y = int(bounding_box[1] + bounding_box[3] / 2.)
        return np.array([[centre_x, centre_y]], dtype=np.int32)

    while len(coord_set) < num_points:
        x, y = np.random.randint(bounding_box[0], high=bounding_box[0] + bounding_box[2]), np.random.randint(bounding_box[1], high=bounding_box[1] + bounding_box[3])
        # that will make sure we don't add (7, 0) to cords_set
        coord_set.add((x, y))

    return np.array(list(coord_set))


def sample_points_from_mask(mask, num_points):
    """
    Samples num_points random points from the mask
    """
    coord_set = set()
    random_indices = np.random.randint(0, high=cv2.countNonZero(mask), size=(num_points,))
    bground = np.nonzero(mask)
    bground = np.concatenate((bground[1].reshape(-1,1), bground[0].reshape(-1,1)), axis=1)
    bground = bground[random_indices]
    return bground


def binary_to_decimal(binary, level):
    round_level = binary[..., :level]
    bin_mult = 2**torch.arange(
        start=level - 1,
        end=-1, step=-1, device=binary.device
    )
    decimal = (round_level * bin_mult[None, None, :]).sum(-1)

    return decimal


def map_color_hierarchcial(sem, colormap, binary, only_level=None):
    n_levels = binary.shape[-1]
    h_sem_vis = []

    if (only_level == 0) or (only_level is None):
        sem_lev0 = sem[..., :1]
        col0 = (1. - sem_lev0) * colormap[0][0][..., :]
        col1 = sem_lev0 * colormap[0][1][..., :]
        col_lev0 = col0 + col1
        h_sem_vis.append(col_lev0)

    for level in range(1, n_levels):
        if (only_level == level) or (only_level is None):
            decimal = binary_to_decimal(binary, level)
            n_prev_nodes = 2**level - 1
            inds_lev = decimal + n_prev_nodes

            sem_lev = sem[..., level:level + 1]
            select_color = colormap[inds_lev, :, :]
            col00 = (1. - sem_lev) * select_color[..., 0, :]
            col01 = sem_lev * select_color[..., 1, :]
            col_lev = col00 + col01
            h_sem_vis.append(col_lev)

    return h_sem_vis


def select_label_entropy(render_trainer, T_WC_kf, factor=8):
    _, _, _, sem_kf = trainer.render_vis(
        1,
        render_trainer,
        T_WC_kf[None, ...],
        render_trainer.do_fine,
        do_var=False,
        do_color=False,
        do_sem=True,
        do_hierarchical=render_trainer.do_hierarchical
    )
    entropy = render_rays.entropy(sem_kf[0])
    w_block = render_trainer.W_vis // factor
    h_block = render_trainer.H_vis // factor

    entropy_avg = entropy.view(
        factor, h_block, factor, w_block)
    entropy_avg = entropy_avg.sum(dim=(1, 3))

    max_ind = torch.argmax(entropy_avg).item()
    print(max_ind)
    y_ind, x_ind = np.unravel_index(
        max_ind, (factor, factor))
    print(y_ind, x_ind)
    entropy_avg = entropy_avg.cpu().numpy()
    entropy_avg_kf_vis = imgviz.depth2rgb(entropy_avg)
    entropy_avg_kf_vis = cv2.cvtColor(
        entropy_avg_kf_vis, cv2.COLOR_BGR2RGB)

    entropy = entropy.cpu().numpy()
    entropy_kf_vis = imgviz.depth2rgb(entropy)
    entropy_kf_vis = cv2.cvtColor(
        entropy_kf_vis, cv2.COLOR_BGR2RGB)

    w_block = render_trainer.W // factor
    h_block = render_trainer.H // factor

    y_auto = y_ind * h_block + \
        random.randint(0, h_block)
    x_auto = x_ind * w_block + \
        random.randint(0, w_block)

    return [x_auto, y_auto], entropy_avg_kf_vis, entropy_kf_vis


def sem_idx_hier(h_label):
    n_classes = len(h_label)

    binary = []
    h_level = 0
    for level in range(n_classes):
        label = h_label[level]
        if label == -1:
            break
        else:
            if level != 0:
                h_level += 1
        binary.append(label)

    n_levels = len(binary)
    binary = np.array(binary)
    bin_mult = 2**np.arange(
        start=n_levels - 1, stop=-1, step=-1
    )
    label_idx = (binary * bin_mult).sum() + 2**(n_levels) - 2
    if label_idx < 0:
        label_idx = 0

    return label_idx, h_level

def bootstrap_label_image(x, y, param):
    """
    Uses pre-computed points to label image

    :param x: Array of x-locations
    :param y: Array of y-locations
    :param param: params mimicking mouse_callback_parameters.
    """

    batch_label = param["batch_label"]
    im = param["ims"][batch_label]
    palette = param["col_palette"]

    param["indices_b"].extend([batch_label for i in range(0, x.shape[0])])
    param["indices_h"].extend([i for i in y])
    param["indices_w"].extend([i for i in x])
    param["classes"].extend([param["class"]] * x.shape[0])
    color = palette[param["class"]]

    for i in range(x.shape[0]):
        cv2.circle(im, (x[i], y[i]), 8, color.astype(np.uint8).tolist(), -1)

def label_image(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN or event == "center":
        batch_label = param["batch_label"]

        param["indices_b"].append(batch_label)
        if event == cv2.EVENT_LBUTTONDOWN:
            x_orig = x #* param["vis_scale"]
            y_orig = y #* param["vis_scale"]
            x_scale = x
            y_scale = y
        else:
            x_orig = x
            y_orig = y
            x_scale = x #// param["vis_scale"]
            y_scale = y #// param["vis_scale"]

        param["indices_h"].append(y_orig)
        param["indices_w"].append(x_orig)
        do_hierarchical = param["do_hierarchical"]
        if do_hierarchical:
            h_mask = param["h_class"] != -1
            h_label = param["h_class"].copy()
            param["h_labels"].append(h_label)
            param["h_masks"].append(h_mask)

        param["classes"].append(param["class"])

        palette = param["col_palette"]
        color = palette[param["class"]]

        im = param["ims"][batch_label]
        cv2.circle(im, (x_scale, y_scale), 5,
                   color.astype(np.uint8).tolist(), -1)


def label_image_scribble(event, x, y, flags, param):
    """
    Allows labelling an image with mouse movement while left button is held down.

    :param event: Mouse event
    :param x: x location
    :param y: y location
    :param param: Mouse callback parameters.
    :return: None
    """

    # grab references to the global variables
    do_hierarchical = param["do_hierarchical"]
    batch_label = param["batch_label"]
    im = param["ims"][batch_label]
    palette = param["col_palette"]
    drawing = param["drawing"]
    color = palette[param["class"]]

    if event == cv2.EVENT_LBUTTONDOWN and not drawing:
        param["drawing"] = True
        param["s_indices_b"].append(batch_label)
        param["s_indices_h"].append(y)
        param["s_indices_w"].append(x)
        print(f"sem_class: {param['class']}")

    elif drawing and (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONUP):
        cv2.line(im, (param["s_indices_w"][-1], param["s_indices_h"][-1]), (x, y), color.astype(np.uint8).tolist(), 5)
        param["s_indices_b"].append(batch_label)
        param["s_indices_h"].append(y)
        param["s_indices_w"].append(x)

        if event == cv2.EVENT_LBUTTONUP:
            param["drawing"] = False
            scribble_length = len(param["s_indices_b"])
            sem_class = param["class"]
            # Sample n evenly-spaced points from completed scribble
            if scribble_length > param["n_points_per_scribble"]:
                sample_idx = np.linspace(0, scribble_length - 1, param["n_points_per_scribble"], dtype=int)
                if do_hierarchical:
                    h_mask = param["h_class"] != -1
                    h_label = param["h_class"].copy()
                    param["h_labels"].append(h_label)
                    param["h_masks"].append(h_mask)

                param["indices_b"].extend([param["s_indices_b"][i] for i in sample_idx[:param["n_points_per_scribble"]]])
                param["indices_h"].extend([param["s_indices_h"][i] for i in sample_idx[:param["n_points_per_scribble"]]])
                param["indices_w"].extend([param["s_indices_w"][i] for i in sample_idx[:param["n_points_per_scribble"]]])
                param["classes"].extend(param["n_points_per_scribble"] * [sem_class])
            else:
                # Otherwsie use all points in scribble
                param["indices_b"].extend(param["s_indices_b"])
                param["indices_h"].extend(param["s_indices_h"])
                param["indices_w"].extend(param["s_indices_w"])
                param["classes"].extend(scribble_length * [sem_class])

            # reset temp scribble params
            param["s_indices_b"] = []
            param["s_indices_w"] = []
            param["s_indices_h"] = []
            param["drawing"] = False


class iLabel(object):
    def __init__(self, do_hierarchical, do_names, label_cam, device, n_classes, query_selector, use_scribble=False, n_points_per_scribble=10, headless=False, bootstrap=False, bootstrap_dir=None):

        super(iLabel, self).__init__()
        self.do_hierarchical = do_hierarchical
        self.device = device
        self.do_names = do_names
        self.n_classes = n_classes
        self.use_scribble = use_scribble
        self.label_cam = label_cam
        self.query_selector = query_selector

        self.do_label_auto = False
        self.success_query = False

        self.do_continuous = False ## For continuous auto labelling
        self.n_label = 0 ## Number of successful auto labels
        self.n_label_pause = 0 ##
        self.hessian_plane = None ## For plane fit
        self.label_plane = True
        self.vertical = True
        self.mask_edges = True
        self.do_analysis = True
        self.batch_auto = None
        self.edge_mask = None

        self.n_clicks = 0
        self.batch_label = 0
        self.h_level = 0
        self.w_node = 0
        self.headless = headless
        self.bootstrap = False
        self.bootstrap_dir = None
        # Check file exists
        if bootstrap:
            if not glob.glob(bootstrap_dir):
                self.bootstrap = False
            else:
                self.bootstrap = bootstrap
                self.bootstrap_files = sorted(glob.glob(bootstrap_dir + "*.npy"))
        self.vis_scale = 2

        self.mouse_callback_params = {}
        self.mouse_callback_params["indices_b"] = []
        self.mouse_callback_params["indices_w"] = []
        self.mouse_callback_params["indices_h"] = []
        self.mouse_callback_params["classes"] = []
        self.mouse_callback_params["h_labels"] = []
        self.mouse_callback_params["h_masks"] = []
        self.mouse_callback_params["ims"] = []
        self.mouse_callback_params["kfs"] = []
        self.mouse_callback_params["query_masks"] = []
        self.mouse_callback_params["class"] = 0
        self.mouse_callback_params["h_class"] = np.array(
            [-1] * self.n_classes)
        self.mouse_callback_params["batch_label"] = 0
        self.mouse_callback_params["do_hierarchical"] = self.do_hierarchical
        self.mouse_callback_params["vis_scale"] = self.vis_scale

        # Scribble parameters
        self.mouse_callback_params["s_indices_b"] = []
        self.mouse_callback_params["s_indices_w"] = []
        self.mouse_callback_params["s_indices_h"] = []
        self.mouse_callback_params["s_class"] = []
        self.mouse_callback_params["drawing"] = False
        self.mouse_callback_params["n_points_per_scribble"] = n_points_per_scribble # Number of points to sample from each scribble

        self.label_names = None
        if self.do_hierarchical:
            self.colormap = torch.tensor(
                [
                    [[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [255, 255, 0]],
                    [[0, 255, 255], [255, 128, 0]],
                    [[255, 153, 0], [128, 0, 0]],
                    [[0, 128, 0], [204, 255, 255]],
                    [[0, 102, 204], [230, 0, 126]],
                    [[153, 204, 0], [153, 153, 255]],
                ],
                device=self.device
            )
            if self.do_names:
                n_nodes = 2**(self.n_classes + 1) - 2
                self.label_names = ["-"] * n_nodes

        else:

            # generate colourmap
            self.colormap = torch.tensor(
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
                device=self.device
            )
            self.colormap = self.colormap[:n_classes, :]
            if self.do_names:
                self.label_names = ["-"] * self.n_classes

        colormap_np = self.colormap.cpu().numpy().astype(np.uint8)
        colormap_np = colormap_np.reshape(-1, 3)
        self.mouse_callback_params["col_palette"] = colormap_np

        if not self.headless:
            cv2.namedWindow('label', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow("label", 1000, 800)

            if not self.bootstrap:
                if self.use_scribble:
                    cv2.setMouseCallback('label', label_image_scribble, self.mouse_callback_params)  # set callback function for scribble in "label" window
                else:
                    cv2.setMouseCallback('label', label_image, self.mouse_callback_params)  # set callback function for a click in "label" window

            if do_hierarchical:
                self.tree_im = visualisation.draw.draw_tree(
                    500, 500, self.n_classes + 1,
                    self.mouse_callback_params["col_palette"],
                    0, None
                )
                cv2.imshow("tree hierarchy", self.tree_im)

        self.h_class_to_col = {}
        self.h_class_to_col[-1] = [0, 0, 255]
        self.h_class_to_col[0] = [0, 0, 0]
        self.h_class_to_col[1] = [255, 255, 255]

        self.roi_count = 0

    def add_keyframe(self, kf, frame_count=0):
        # Added frame_count parameter for bootstrap testing.
        self.mouse_callback_params["ims"].append(kf)
        self.mouse_callback_params["kfs"].append(kf.copy())
        self.mouse_callback_params["query_masks"].append(
            np.zeros((kf.shape[0], kf.shape[1]), dtype=np.uint8))

        # if this is the first key frame and bootstrap is true then start labelling using bootstrap file
        if len(self.mouse_callback_params["ims"]) == 1 and self.bootstrap:
            # load file
            bounding_boxes = np.load(self.bootstrap_files[frame_count])

            num_objects = bounding_boxes.shape[0]
            # compute bounding box sizes
            bbox_areas = [i[2] * i[3] for i in bounding_boxes]
            sorted_idx = np.argsort(bbox_areas)
            sorted_idx = sorted_idx[::-1] # descending order, start with largest

            # Also sample points from background, use mask to generate background
            temp_mask = np.zeros((kf.shape[0], kf.shape[1]), dtype=np.int32)

            for i, s_i in enumerate(sorted_idx):
                # Sample n random points from bounding box
                self.mouse_callback_params["class"] = i
                sampled_points = sample_points_from_bounding_box(bounding_boxes[s_i, :], self.mouse_callback_params["n_points_per_scribble"])
                bootstrap_label_image(sampled_points[:, 0], sampled_points[:, 1], self.mouse_callback_params)
                temp_mask[bounding_boxes[s_i, 1]: bounding_boxes[s_i, 1] + bounding_boxes[s_i, 3], bounding_boxes[s_i, 0]: bounding_boxes[s_i, 0] + bounding_boxes[s_i, 2]] = 1

            temp_mask = 1. - temp_mask
            bground_points = sample_points_from_mask(temp_mask, num_points=100)
            bootstrap_label_image(bground_points[:, 0], bground_points[:, 1], self.mouse_callback_params)

    def clear_keyframes(self):
        self.mouse_callback_params["indices_b"] = []
        self.mouse_callback_params["indices_w"] = []
        self.mouse_callback_params["indices_h"] = []
        self.mouse_callback_params["classes"] = []
        self.mouse_callback_params["h_labels"] = []
        self.mouse_callback_params["h_masks"] = []
        self.mouse_callback_params["ims"] = []
        self.mouse_callback_params["kfs"] = []
        self.mouse_callback_params["query_masks"] = []
        self.mouse_callback_params["class"] = 0
        self.mouse_callback_params["h_class"] = np.array(
            [-1] * self.n_classes)
        self.mouse_callback_params["batch_label"] = 0

        self.n_clicks = 0
        self.batch_label = 0
        self.h_level = 0
        self.w_node = 0

    def select_keyframe(self, key, batch_size):
        if key == 100:
            batch = self.mouse_callback_params["batch_label"] + 1
        elif key == 97:
            batch = self.mouse_callback_params["batch_label"] - 1
        elif key == 102: # l for last keyframe
            batch = batch_size - 1
        else:
            batch = key - 190
        if batch < batch_size and batch >= 0:
            self.mouse_callback_params["batch_label"] = batch

    def select_class_flat(self, key, batch_label, robot_label=None):
        print("key: ", key)
        if key == 119:
            if self.mouse_callback_params["class"] + 1 < self.n_classes:
                key_no = self.mouse_callback_params["class"] + 1
            else:
                key_no = self.mouse_callback_params["class"]
        elif key == 115:
            if self.mouse_callback_params["class"] > 0:
                key_no = self.mouse_callback_params["class"] - 1
            else:
                key_no = self.mouse_callback_params["class"]
        else:
            key_no = key - 49

        self.mouse_callback_params["class"] = key_no
        print(f"Semantic class: {key_no + 1}")
        print("self.do_label_auto: ", self.do_label_auto)
        if self.do_names:
            if self.label_names[key_no] == "-":
                name = input("Class name: ")
                self.label_names[key_no] = name

        if self.do_label_auto and key_no is not None:
            self.mouse_callback_params["indices_b"].append(
                self.batch_auto)
            self.mouse_callback_params["indices_h"].append(
                self.pixel_auto[1])
            self.mouse_callback_params["indices_w"].append(
                self.pixel_auto[0])
            self.mouse_callback_params["classes"].append(key_no)
            im = self.mouse_callback_params["ims"][self.batch_auto]
            palette = self.mouse_callback_params["col_palette"]
            color = palette[key_no]
            cv2.circle(
                im, (self.pixel_auto[0], self.pixel_auto[1]), 4,
                color.astype(np.uint8).tolist(), -1)
            self.do_label_auto = False
            self.n_label_pause = 0

            if self.do_analysis:
                f = open("clicks.txt", "a")
                text = str(self.n_clicks) + " " \
                    + str(key_no) + " " \
                    + str(self.batch_auto) + " " \
                    + str(self.pixel_auto[0]) + " " \
                    + str(self.pixel_auto[1]) + "\n"
                f.write(text)
                f.close()

    def select_class_hier(self, key, batch_label):
        key_no = key - 49

        h_class = self.mouse_callback_params["h_class"][key_no]
        if h_class == 1:
            self.mouse_callback_params["h_class"][key_no] = -1
        else:
            self.mouse_callback_params["h_class"][key_no] += 1

        label_idx, self.h_level = sem_idx_hier(
            self.mouse_callback_params["h_class"])

        self.mouse_callback_params["class"] = label_idx

        if self.do_names:
            if self.label_names[label_idx] == "-":
                name = input("Class name: ")
                self.label_names[label_idx] = name

        self.tree_im = visualisation.draw.draw_tree(
            500, 500, self.n_classes + 1,
            self.mouse_callback_params["col_palette"],
            label_idx, None
        )
        cv2.imshow("tree hierarchy", self.tree_im)

    def select_class_hier_decimal(self, key, batch_label):
        if key == 119:
            if self.h_level + 1 < self.n_classes:
                self.h_level = self.h_level + 1
                self.w_node = 0
        elif key == 115:
            if self.h_level > 0:
                self.h_level = self.h_level - 1
                self.w_node = 0
        else:
            max_node = 2**(self.h_level + 1)
            w_node = key - 49
            if w_node < max_node:
                self.w_node = w_node

        prev_nodes = 2**(self.h_level + 1) - 2
        label_idx = prev_nodes + self.w_node
        self.mouse_callback_params["class"] = label_idx

        if self.do_names:
            if self.label_names[label_idx] == "-":
                name = input("Class name: ")
                self.label_names[label_idx] = name

        binary = "{0:b}".format(self.w_node)
        binary = "0" * (self.h_level + 1 - len(binary)) + binary

        for idx, s in enumerate(binary):
            self.mouse_callback_params["h_class"][idx] = int(s)

        for idx_remainder in range(idx + 1, self.n_classes):
            self.mouse_callback_params["h_class"][idx_remainder] = -1

        self.tree_im = visualisation.draw.draw_tree(
            500, 500, self.n_classes + 1,
            self.mouse_callback_params["col_palette"],
            label_idx, None
        )
        cv2.imshow("tree hierarchy", self.tree_im)

    def select_class(self, key, batch_label, robot_label=None):
        if self.do_hierarchical:
            self.select_class_hier_decimal(key, batch_label)
        else:
            self.select_class_flat(key, batch_label, robot_label=robot_label)
        print(f"Label class: {key}")

    def send_params_mapping(self, vis_to_map_labels):
        if len(self.mouse_callback_params["indices_b"]) > self.n_clicks:
            print("clicks:", self.n_clicks)
            self.n_clicks = len(self.mouse_callback_params["indices_b"])
            vis_to_map_labels.put(
                (self.mouse_callback_params["indices_b"],
                 self.mouse_callback_params["indices_h"],
                 self.mouse_callback_params["indices_w"],
                 self.mouse_callback_params["classes"],
                 self.mouse_callback_params["h_labels"],
                 self.mouse_callback_params["h_masks"])
            )

            return True
        else:
            return False

    def label_auto(self, T_WC_track, render_trainer):
        if T_WC_track is not None:
            self.batch_auto = self.mouse_callback_params["batch_label"]
            T_WC_kf = T_WC_track[self.batch_auto]
            query_mask_auto = self.mouse_callback_params["query_masks"][self.batch_auto]
            kf_im_resize = render_trainer.kfs_im[self.batch_auto]

            self.pixel_auto, entropy_avg_kf_vis, entropy_kf_vis = self.query_selector(render_trainer, T_WC_kf,
                                                                                      query_mask_auto, img=kf_im_resize)

            batch_label = self.mouse_callback_params["batch_label"]
            cv2.circle(
                self.mouse_callback_params["ims"][batch_label],
                (self.pixel_auto[0], self.pixel_auto[1]), 5, [0, 0, 0], 2
            )
            entropy_avg_kf_vis = cv2.cvtColor(
                entropy_avg_kf_vis, cv2.COLOR_BGR2RGB)
            entropy_kf_vis = cv2.cvtColor(entropy_kf_vis, cv2.COLOR_BGR2RGB)
            cv2.imshow("entropy kf_avg", entropy_avg_kf_vis)
            cv2.imshow("entropy kf", entropy_kf_vis)
            self.do_label_auto = True

    def draw_kinematic_limits(self, T_WC_track, render_trainer, x_limits, y_limits):
        T_WC_kf = T_WC_track[self.batch_auto]
        T_WC_kf_ = T_WC_kf[None, ...]
        T_WC_kf_np = T_WC_kf_.cpu().numpy()
        im = self.mouse_callback_params["ims"][self.batch_auto]

        limit_points = [np.array([x_limits[0], y_limits[0], 0., 1.]),
                    np.array([x_limits[1], y_limits[0], 0., 1.]),
                    np.array([x_limits[1], y_limits[1], 0., 1.]),
                    np.array([x_limits[0], y_limits[1], 0., 1.])]

        limit_pixels = []
        for limit_point in limit_points:
            limit_pixel = raster_point.raster_point(limit_point,
                        T_WC_kf_np,
                        render_trainer.W,
                        render_trainer.H,
                        render_trainer.fx,
                        -render_trainer.fy,
                        render_trainer.cx,
                        render_trainer.cy,
            )
            limit_pixels.append(limit_pixel)

        cv2.line(im, limit_pixels[0], limit_pixels[1], (255, 255, 255), thickness=4)
        cv2.line(im, limit_pixels[1], limit_pixels[2], (255, 255, 255), thickness=4)
        cv2.line(im, limit_pixels[2], limit_pixels[3], (255, 255, 255), thickness=4)
        cv2.line(im, limit_pixels[3], limit_pixels[0], (255, 255, 255), thickness=4)
        return True

    def add_hessian_plane(self, T_WC_track, render_trainer, pcd, pcd_mask):

        border_frac = 0.05
        plane_uvs = [[int(border_frac*render_trainer.W_vis), int(border_frac*render_trainer.H_vis)],
                    [int(border_frac*render_trainer.W_vis), int(render_trainer.H_vis - border_frac*render_trainer.H_vis)],
                    [int(render_trainer.W_vis - border_frac*render_trainer.W_vis), int(render_trainer.H_vis - border_frac*render_trainer.H_vis)],
                    [int(render_trainer.W_vis - border_frac*render_trainer.W_vis), int(border_frac*render_trainer.H_vis)]]

        print("plane_uvs: ", plane_uvs)
        print("pcd.shape: ", pcd.shape)

        pcd_plane = []
        for plane_uv in plane_uvs:
            print("int(plane_uv[0] + plane_uv[1]*render_trainer.H_vis): ", int(plane_uv[0] + plane_uv[1]*render_trainer.H_vis))
            pcd_plane.append(pcd[int(plane_uv[0] + plane_uv[1]*render_trainer.W_vis)])
        pcd_plane = np.array(pcd_plane)
        pcd_plane_arg = np.arange(pcd_plane.shape[0])
        self.hessian_plane = geometry.plane_fit.hessian_fit(pcd_plane[pcd_plane_arg])

        ## Draw points used
        T_WC_kf = T_WC_track[self.batch_auto]
        T_WC_kf_ = T_WC_kf[None, ...]
        T_WC_kf_np = T_WC_kf_.cpu().numpy()

        for point_ in pcd_plane[pcd_plane_arg]:

            point = np.copy(point_)
            point = np.append(point, 1)
            pixel = raster_point.raster_point(point,
                        T_WC_kf_np,
                        render_trainer.W,
                        render_trainer.H,
                        render_trainer.fx,
                        -render_trainer.fy,
                        render_trainer.cx,
                        render_trainer.cy,
            )

        pcd_mask_plane = map(lambda point: geometry.plane_fit.point_on_plane(point, self.hessian_plane), pcd)
        pcd_mask_plane = np.array(list(pcd_mask_plane))

        pcd_mask_plane = pcd_mask_plane.reshape((render_trainer.H_vis, render_trainer.W_vis))

        mask_plane_vis = pcd_mask_plane.astype('uint8')*255

        mask_plane_vis = cv2.cvtColor(mask_plane_vis,
                                    cv2.COLOR_BGR2RGB,
        )
        cv2.imshow("plane mask", mask_plane_vis)
        return pcd_mask_plane

    def mask_edges_corners(self, T_WC_track, render_trainer, depth, normals):
        ## Determine curvature
        normals_ = np.reshape(normals, (render_trainer.H_vis, render_trainer.W_vis, 3))

        normals_c = np.roll(normals_, -1, axis=0)
        normals_r = np.roll(normals_, -1, axis=1)

        normals_c_dot = np.zeros(normals_.shape[:2])
        normals_r_dot = np.zeros(normals_.shape[:2])
        normals_dot = np.zeros(normals_.shape[:2])
        for i in range(0, normals_.shape[0]):
            for j in range(0, normals_.shape[1]):
                normals_c_dot[i,j] = np.dot(normals_c[i,j], normals_[i,j])
                normals_r_dot[i,j] = np.dot(normals_r[i,j], normals_[i,j])
                normals_dot[i,j] = np.linalg.norm(np.array([normals_c_dot[i,j], normals_r_dot[i,j]]))
                if normals_dot[i,j] > np.sqrt(2):
                    normals_dot[i,j] = 2*np.sqrt(2) - normals_dot[i,j]
                ## normalise
                normals_dot[i,j] = 1 - normals_dot[i,j]/np.sqrt(2)

        depth_c = np.diff(depth, axis=0, prepend=depth[0,:][None,:])
        depth_r = np.diff(depth, axis=1, prepend=depth[:,0][:,None])
        ## normalise
        depth_c = depth_c/depth_c.max()
        depth_r = depth_r/depth_r.max()
        depth = np.copy(depth)
        for i in range(0, depth.shape[0]):
            for j in range(0, depth.shape[1]):
                depth[i,j] = np.linalg.norm(np.array([depth_c[i,j], depth_r[i,j]]))
                if depth[i,j] > np.sqrt(2):
                    depth[i,j] = 2*np.sqrt(2) - normals_dot[i,j]
                ## normalise
                depth[i,j] = depth[i,j]/np.sqrt(2)

        ## Cut on regions to avoid
        depth[depth > 0.03] = 1.
        normals_dot[normals_dot > 0.02] = 1.
        ## Add both together
        edges_mask = normals_dot + depth
        edges_mask = edges_mask/edges_mask.max()
        edges_mask[edges_mask > 0.03] = 1.

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges_mask = cv2.dilate(edges_mask, kernel, iterations=1)
        edges_mask = np.greater(edges_mask, np.full(edges_mask.shape, 0.5))

        return edges_mask

    def create_mask_auto(self, T_WC_track, render_trainer):

        T_WC_kf = T_WC_track[self.batch_auto]
        query_mask_auto = self.mouse_callback_params["query_masks"][self.batch_auto]
        kf_im_resize = render_trainer.kfs_im[self.batch_auto]
        kf_depth_resize = render_trainer.kfs_depth[self.batch_auto]

        pcd_in_camera = render_trainer.pcs_gt_cam[self.batch_auto]
        T_WC_kf_np = T_WC_kf.cpu().numpy()

        camera_in_world = T_WC_kf_np
        self.pcd_in_global = mercury.geometry.transform_points(
            pcd_in_camera,
            camera_in_world,
        )
        self.normals = mercury.geometry.normals_from_pointcloud(self.pcd_in_global)

        x_limits = [0.0, 1.2]
        y_limits = [-0.8, 0.8]

        ## Set query mask to 1 if point out of reach of robot:
        pcdx_less = np.less(self.pcd_in_global[:,0], np.full(self.pcd_in_global.shape[0], x_limits[0]))
        pcdx_greater = np.greater(self.pcd_in_global[:,0], np.full(self.pcd_in_global.shape[0], x_limits[1]))
        pcdx_mask = np.logical_or(pcdx_less, pcdx_greater)

        pcdy_less = np.less(self.pcd_in_global[:,1], np.full(self.pcd_in_global.shape[0], y_limits[0]))
        pcdy_greater = np.greater(self.pcd_in_global[:,1], np.full(self.pcd_in_global.shape[0], x_limits[1]))
        pcdy_mask = np.logical_or(pcdy_less, pcdy_greater)

        pcd_mask = np.logical_or(pcdx_mask, pcdy_mask)


        # ## Based on pointcloud normal
        if self.vertical:

            table_plane = np.array([0.,0.,1.])
            normals_dot = np.full(self.normals.shape[0], 0.)
            for i in range(0, normals_dot.shape[0]):
                normals_dot[i] = np.abs(np.dot(self.normals[i], table_plane))
            normal_less = np.less(normals_dot, np.full(normals_dot.shape, 0.9))

            pcd_mask = np.logical_or(pcd_mask, normal_less)

        pcd_mask = pcd_mask.reshape((render_trainer.H_vis, render_trainer.W_vis))

        ## Determine hessian plane
        if self.hessian_plane is None: # and self.label_plane:
            pcd_plane = self.add_hessian_plane(T_WC_track, render_trainer, self.pcd_in_global, pcd_mask)

        if not self.label_plane:
            pcd_mask = np.logical_or(pcd_mask, pcd_plane)




        if self.mask_edges:
            edges_mask = self.mask_edges_corners(T_WC_track, render_trainer, kf_depth_resize, self.normals)
            pcd_mask = np.logical_or(pcd_mask, edges_mask)

        # pcd_mask = np.logical_not(pcd_mask)
        pcd_mask = pcd_mask.astype('uint8')*255
        pcd_mask = cv2.resize(pcd_mask, (render_trainer.W, render_trainer.H), interpolation = cv2.INTER_AREA)

        pcd_mask = np.greater(pcd_mask, np.full(pcd_mask.shape, 100))
        query_mask_auto = np.greater(query_mask_auto, np.full(query_mask_auto.shape, 100))
        pcd_mask_vis = query_mask_auto.astype('uint8')*255

        query_mask_auto = np.logical_or(query_mask_auto, pcd_mask)
        pcd_mask_vis = query_mask_auto.astype('uint8')*255

        query_mask_auto = query_mask_auto.astype('uint8')*255

        return query_mask_auto


    def label_auto_robot(self, T_WC_track, render_trainer, robot_target, robot_label):
        if T_WC_track is None:
            return

        if self.n_clicks > 10 and self.label_plane == False:
            print("Labelling plane.........")
            self.label_plane = True
            query_mask_auto = np.full(self.mouse_callback_params["query_masks"][self.batch_auto].shape, 0)
            query_mask_auto = query_mask_auto.astype('uint8')*255
            self.mouse_callback_params["query_masks"][self.batch_auto] = query_mask_auto
            query_mask_auto = self.create_mask_auto(T_WC_track, render_trainer)
            self.mouse_callback_params["query_masks"][self.batch_auto] = query_mask_auto

        if self.do_analysis:
            if self.n_clicks == 0:
                batch_label = self.mouse_callback_params["batch_label"]
                label_kf = self.mouse_callback_params["ims"][batch_label]
                label_kf = cv2.cvtColor(label_kf, cv2.COLOR_BGR2RGB)
                cv2.imwrite("label_kf_" + str(self.n_clicks) + ".png", label_kf)
            # else:
            surface_normals = self.render_hq_(render_trainer)

        if self.mouse_callback_params["batch_label"] != self.batch_auto:
            self.batch_auto = self.mouse_callback_params["batch_label"]
            query_mask_auto = self.create_mask_auto(T_WC_track, render_trainer)
            self.mouse_callback_params["query_masks"][self.batch_auto] = query_mask_auto

            self.edge_mask = edge_detection.determine_edge_mask(self.mouse_callback_params["ims"][self.batch_auto])

            if self.do_analysis:
                ## keyframe
                kf_im = render_trainer.kfs_im[self.batch_auto]
                kf_im_vis = cv2.cvtColor(
                    kf_im, cv2.COLOR_BGR2RGB)
                cv2.imwrite("kf_im_vis" + str(self.n_clicks) + ".png", kf_im_vis)

                ## mask
                kf_im = render_trainer.kfs_im[self.batch_auto]
                query_mask_auto_vis = cv2.cvtColor(
                    query_mask_auto, cv2.COLOR_BGR2RGB)
                cv2.imwrite("query_mask_auto_vis" + str(self.n_clicks) + ".png", query_mask_auto_vis)


        T_WC_kf = T_WC_track[self.batch_auto]
        query_mask_auto = self.mouse_callback_params["query_masks"][self.batch_auto]
        kf_im_resize = render_trainer.kfs_im[self.batch_auto]
        kf_depth_resize = render_trainer.kfs_depth[self.batch_auto]


        plane_norm = np.array([0.,0.,1.])
        self.success_query = False
        while not self.success_query:

            ## Now returns an array of pixels
            self.pixel_auto, entropy_avg_kf_vis, entropy_kf_vis = self.query_selector(render_trainer, T_WC_kf,
                            query_mask_auto, img=kf_im_resize, edge_mask=self.edge_mask,
            )

            u_vis = self.pixel_auto[0]*(render_trainer.W_vis/render_trainer.W)
            v_vis = self.pixel_auto[1]*(render_trainer.H_vis/render_trainer.H)
            u_vis = u_vis.astype(int)
            v_vis = v_vis.astype(int)
            n_points = v_vis*render_trainer.W_vis + u_vis
            n_points = n_points.astype(int)

            query_pixel_im = np.copy(self.mouse_callback_params["ims"][self.batch_auto])
            query_pixel_im = np.copy(render_trainer.frames.im_batch_np[self.batch_auto])
            for n_point in range(0, len(n_points)):
                cv2.circle(
                    query_pixel_im, (self.pixel_auto[0][n_point], self.pixel_auto[1][n_point]), 5,
                    [0, 0, 0], 2)
            query_pixel_im_vis = cv2.cvtColor(query_pixel_im, cv2.COLOR_BGR2RGB)
            cv2.imshow("query_pixel_im", query_pixel_im_vis)

            poses = []
            for n_point in range(0, len(n_points)):
                point = self.pcd_in_global[n_points[n_point]]

                normal = surface_normals[self.pixel_auto[1], self.pixel_auto[0]][0]
                theta = np.arccos(np.dot(plane_norm, normal)/np.linalg.norm(plane_norm)*np.linalg.norm(normal))
                quat = mercury.geometry.quaternion_from_vec2vec([0.,0.,1.], normal)
                pose = mercury.geometry.transformation_matrix(point, quat)
                angle_thres = 30 ## To force top down interaction

                if self.label_plane:
                    if geometry.plane_fit.point_on_plane(point, self.hessian_plane):
                        pose = [pose, None]
                        poses.append(pose)
                    else:
                        poses.append(pose)
                else:
                    poses.append(pose)

            ## Send to robot
            _ = get_latest_queue(robot_label) ## First clear pipe
            robot_target.put(poses, block=False)
            self.success_query = True

        self.do_label_auto = True

        return

    def render_hq(self, render_trainer):
        batch_label = self.mouse_callback_params["batch_label"]
        kf = self.mouse_callback_params["kfs"][batch_label]
        T_WC_kf = render_trainer.frames.T_WC_track[batch_label]
        depth, _, col, sem_pred = render_rays.render_images_chunks(
            T_WC_kf[None, ...],
            render_trainer.min_depth,
            render_trainer.max_depth,
            render_trainer.n_embed_funcs,
            render_trainer.n_bins,
            render_trainer.fc_occ_map,
            render_trainer.B_layer,
            H=render_trainer.H,
            W=render_trainer.W,
            fx=render_trainer.fx,
            fy=render_trainer.fy,
            cx=render_trainer.cx,
            cy=render_trainer.cy,
            grad=False,
            dirs_C=render_trainer.dirs_C.view(1, -1, 3),
            do_fine=render_trainer.do_fine,
            do_var=False,
            do_color=render_trainer.do_color,
            do_sem=render_trainer.do_sem,
            n_bins_fine=render_trainer.n_bins_fine_vis,
        )

        col = col.view(
            render_trainer.H, render_trainer.W, 3)
        col = (col.cpu().numpy() * 255).astype(np.uint8)
        col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)

        depth = depth.view(
            render_trainer.H, render_trainer.W)
        depth = depth.cpu().numpy()
        depth = imgviz.depth2rgb(depth)

        if render_trainer.do_hierarchical:
            sem_pred = torch.sigmoid(sem_pred)
        else:
            sem_pred = torch.nn.functional.softmax(sem_pred, dim=1)
        sem_pred = sem_pred.view(
            render_trainer.H, render_trainer.W, render_trainer.n_classes)

        sem_vis_kf, label_viz_kf, kf_entropy, label_masks = self.get_vis_sem(
            sem_pred, self.mouse_callback_params["ims"][batch_label], font_size=30)

        label_viz_kf = cv2.cvtColor(label_viz_kf, cv2.COLOR_BGR2RGB)

        class_idx = self.mouse_callback_params["class"]
        if self.do_names:
            class_name = self.label_names[class_idx]
        else:
            class_name = ""

        label_viz_kf = imgviz.draw.text_in_rectangle(
            label_viz_kf,
            "lt",
            "  " + class_name,
            35,
            [220, 220, 220],
            aabb1=[10, 10],
        )
        color = self.mouse_callback_params["col_palette"][class_idx]
        label_viz_kf = imgviz.draw.rectangle(
            label_viz_kf, [13, 12], [41, 40], fill=color[::-1])

        sem_vis = cv2.cvtColor(sem_vis_kf[0], cv2.COLOR_BGR2RGB)

        label_kf = self.mouse_callback_params["ims"][batch_label]
        label_kf = cv2.cvtColor(label_kf, cv2.COLOR_BGR2RGB)

        if not self.headless:
            cv2.imshow("col_hq", col)
            cv2.imwrite("col_hq.png", col)
            cv2.imshow("depth_hq", depth)
            cv2.imwrite("depth_hq.png", depth)
            cv2.imshow("label_kf_hq", label_viz_kf)
            cv2.imwrite("label_kf_hq.png", label_viz_kf)
            cv2.imshow("sem_vis_kf", sem_vis)
            cv2.imwrite("sem_vis_kf_hq.png", sem_vis)
            cv2.imwrite("label_kf.png", label_kf)
            return None
        else:
            return col, depth, label_viz_kf, sem_vis, label_kf

    def render_hq_(self, render_trainer, do_3D=True):
        batch_label = self.mouse_callback_params["batch_label"]
        kf = self.mouse_callback_params["kfs"][batch_label]
        T_WC_kf = render_trainer.frames.T_WC_track[batch_label]
        depth, _, col, sem_pred = render_rays.render_images_chunks(
            T_WC_kf[None, ...],
            render_trainer.min_depth,
            render_trainer.max_depth,
            render_trainer.n_embed_funcs,
            render_trainer.n_bins,
            render_trainer.fc_occ_map,
            render_trainer.B_layer,
            H=render_trainer.H,
            W=render_trainer.W,
            fx=render_trainer.fx,
            fy=render_trainer.fy,
            cx=render_trainer.cx,
            cy=render_trainer.cy,
            grad=False,
            dirs_C=render_trainer.dirs_C.view(1, -1, 3),
            do_fine=render_trainer.do_fine,
            do_var=False,
            do_color=render_trainer.do_color,
            do_sem=render_trainer.do_sem,
            n_bins_fine=render_trainer.n_bins_fine_vis,
        )

        if render_trainer.do_hierarchical:
            sem_pred = torch.sigmoid(sem_pred)
        else:
            sem_pred = torch.nn.functional.softmax(sem_pred, dim=1)
        sem_pred = sem_pred.view(
            render_trainer.H, render_trainer.W, render_trainer.n_classes)

        sem_vis_kf, label_viz_kf, kf_entropy, label_masks = self.get_vis_sem(
            sem_pred, self.mouse_callback_params["ims"][batch_label], font_size=30)

        label_viz_kf = cv2.cvtColor(label_viz_kf, cv2.COLOR_BGR2RGB)

        class_idx = self.mouse_callback_params["class"]
        if self.do_names:
            class_name = self.label_names[class_idx]
        else:
            class_name = ""



        if do_3D:
            origins_dirs = render_rays.origin_dirs_W(
                T_WC_kf[None, ...], render_trainer.dirs_C.view(1, -1, 3))

            surface_normals, diffuse = render_rays.render_normals(
                T_WC_kf[None, ...],
                depth,
                render_trainer.fc_occ_map,
                render_trainer.dirs_C.view(1, -1, 3),
                render_trainer.n_embed_funcs,
                origins_dirs,
                render_trainer.B_layer,
                noise_std=None,
                radius=render_trainer.radius_vis_up,
                do_mip=render_trainer.do_mip
            )
            diffuse = diffuse.view(
                render_trainer.H,
                render_trainer.W).detach().cpu().numpy()

            diffuse = np.repeat(diffuse[:, :, np.newaxis], 3, axis=2)
            diffuse_im = diffuse
            specular_im = (diffuse**40)
            sem_vis = sem_vis_kf[0] / 255.
            phong_sem = sem_vis * 0.6 + diffuse_im * 0.2 + specular_im * 0.2
            phong_sem = np.clip(phong_sem, 0., 1.)
            phong_sem = (phong_sem * 255).astype(np.uint8)

            phong_sem = cv2.cvtColor(phong_sem, cv2.COLOR_BGR2RGB)

            surface_normals = surface_normals.view(
                        render_trainer.H,
                        render_trainer.W, 3).detach().cpu().numpy()
            surface_normals_vis = (-surface_normals + 1.) / 2.
            surface_normals_vis = np.clip(surface_normals_vis, 0., 1.)
            surface_normals_vis = (surface_normals_vis * 255).astype(np.uint8)
            surface_normals_vis = cv2.cvtColor(surface_normals_vis, cv2.COLOR_BGR2RGB)

        sem_vis = cv2.cvtColor(sem_vis_kf[0], cv2.COLOR_BGR2RGB)

        entropy_np = kf_entropy[0]
        entropy_vis = imgviz.depth2rgb(kf_entropy[0])
        entropy_vis = cv2.cvtColor(entropy_vis, cv2.COLOR_BGR2RGB)

        color_map = (self.mouse_callback_params["col_palette"] * 1.0).astype(np.uint8) # 0.7. 0.5
        sem_masks_vis = color_map[label_masks]
        sem_masks_vis = cv2.cvtColor(sem_masks_vis, cv2.COLOR_BGR2RGB)

        ## RGB
        label_kf = self.mouse_callback_params["ims"][batch_label]
        label_kf = cv2.cvtColor(label_kf, cv2.COLOR_BGR2RGB)

        ##Depth
        depth_kf_np = render_trainer.frames.depth_batch_np[batch_label]
        depth_kf_vis = imgviz.depth2rgb(depth_kf_np)

        if not self.headless:
            cv2.imwrite("label_kf_hq_" + str(self.n_clicks) + ".png", label_viz_kf)
            cv2.imwrite("sem_vis_kf_hq_" + str(self.n_clicks) + ".png", sem_vis)
            cv2.imwrite("label_kf_" + str(self.n_clicks) + ".png", label_kf)

            cv2.imwrite("depth_kf_vis" + str(self.n_clicks) + ".png", depth_kf_vis)
            np.save("depth_kf_np" + str(self.n_clicks) + ".npy", depth_kf_np)

            cv2.imwrite("entropy_vis" + str(self.n_clicks) + ".png", entropy_vis)
            np.save("entropy_np" + str(self.n_clicks) + ".npy", entropy_np)

            cv2.imwrite("sem_masks_vis" + str(self.n_clicks) + ".png", sem_masks_vis)

            return surface_normals
        else:
            return col, depth, label_viz_kf, sem_vis, label_kf


    def kf_vis(self, render_trainer):
        batch_label = self.mouse_callback_params["batch_label"]
        T_WC_kf = render_trainer.frames.T_WC_track[batch_label]
        kf_view_depths, kf_view_vars, kf_view_cols, kf_view_sems = trainer.render_vis(
            1,
            render_trainer,
            T_WC_kf[None, ...],
            render_trainer.do_fine,
            do_var=True,
            do_color=render_trainer.do_color,
            do_sem=render_trainer.do_sem,
            do_hierarchical=render_trainer.do_hierarchical
        )

        sem_pred = kf_view_sems[0]
        print("render_trainer.kfs_im[batch_label].shape: ", render_trainer.kfs_im[batch_label].shape)
        sem_vis_kf, label_viz_kf, kf_entropy, label_masks = self.get_vis_sem(
            sem_pred, render_trainer.kfs_im[batch_label])

        if render_trainer.do_color:
            render_col = kf_view_cols[0]
        else:
            render_col = label_viz_kf

        kf_viz = trainer.live_vis(
            render_trainer.kfs_depth[batch_label],
            kf_view_depths[0].cpu().numpy(),
            kf_view_vars[0],
            view_cols=render_col,
            im_batch_np=label_viz_kf,
            sem_ims=sem_vis_kf,
            entropies=kf_entropy
        )
        kf_viz = cv2.cvtColor(kf_viz, cv2.COLOR_BGR2RGB)

        if not self.headless:
            cv2.imshow("kf vis", kf_viz)

        return label_masks, kf_viz

    def draw_label_kf(self, label_kf, label_masks=None, render_trainer=None):
        names = None
        if self.do_names:
            names = self.label_names

        if label_masks is not None:
            label_masks_resize = imgviz.resize(
                label_masks,
                width=render_trainer.W,
                height=render_trainer.H,
                interpolation="nearest"
            )

            label_kf = imgviz.label2rgb(
                label_masks_resize,
                image=label_kf,
                colormap=(
                    self.mouse_callback_params["col_palette"] * 0.5).astype(np.uint8),
                label_names=names,
                font_size=25
            )

        label_kf = cv2.cvtColor(
            label_kf, cv2.COLOR_BGR2RGB)

        class_idx = self.mouse_callback_params["class"]
        if self.do_hierarchical:
            for level in range(self.n_classes):
                cv2.circle(
                    label_kf,
                    (20 * (level + 1), 40), 8,
                    self.h_class_to_col[self.mouse_callback_params["h_class"][level]],
                    -1
                )

        if self.do_names:
            class_name = self.label_names[class_idx]
        else:
            class_name = ""

        label_kf = imgviz.draw.text_in_rectangle(
            label_kf,
            "lt",
            "  " + class_name,
            15,
            [220, 220, 220],
            aabb1=[10, 10],
        )
        color = self.mouse_callback_params["col_palette"][class_idx]
        label_kf = imgviz.draw.rectangle(
            label_kf, [13, 12], [23, 22], fill=color[::-1])

        return label_kf

    def select_closest_kf(self, render_trainer):
        if render_trainer.frames.T_WC_track is not None:
            if render_trainer.frames.T_WC_track.shape[0] > 1:
                T_WC_kfs = render_trainer.frames.T_WC_track[:-1]
                T_CW = torch.inverse(render_trainer.T_WC)
                T_rel = torch.matmul(T_WC_kfs, T_CW)
                t_rel = T_rel[:, :3, 3]
                R_rel = T_rel[:, :3, :3]
                q_rel = matrix_to_quaternion(R_rel)
                q_rel_real = q_rel[:, 0]
                angle_dist = (2 * torch.acos(q_rel_real)) / np.pi
                trans_dist = torch.norm(t_rel, dim=1) / 5
                transf_dist = angle_dist + trans_dist
                self.mouse_callback_params["batch_label"] = torch.argmin(
                    transf_dist).item()

    def label_cam_center(self, render_trainer):
        u_Cf = (render_trainer.W_vis - 1) // 2
        v_Cf = (render_trainer.H_vis - 1) // 2

        d = render_trainer.latest_depth[v_Cf, u_Cf].item()
        if d > 0:
            x_Cf = ((u_Cf - render_trainer.cx_vis) /
                    render_trainer.fx_vis) * d
            y_Cf = ((v_Cf - render_trainer.cy_vis) /
                    render_trainer.fy_vis) * d
            z_Cf = d

            p_Cf = torch.tensor([x_Cf, y_Cf, z_Cf],
                                device=render_trainer.device).float()

            batch_label = self.mouse_callback_params["batch_label"]
            T_WCf = render_trainer.T_WC
            T_WCkf = render_trainer.frames.T_WC_track[batch_label]
            T_CkfW = torch.inverse(T_WCkf)
            T_CkfCf = torch.matmul(T_CkfW, T_WCf)

            R_CkfCf = T_CkfCf[0, :3, :3]
            p_Ckf = torch.matmul(R_CkfCf, p_Cf)
            p_Ckf += T_CkfCf[0, :3, 3]
            u_Ckf = (p_Ckf[0] / p_Ckf[2] *
                     render_trainer.fx + render_trainer.cx)
            v_Ckf = (p_Ckf[1] / p_Ckf[2] *
                     render_trainer.fy + render_trainer.cy)

            if (u_Ckf > 0 and u_Ckf < render_trainer.W) and (v_Ckf > 0 and v_Ckf < render_trainer.H):
                label_image("center",
                            int(u_Ckf),
                            int(v_Ckf),
                            None, self.mouse_callback_params)

    def label(
            self, batch_size, key, vis_to_map_labels,
            render_trainer=None, do_kf_vis=True,
            vis_to_map_rois = None,
            robot_target=None,
            robot_label=None,
    ):
        pixel_clicked = False

        robot_auto_label = False
        # print("type(robot_label): ", type(robot_label))
        if robot_label is not None and self.do_continuous:
            robot_return = get_latest_queue(robot_label)

            if robot_return is not None:
                key = robot_return[0]
                point = robot_return[1]

                print("label key: ", key)
                print("label point: ", point)

                if self.batch_auto is None:
                    self.batch_auto = self.mouse_callback_params["batch_label"]
                    T_WC_track = render_trainer.frames.T_WC_track
                    query_mask_auto = self.create_mask_auto(T_WC_track, render_trainer)
                    self.mouse_callback_params["query_masks"][self.batch_auto] = query_mask_auto

                if isinstance(point, int):
                    query_mask_auto = self.mouse_callback_params["query_masks"][self.batch_auto]
                    for n_pixel in range(0, point+1):
                        cv2.circle(
                            query_mask_auto,
                            (self.pixel_auto[0][n_pixel], self.pixel_auto[1][n_pixel]), 20, 255, -1
                        )
                    self.mouse_callback_params["query_masks"][self.batch_auto] = query_mask_auto
                    self.pixel_auto[0] = self.pixel_auto[0][point]
                    self.pixel_auto[1] = self.pixel_auto[1][point]
                else:
                    point = np.append(point, 1)
                    T_WC_kf = render_trainer.frames.T_WC_track[self.batch_auto]
                    T_WC_kf = T_WC_kf[None, ...]
                    T_WC_kf = T_WC_kf.cpu().numpy()
                    pixel = raster_point.raster_point(point,
                                T_WC_kf,
                                render_trainer.W,
                                render_trainer.H,
                                render_trainer.fx,
                                -render_trainer.fy, ## Needs to be minus when replaying rosbags!
                                render_trainer.cx,
                                render_trainer.cy,
                    )
                    self.pixel_auto = [pixel[0], pixel[1]]

                if key == -99:
                    self.do_label_auto = False
                else:
                    self.n_label += 1

                robot_auto_label = True

                batch_label = self.mouse_callback_params["batch_label"]

        if self.mouse_callback_params["ims"] or robot_auto_label:
            batch_label = self.mouse_callback_params["batch_label"]
            label_kf = self.mouse_callback_params["ims"][batch_label]
            label_masks = None
            if do_kf_vis:
                label_masks, _ = self.kf_vis(render_trainer)
            label_kf = self.draw_label_kf(
                label_kf, label_masks, render_trainer=render_trainer)
            cv2.imshow("label", label_kf)

            # space bar to label pixel at camera center
            if self.label_cam:
                if key == 32:
                    self.label_cam_center(render_trainer)

            # 1 to 9 keys to select class or 'w', 's' keys to move class selection
            if self.do_hierarchical:
                n_keys = 2**self.n_classes
            else:
                n_keys = self.n_classes

            if key >= 49 and key < 49 + n_keys or key == 119 or key == 115:
                print("Selecting class...")
                self.select_class(key, batch_label, robot_label=robot_label)

            # F1 to F12 keys to select keyframe or 'a', 'd' keys to move keyframe selection, 'f' for final (latest) keyframe
            elif (key >= 190 and key <= 201) or key == 97 or key == 100 or key == 102:
                self.select_keyframe(key, batch_size)

            # Key 'j' for stoping continous
            elif key == 106:
                self.do_continuous = False

            # Key 'l' for selecting automatic pixel
            elif key == 108 or self.do_continuous:
                T_WC_track = render_trainer.frames.T_WC_track

                if robot_target is None:
                    self.label_auto(T_WC_track,
                        render_trainer,
                    )
                else:
                    self.do_continuous = True
                    # if self.n_label < 15 and self.n_label_pause > 10 and self.do_label_auto == False:
                    if self.n_label_pause > 20 and self.do_label_auto == False:
                    # if self.do_label_auto == False:
                        time_start = time.time()
                        self.label_auto_robot(T_WC_track,
                            render_trainer,
                            robot_target,
                            robot_label,
                        )
                        time_end= time.time()
                        print("Auto label took: {0:.3f} secs".format((time_end-time_start)))
                    else:
                        self.n_label_pause += 1
            # Press 'i' for high quality render
            elif key == 105:
                self.render_hq(render_trainer)

            # Press r to switch to ROI selection mode for feature visualisation
            elif key == 114 and vis_to_map_rois is not None:
                # wait for ROI selection
                print("selecting ROIS")
                selection_done = False
                roi_key = cv2.imshow("roi_select", label_kf)
                while not selection_done:
                    rois = cv2.selectROIs("roi_select", label_kf, True)
                    # sample points from ROI
                    # populate vis_to_map_rois
                    x_0 = []
                    y_0 = []
                    w = []
                    h = []
                    b = []
                    if rois is not None:
                        for roi in rois:
                            b.append(batch_label)
                            x_0.append(roi[0])
                            y_0.append(roi[1])
                            w.append(roi[2])
                            h.append(roi[3])
                            # save rois
                            #save a slightly larger region to get borders
                            x_0_new = np.floor(roi[0] * 0.9).astype(np.int32)
                            y_0_new = np.floor(roi[1] * 0.9).astype(np.int32)
                            x_1_new = min(np.floor((roi[0] + roi[2])*1.1).astype(np.int32), label_kf.shape[1])
                            y_1_new = min(np.floor((roi[1] + roi[3])*1.1).astype(np.int32), label_kf.shape[0])
                            image_roi = np.copy(label_kf[y_0_new:y_1_new, x_0_new:x_1_new, :])
                            filename = f""
                            cv2.imwrite(filename + ".png", image_roi)
                            self.roi_count += 1

                        vis_to_map_rois.put((
                            b,
                            x_0,
                            y_0,
                            w,
                            h
                        ))


                    selection_done = True
                cv2.destroyWindow("roi_select")


            if self.label_cam:
                self.select_closest_kf(render_trainer)

            if not self.mouse_callback_params["drawing"]:
                pixel_clicked = self.send_params_mapping(vis_to_map_labels)
            else:
                cv2.imshow("label", label_kf)
                cv2.waitKey(1)
                return False, False

        return pixel_clicked, True

    def get_vis_sem_hier(self, sem_pred, get_entropy=True):
        binary = (torch.round(
            sem_pred).long()).squeeze(-1)

        h_sem_vis = map_color_hierarchcial(
            sem_pred, self.colormap, binary
        )
        sem_vis = [(s.cpu().numpy()) for s in h_sem_vis]

        entropies = None
        if get_entropy:
            entropies = []
            for level in range(self.n_classes):
                sem_pred_lev = sem_pred[..., level:level + 1]
                entropy = render_rays.entropy(sem_pred_lev).cpu().numpy()
                entropies.append(entropy)

        if self.h_level == 0:
            label_im = binary[..., 0].cpu().numpy()
        else:
            decimal = binary_to_decimal(binary, self.h_level + 1)
            n_prev_nodes = 2**(self.h_level + 1) - 2
            label_im = decimal + n_prev_nodes
            label_im = label_im.cpu().numpy()

        return sem_vis, label_im, entropies

    def get_vis_sem_flat(self, sem_pred, get_entropy=True):
        entropy = None
        if get_entropy:
            entropy = render_rays.entropy(sem_pred)
            entropy = [entropy.cpu().numpy()]

        sem_vis = render_rays.map_color(sem_pred, self.colormap)
        sem_vis = [(sem_vis.cpu().numpy())]
        label_im = torch.argmax(sem_pred, axis=2).cpu().numpy()

        return sem_vis, label_im, entropy

    def get_vis_sem(self, sem_pred, rgb, font_size=15, get_entropy=True):
        if self.do_hierarchical:
            sem_vis, label_im, entropy = self.get_vis_sem_hier(
                sem_pred, get_entropy=get_entropy)

        else:
            sem_vis, label_im, entropy = self.get_vis_sem_flat(
                sem_pred, get_entropy=get_entropy)

        names = None
        if self.do_names:
            names = self.label_names

        label_vis = imgviz.label2rgb(
            label_im,
            image=rgb,
            colormap=(
                self.mouse_callback_params["col_palette"] * 0.7).astype(np.uint8),
            label_names=names,
            font_size=font_size,
            loc="rb"
        )

        return sem_vis, label_vis, entropy, label_im
