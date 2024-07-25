#!/usr/bin/env python
import torch

import cv2
import numpy as np
import imgviz
import random

from RobEx.train import trainer
from RobEx.render import render_rays
from RobEx import visualisation


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


def label_image(event, x, y, flags, param):
    # grab references to the global variables
    do_hierarchical = param["do_hierarchical"]
    batch_label = param["batch_label"]
    im = param["ims"][batch_label]
    palette = param["col_palette"]

    if event == cv2.EVENT_LBUTTONDOWN:
        param["indices_b"].append(batch_label)
        param["indices_h"].append(y)
        param["indices_w"].append(x)
        if do_hierarchical:
            h_mask = param["h_class"] != -1
            h_label = param["h_class"].copy()
            param["h_labels"].append(h_label)
            param["h_masks"].append(h_mask)
        else:
            param["classes"].append(param["class"])

        color = palette[param["class"]]

        cv2.circle(im, (x, y), 4, color.astype(np.uint8).tolist(), -1)


class iLabel(object):
    def __init__(self, do_hierarchical, do_names, device, n_classes):
        super(iLabel, self).__init__()
        self.do_hierarchical = do_hierarchical
        self.device = device
        self.do_names = do_names
        self.n_classes = n_classes

        self.do_label_auto = False
        self.n_clicks = 0
        self.batch_label = 0
        self.h_level = 0
        self.n_clicks = 0

        self.mouse_callback_params = {}
        self.mouse_callback_params["indices_b"] = []
        self.mouse_callback_params["indices_w"] = []
        self.mouse_callback_params["indices_h"] = []
        self.mouse_callback_params["classes"] = []
        self.mouse_callback_params["h_labels"] = []
        self.mouse_callback_params["h_masks"] = []
        self.mouse_callback_params["ims"] = []
        self.mouse_callback_params["class"] = 0
        self.mouse_callback_params["h_class"] = np.array(
            [-1] * self.n_classes)
        self.mouse_callback_params["batch_label"] = 0
        self.mouse_callback_params["do_hierarchical"] = self.do_hierarchical

        self.label_names = None
        if self.do_hierarchical:
            self.colormap = torch.tensor(
                [
                    [[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [255, 255, 0]],
                    [[0, 255, 255], [255, 128, 0]],
                    [[255, 0, 255], [128, 0, 0]],
                    [[0, 128, 0], [204, 255, 255]],
                    [[230, 0, 126], [153, 153, 255]],
                    [[235, 150, 135], [120, 120, 120]],
                ],
                device=self.device
            )
            if self.do_names:
                n_nodes = 2**(self.n_classes + 1) - 2
                self.label_names = ["-"] * n_nodes

        else:
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
            if self.do_names:
                self.label_names = ["-"] * self.n_classes

        colormap_np = self.colormap.cpu().numpy().astype(np.uint8)
        colormap_np = colormap_np.reshape(-1, 3)
        self.mouse_callback_params["col_palette"] = colormap_np

        cv2.namedWindow('label', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("label", 1000, 800)
        cv2.setMouseCallback('label', label_image, self.mouse_callback_params)

        self.h_class_to_col = {}
        self.h_class_to_col[-1] = [0, 0, 255]
        self.h_class_to_col[0] = [0, 0, 0]
        self.h_class_to_col[1] = [255, 255, 255]

    def add_keyframe(self, kf):
        self.mouse_callback_params["ims"].append(kf)

    def select_keyframe(self, key, batch_size):
        if key == 100:
            batch = self.mouse_callback_params["batch_label"] + 1
        elif key == 97:
            batch = self.mouse_callback_params["batch_label"] - 1
        else:
            batch = key - 190
        if batch < batch_size and batch >= 0:
            self.mouse_callback_params["batch_label"] = batch

    def select_class_flat(self, key_no, batch_label):
        self.mouse_callback_params["class"] = key_no
        if self.do_names:
            if self.label_names[key_no] == "-":
                name = input("Class name: ")
                self.label_names[key_no] = str(
                    key_no + 1) + ":" + name

        if self.do_label_auto:
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
                color[::-1].astype(np.uint8).tolist(), -1)
            self.do_label_auto = False

    def select_class_hier(self, key_no, batch_label):
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

        tree_im = visualisation.draw.draw_tree(
            500, 500, self.n_classes + 1,
            self.mouse_callback_params["col_palette"],
            label_idx, self.label_names
        )
        cv2.imshow("tree hierarchy", tree_im)

    def select_class(self, key, batch_label):
        if key == 119:
            if self.mouse_callback_params["class"] + 1 < self.n_classes:
                key_no = self.mouse_callback_params["class"] + 1
        elif key == 115:
            if self.mouse_callback_params["class"] > 0:
                key_no = self.mouse_callback_params["class"] - 1
        else:
            key_no = key - 49
        if self.do_hierarchical:
            self.select_class_hier(key_no, batch_label)
        else:
            self.select_class_flat(key_no, batch_label)

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

    def label_auto(self, T_WC_track, render_trainer):
        if T_WC_track is not None:
            self.batch_auto = self.mouse_callback_params["batch_label"]
            T_WC_kf = T_WC_track[self.batch_auto]
            self.pixel_auto, entropy_avg_kf_vis, entropy_kf_vis = select_label_entropy(
                render_trainer, T_WC_kf)

            batch_label = self.mouse_callback_params["batch_label"]
            cv2.circle(
                self.mouse_callback_params["ims"][batch_label],
                (self.pixel_auto[0], self.pixel_auto[1]), 5, [0, 0, 0], 2
            )
            cv2.imshow("entropy kf_avg", entropy_avg_kf_vis)
            cv2.imshow("entropy kf", entropy_kf_vis)
            # loss_approx = loss_approx.sum(dim=(2, 4))
            self.do_label_auto = True

    def kf_vis(self, render_trainer):
        batch_auto = self.mouse_callback_params["batch_label"]
        T_WC_kf = render_trainer.frames.T_WC_track[batch_auto]
        kf_view_depths, kf_view_vars, kf_view_cols, kf_view_sems = trainer.render_vis(
            1,
            render_trainer,
            T_WC_kf[None, ...],
            render_trainer.do_fine,
            do_var=True,
            do_color=True,
            do_sem=render_trainer.do_sem,
            do_hierarchical=render_trainer.do_hierarchical
        )

        sem_pred = kf_view_sems[0]
        sem_vis_kf, label_viz_kf, kf_entropy, label_masks = self.get_vis_sem(
            sem_pred, render_trainer.kfs_im[batch_auto])

        kf_viz = trainer.live_vis(
            render_trainer.kfs_depth[batch_auto],
            kf_view_depths[0],
            kf_view_vars[0],
            view_cols=kf_view_cols[0],
            im_batch_np=label_viz_kf,
            sem_ims=sem_vis_kf,
            entropies=kf_entropy
        )
        kf_viz = cv2.cvtColor(kf_viz, cv2.COLOR_BGR2RGB)
        cv2.imshow("kf vis", kf_viz)

        label_masks_resize = imgviz.resize(
            label_masks,
            width=render_trainer.W,
            height=render_trainer.H,
            interpolation="nearest")

        return label_masks_resize

    def draw_label_kf(self, label_kf, label_masks=None):
        names = None
        if self.do_names:
            names = self.label_names

        if label_masks is not None:
            label_kf = imgviz.label2rgb(
                label_masks,
                img=label_kf,
                colormap=(
                    self.mouse_callback_params["col_palette"] * 0.5).astype(int),
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

    def label(self, batch_size, key, vis_to_map_labels, render_trainer=None, do_kf_vis=True):
        if self.mouse_callback_params["ims"]:
            batch_label = self.mouse_callback_params["batch_label"]
            label_kf = self.mouse_callback_params["ims"][batch_label]
            label_masks = None
            if do_kf_vis:
                label_masks = self.kf_vis(render_trainer)
            label_kf = self.draw_label_kf(label_kf, label_masks)
            cv2.imshow("label", label_kf)

            # 1 to 9 keys to select class or 'w', 's' keys to move class selection
            if key >= 49 and key < 49 + self.n_classes or key == 119 or key == 115:
                self.select_class(key, batch_label)

            # F1 to F12 keys to select keyframe or 'a', 'd' keys to move keyframe selection
            elif (key >= 190 and key <= 201) or key == 97 or key == 100:
                self.select_keyframe(key, batch_size)

            # Key 'l' for selecting automatic pixel
            if key == 108:
                T_WC_track = render_trainer.frames.T_WC_track
                self.label_auto(T_WC_track, render_trainer)

            self.send_params_mapping(vis_to_map_labels)

    def get_vis_sem_hier(self, sem_pred):
        binary = (torch.round(
            sem_pred).long()).squeeze(-1)

        h_sem_vis = map_color_hierarchcial(
            sem_pred, self.colormap, binary
        )
        sem_vis = [(s.cpu().numpy()).astype(np.uint8) for s in h_sem_vis]

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

    def get_vis_sem_flat(self, sem_pred):
        entropy = render_rays.entropy(sem_pred)
        entropy = [entropy.cpu().numpy()]

        sem_vis = render_rays.map_color(sem_pred, self.colormap)
        sem_vis = [(sem_vis.cpu().numpy()).astype(np.uint8)]
        label_im = torch.argmax(sem_pred, axis=2).cpu().numpy()

        return sem_vis, label_im, entropy

    def get_vis_sem(self, sem_pred, rgb):
        if self.do_hierarchical:
            sem_vis, label_im, entropy = self.get_vis_sem_hier(
                sem_pred)

        else:
            sem_vis, label_im, entropy = self.get_vis_sem_flat(
                sem_pred)

        names = None
        if self.do_names:
            names = self.label_names

        label_vis = imgviz.label2rgb(
            label_im,
            img=rgb,
            colormap=(
                self.mouse_callback_params["col_palette"] * 0.8).astype(int),
            label_names=names,
            font_size=15,
            loc="rb"
        )

        return sem_vis, label_vis, entropy, label_im
