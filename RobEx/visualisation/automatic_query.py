import pickle as pkl
import imgviz
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from RobEx.train import trainer
from RobEx.render import render_rays
import skimage
from skimage import measure
from skimage import segmentation

import torchvision


class AnnotationSelector:
    def __init__(self, W, H, W_vis, H_vis, query_strategy="entropy", split_mode="grid", pixel_mode="max",
                top_n_percent=5e-2, grid_num=8, n_pixels_sampled=1):
        """
        Input:
        render_trainer
        T_WC_kf: pose of rendered keyframes
        sem_prob: semantic soft-max probabilities of multiple classes
        img: corresponding images
        grid_num: the number of grid per axis

        query_strategy: ["least_confidence", "margin_sampling", "entropy", "random"]

        split_mode:
            grid: Split the entropy into grids and find the grid with highest entropy.
                  Using fixed sized rectangular regions is a simple option, but most of the time contains more than one object-class,
                  leading to more effort for the annotator to precisely delineate the boundary between objects.
            sp: use super-pixel of colour images to compute regional entropy
            pixel: individually find the suitable pixels


        pixel_mode:
            "max": use the pixel with highest entropy in the super-pixel containing highest average entropy
            "centriod": use the centiod of super-pixel having highest average entropy
            "rand": use the centiod of super-pixel having highest average entropy
    """
        self.query_strategy = query_strategy
        self.split_mode = split_mode
        self.pixel_mode = pixel_mode


        self.grid_num = grid_num
        self.n_pixels_sampled = n_pixels_sampled # select the top pixel
        self.top_n_percent = top_n_percent

        self.W = W
        self.H = H
        self.W_vis = W_vis
        self.H_vis = H_vis

    def __call__(self, render_trainer, T_WC_kf, query_mask, img=None, edge_mask=None):

        ## High res
        _, _, _, view_sem = render_rays.render_images_chunks(
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
            do_color=False,
            do_sem=True,
            n_bins_fine=render_trainer.n_bins_fine_vis,
        )
        view_sem = torch.nn.functional.softmax(view_sem, dim=1)
        sem_kf_hq = view_sem.view(
            render_trainer.H, render_trainer.W, render_trainer.n_classes)

        print(f"Choosing pixels by {self.query_strategy}")
        if self.split_mode == "sp":
            assert img is not None, "colour image should be provided in super-pixel mode"

        # list_queries, n_pixels = list(), 0
        with torch.no_grad():
            query_mask_resize = cv2.resize(query_mask, (self.W_vis, self.H_vis), cv2.INTER_NEAREST)
            # get uncertainty map
            uncertainty_map = getattr(self, f"_{self.query_strategy}")(sem_kf_hq.squeeze(dim=0),
                edge_mask=edge_mask, W_vis=self.W_vis, H_vis=self.H_vis,
            ) # h x w
            # exclude pixels that are already annotated
            uncertainty_map[query_mask_resize.astype(np.bool)] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0

            return self.select_queries(uncertainty_map, self.split_mode, self.pixel_mode, img=img)

    def select_queries(self, uc_map, split_mode, pixel_mode, img=None):
        if  split_mode == "grid":
            x_auto, y_auto, entropy_avg_kf_vis, entropy_kf_vis = self._query_grid(uc_map, pixel_mode, self.grid_num)


        elif split_mode == "sp": # use super-pixel as basic unit to find most-informative pixels
            assert img is not None
            x_auto, y_auto, entropy_avg_kf_vis, entropy_kf_vis = self._query_sp(uc_map, pixel_mode, img)

        elif split_mode == "pixel": # treat each pixel as basic unit to find most-informative pixels
            x_auto, y_auto, entropy_avg_kf_vis, entropy_kf_vis = self._query_pixel(uc_map)


        return [x_auto, y_auto], entropy_avg_kf_vis, entropy_kf_vis




    def _query_grid(self, uc_map, pixel_mode, grid_num):
        uc_map_np = uc_map.cpu().numpy()
        h, w = uc_map.shape
        assert h ==self.H_vis and w == self.W_vis

        w_grid = self.W_vis // grid_num
        h_grid = self.H_vis // grid_num

        new_h, new_w = h_grid*grid_num, w_grid*grid_num
        if new_h !=h or new_w !=w:
        # resize input to multiple times of grids so that the images can be split into full grids
            uc_map_resize = F.interpolate(uc_map[None, None, ...], size=(new_h, new_w), mode='bilinear')[0,0,...] #1x1xhxw

        uc_map_grid = uc_map_resize.reshape((grid_num, h_grid, grid_num, w_grid))
        uc_map_grid_avg = uc_map_grid.mean(dim=(1, -1)) # grid_numxgrid_num

        # find k grids having the maximum sum/mean uc_map
        uc_map_flat = uc_map_grid_avg.flatten()
        k = int(grid_num * grid_num * self.top_n_percent) if self.top_n_percent > 0. else self.n_pixels_sampled
        ind_queries = uc_map_flat.topk(k=k, dim=0, largest=self.query_strategy in ["entropy", "least_confidence"]).indices.cpu().numpy()

        assert k>=self.n_pixels_sampled
        if self.top_n_percent > 0.:
            ind_queries = np.random.choice(ind_queries, self.n_pixels_sampled, False)

        max_grid_ind = np.unravel_index(ind_queries, uc_map_grid_avg.shape)


        if pixel_mode == "max": # find the maximum point within the selected grid
            increments_h = np.arange(h_grid, dtype=np.float32) + max_grid_ind[0]*h_grid
            increments_w = np.arange(w_grid, dtype=np.float32) + max_grid_ind[1]*w_grid

            i, j = np.meshgrid(increments_w, increments_h, indexing='xy') # i-[h_grid, w_grid], x-coordinates; j-[h_grid, w_grid], y-coordinates;
            uc_map_max_block = uc_map_grid[j,i]
            max_pixel_rel_ind = np.unravel_index(torch.argmax(uc_map_max_block, dim=None).item(), (h_grid, w_grid))
            max_pixel_ind = [increments_h[max_pixel_rel_ind[0]], increments_w[max_pixel_rel_ind[1]]] # [max_h_index, max_w_index]

            y_auto, x_auto = max_pixel_ind

        elif pixel_mode == "centriod":
            y_auto = max_grid_ind[0] * h_grid + h_grid//2
            x_auto = max_grid_ind[1] * w_grid + w_grid//2

        elif pixel_mode == "rand":
            y_auto = max_grid_ind[0] * h_grid + np.random.randint(0, h_grid)
            x_auto = max_grid_ind[1] * w_grid + np.random.randint(0, w_grid)


        # we should scale the pixel coordinates to original resolution
        scaling_y = self.H/new_h
        scaling_x = self.W/new_w

        y_auto = int(scaling_y*y_auto)
        x_auto = int(scaling_x*x_auto)

        uc_map_grid_avg_vis = imgviz.depth2rgb(uc_map_grid_avg.cpu().numpy())
        uc_map_vis = imgviz.depth2rgb(uc_map_np)
        return x_auto, y_auto, uc_map_grid_avg_vis, uc_map_vis


    def _query_sp(self, uc_map, pixel_mode, img):
        uc_map_np = uc_map.cpu().numpy()
        h, w = uc_map.shape

        # extract super-pixels from image
        sp_slic_label = segmentation.slic(img, compactness=10, sigma=0, start_label=1,
                                convert2lab=True, enforce_connectivity=True, slic_zero=False)
        sp_slic_rp = measure.regionprops(label_image=sp_slic_label, intensity_image=uc_map_np)
        num_sp = len(sp_slic_rp)

        sort_sp_slic_rp = sorted(sp_slic_rp, key=lambda x: x['mean_intensity']) # ascendin order

        k = int(num_sp * self.top_n_percent) if self.top_n_percent > 0. else self.n_pixels_sampled

        # find the super-pixel with the k highest overall uncertainty
        if self.query_strategy in ["entropy", "least_confidence"]:
            maxUncertainty_region = sort_sp_slic_rp[-k:]  # region with maximum value
        else:
            maxUncertainty_region = sort_sp_slic_rp[:k]  # region with maximum value


        assert k>=self.n_pixels_sampled
        if self.top_n_percent > 0.:
            ind_queries = np.random.choice(k, self.n_pixels_sampled, False).item() # extract the scalar from [1, ] shape array
            ind_queries = ind_queries
        maxUncertainty_region = maxUncertainty_region[ind_queries]

        maxUncertainty_label = maxUncertainty_region.label
        maxUncertainty_value = maxUncertainty_region.max_intensity if self.query_strategy in ["entropy", "least_confidence"] else maxUncertainty_region.min_intensity
        maxUncertainty_coords = maxUncertainty_region.coords # [N,2] array of (row, col)
        maxUncertainty_area = maxUncertainty_region.area

        if pixel_mode == "max":
            maxUncertainty_value_coord = (uc_map_np*(sp_slic_label==maxUncertainty_label)==maxUncertainty_value).nonzero()
            y_auto, x_auto = [maxUncertainty_value_coord[0][0], maxUncertainty_value_coord[1][0]] # (y, x)
        elif pixel_mode == "centriod":
            y_auto, x_auto  = [int(maxUncertainty_region.centroid[0]), int(maxUncertainty_region.centroid[1])] # (y,x)
        elif pixel_mode == "weighted_centroid":
            y_auto, x_auto  = [int(maxUncertainty_region.weighted_centroid[0]), int(maxUncertainty_region.weighted_centroid[1])] # (y, x)
        elif pixel_mode == "rand":
            rand_id = np.random.choice(maxUncertainty_area)
            y_auto, x_auto  = [int(maxUncertainty_coords[rand_id][0]), int(maxUncertainty_coords[rand_id][1])] # (y, x)

        # this is the same to the reduce_factor in config
        scaling_y = self.H/(h)
        scaling_x = self.W/(w)

        y_auto = int(scaling_y*y_auto)
        x_auto = int(scaling_x*x_auto)


        uc_map_sp_avg = np.zeros_like(uc_map_np)
        for i in range(1, sp_slic_label.max()+1): # range of all valid super-pixel labels [1, sp_slic_label.max()]
            uc_map_sp_avg[sp_slic_label==i] = sp_slic_rp[i-1]['mean_intensity']


        uc_map_vis = imgviz.depth2rgb(uc_map_np)
        uc_map_avg_vis = imgviz.depth2rgb(uc_map_sp_avg)
        return x_auto, y_auto, uc_map_avg_vis, uc_map_vis


    def _query_pixel(self, uc_map):
        uc_map_np = uc_map.cpu().numpy()
        h, w = uc_map.shape
        assert h ==self.H_vis and w == self.W_vis
        uc_map = uc_map.flatten()
        k = int(h * w * self.top_n_percent) if self.top_n_percent > 0. else self.n_pixels_sampled

        ind_queries = uc_map.topk(k=k, dim=0, largest=self.query_strategy in ["entropy", "least_confidence"]).indices.cpu().numpy()
        if self.top_n_percent > 0.:
            ind_queries = np.random.choice(ind_queries, self.n_pixels_sampled, False)

        h_arr, w_arr = np.unravel_index(ind_queries, (h, w))

        # this is the same to the reduce_factor in config
        scaling_y = self.H/(h)
        scaling_x = self.W/(w)

        y_auto = scaling_y*h_arr
        x_auto = scaling_x*w_arr
        y_auto = y_auto.astype(int)
        x_auto = x_auto.astype(int)

        uc_map_vis = imgviz.depth2rgb(uc_map_np)
        uc_map_avg_vis = imgviz.depth2rgb(uc_map_np)


        return x_auto, y_auto, uc_map_avg_vis, uc_map_vis

    @staticmethod
    def _entropy(prob, edge_mask=None, W_vis=0, H_vis=0):
        entropy = (-prob * torch.log2(prob+1e-12)).sum(dim=-1)
        entropy[entropy.isnan()] = 0

        ## mask edges
        if edge_mask is not None:

            entropy_np = entropy.cpu().numpy()

            entropy_np_norm = (entropy_np - np.min(entropy_np))/np.ptp(entropy_np)
            entropy_mask = np.greater(entropy_np_norm, np.full(entropy_np_norm.shape, 0.7)) # 0.75
            entropy_np[entropy_mask] = 0.

            entropy = torch.from_numpy(entropy_np)

        entropy = torchvision.transforms.functional.gaussian_blur(
            entropy.unsqueeze(0), (41, 41)).squeeze(0)

        entropy_np = entropy.cpu().detach().numpy()

        ## Resize
        entropy_np = imgviz.resize(
            entropy_np, width=W_vis, height=H_vis,
            interpolation="nearest")
        entropy = torch.from_numpy(entropy_np)


        entropy_np = entropy.cpu().detach().numpy()
        entropy_np = imgviz.depth2rgb(entropy_np)
        entropy_np = cv2.cvtColor(
            entropy_np, cv2.COLOR_BGR2RGB)
        cv2.imshow("entropy_np", entropy_np)

        return entropy

    @staticmethod
    def _least_confidence(prob):
        return 1.0 - prob.max(dim=-1)[0]

    @staticmethod
    def _margin_sampling(prob):
        top2 = prob.topk(k=2, dim=-1).values  # h x w x k
        return (top2[:, :, 0] - top2[:, :, 1]).abs()

    @staticmethod
    def _random(prob):
        h, w, c = prob.shape
        return torch.rand((h, w)) # generate uniform variables
