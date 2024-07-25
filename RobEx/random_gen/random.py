import torch
import numpy as np
import matplotlib.pyplot as plt


def sample_pixels(n_rays,
                  batch_size,
                  h,
                  w,
                  device):
    total_rays = n_rays * batch_size
    indices_h = torch.randint(0, h, (total_rays,), device=device)
    indices_w = torch.randint(0, w, (total_rays,), device=device)

    indices_b = torch.arange(batch_size, device=device)
    indices_b = indices_b.repeat_interleave(n_rays)

    return indices_b, indices_h, indices_w


def active_sample(loss_approx,
                  frame_sum, W, H,
                  batch_size,
                  n_rays,
                  increments_single):
    factor = loss_approx.shape[1]
    w_block = W // factor
    h_block = H // factor

    frame_dist = frame_sum / frame_sum.sum()
    total_rays = n_rays * batch_size
    rays_per_frame = frame_dist * total_rays

    loss_dist = loss_approx / frame_sum[:, None, None]
    pixels_per_block = torch.floor(
        loss_dist * rays_per_frame[:, None, None]).long()
    pixels_per_batch = pixels_per_block.sum(dim=(1, 2))
    total_pixels = pixels_per_batch.sum().item()

    increments = increments_single.repeat(batch_size, 1)
    pixels_per_block = pixels_per_block.view(-1)
    increments_repeat = torch.repeat_interleave(
        increments, pixels_per_block, dim=0
    )

    indices_w = torch.randint(
        0, w_block, (total_pixels,), device=loss_approx.device)
    indices_h = torch.randint(
        0, h_block, (total_pixels,), device=loss_approx.device)

    indices_h = indices_h + increments_repeat[:, 0]
    indices_w = indices_w + increments_repeat[:, 1]
    indices_b = torch.arange(batch_size,
                             device=loss_approx.device)
    indices_b = indices_b.repeat_interleave(pixels_per_batch)

    return indices_b, indices_h, indices_w, None
