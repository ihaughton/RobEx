import torch
import trimesh
import numpy as np
from RobEx.mapping import embedding
import skimage.measure

def make_3D_grid(occ_range, dim, device, transform=None, scale=None):
    t = torch.linspace(occ_range[0], occ_range[1], steps=dim, device=device)
    grid = torch.meshgrid(t, t, t)
    grid_3d = torch.cat(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), dim=3
    )

    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)
        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d


def chunks(pc,
           chunk_size,
           fc_occ_map,
           n_embed_funcs,
           B_layer,
           do_alpha=True,
           do_color=False,
           do_sem=False,
           do_hierarchical=False,
           to_cpu=False):
    n_pts = pc.shape[0]
    n_chunks = int(np.ceil(n_pts / chunk_size))
    cols = []
    alphas = []
    sems = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = pc[start:end, :]
        points_embedding = embedding.positional_encoding(
            chunk, B_layer, num_encoding_functions=n_embed_funcs
        )
        alpha, col, sem = fc_occ_map(
            points_embedding,
            do_alpha=do_alpha, do_color=do_color, do_sem=do_sem)

        if do_alpha:
            alpha = alpha.squeeze(dim=-1)
            if to_cpu:
                alpha = alpha.cpu()
            alphas.append(alpha)

        if do_color:
            col = col.squeeze(dim=-1)
            if to_cpu:
                col = col.cpu()
            cols.append(col.detach())

        if do_sem:
            sem = sem.squeeze(dim=-1)
            sem = sem.squeeze(dim=0)
            if do_hierarchical:
                sem = torch.sigmoid(sem)
            else:
                sem = torch.nn.functional.softmax(sem, dim=1)

            if to_cpu:
                sem = sem.cpu()
            sems.append(sem)

    if do_alpha:
        alphas = torch.cat(alphas, dim=-1)
    if do_color:
        cols = torch.cat(cols, dim=-2)
    if do_sem:
        sems = torch.cat(sems, dim=-2)

    return alphas, cols, sems


def marching_cubes(occupancy, level):
    vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(
        occupancy, level=level, gradient_direction='ascent', method="lewiner")

    dim = occupancy.shape[0]
    vertices = vertices / (dim - 1)
    mesh = trimesh.Trimesh(vertices=vertices,
                           vertex_normals=vertex_normals,
                           faces=faces)

    return mesh
