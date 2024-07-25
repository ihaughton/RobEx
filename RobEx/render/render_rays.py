import torch
import numpy as np

from RobEx.mapping import embedding


def ray_dirs_C(B, H, W, fx, fy, cx, cy, device, depth_type='z'):
    c, r = torch.meshgrid(torch.arange(W, device=device),
                          torch.arange(H, device=device))
    c, r = c.t().float(), r.t().float()
    size = [B, H, W]

    C = torch.empty(size, device=device)
    R = torch.empty(size, device=device)
    C[:, :, :] = c[None, :, :]
    R[:, :, :] = r[None, :, :]

    z = torch.ones(size, device=device)
    x = (C - cx) / fx
    y = (R - cy) / fy

    dirs = torch.stack((x, y, z), dim=3)
    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=3)
        dirs = dirs * (1. / norm)[:, :, :, None]

    return dirs


def origin_dirs_W(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]
    dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1)
    origins = T_WC[:, :3, -1]

    return origins, dirs_W


def stratified_bins(min_depth,
                    max_depth,
                    n_bins,
                    n_rays,
                    device):

    bin_limits = torch.linspace(
        min_depth,
        max_depth,
        n_bins + 1,
        device=device,
    )
    lower_limits = bin_limits[:-1]
    bin_length = (max_depth - min_depth) / (n_bins)
    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    z_vals = lower_limits[None, :] + increments

    return z_vals


def occupancy_activation(alpha, distances):
    occ = 1.0 - torch.exp(-alpha * distances)

    return occ


def alpha_to_occupancy(depths, dirs, alpha, add_last=False):
    interval_distances = depths[..., 1:] - depths[..., :-1]

    if add_last:
        last_distance = torch.empty(
            (depths.shape[0], 1),
            device=depths.device,
            dtype=depths.dtype).fill_(0.1)
        interval_distances = torch.cat(
            [interval_distances, last_distance], dim=-1)

    dirs_norm = torch.norm(dirs, dim=-1)
    interval_distances = interval_distances * dirs_norm[:, None]
    occ = occupancy_activation(alpha, interval_distances)

    return occ


def occupancy_to_termination(occupancy):

    first = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
    free_probs = (1. - occupancy + 1e-10)[:, :-1]
    free_probs = torch.cat([first, free_probs], dim=-1)
    term_probs = occupancy * torch.cumprod(free_probs, dim=-1)

    return term_probs


def render(termination, vals, dim=-1):
    weighted_vals = termination * vals
    render = weighted_vals.sum(dim=dim)

    return render


def render_loss(render, gt_depth, loss="L1", normalise=False):
    residual = render - gt_depth
    if loss == "L2":
        loss_mat = residual ** 2
    elif loss == "L1":
        loss_mat = torch.abs(residual)

    if normalise:
        loss_mat = loss_mat / gt_depth

    return loss_mat


def reduce_loss(loss_mat, var=None, avg=True, loss_type="L1"):
    if var is not None:
        eps = 1e-4
        if loss_type == "L2":
            information = 1.0 / (var + eps)
        elif loss_type == "L1":
            information = 1.0 / (torch.sqrt(var) + eps)

        loss_weighted = loss_mat * information
    else:
        loss_weighted = loss_mat

    if avg:
        loss = torch.mean(loss_weighted, dim=0).sum()
    else:
        loss = loss_weighted

    return loss


def mix_zvals_alpha(alpha,
                    alpha_fine,
                    z_vals,
                    z_vals_fine,
                    color=None,
                    color_fine=None,
                    sem=None,
                    sem_fine=None,
                    feat=None,
                    feat_fine=None):

    z_vals_mix, arg_inds = torch.sort(
        torch.cat((z_vals, z_vals_fine), dim=-1), dim=-1)
    alpha_mix = torch.cat((alpha, alpha_fine), dim=-1)

    inds_1 = torch.arange(
        arg_inds.shape[0]).repeat_interleave(arg_inds.shape[1])
    inds_2 = arg_inds.view(-1)

    alpha_mix = alpha_mix[inds_1, inds_2].view(z_vals_mix.shape)

    color_mix = None
    if color is not None:
        color_mix = torch.cat((color, color_fine), dim=-2)
        color_shape = color_mix.shape
        color_mix = color_mix[inds_1, inds_2, :].view(color_shape)

    sem_mix = None
    if sem is not None:
        sem_mix = torch.cat((sem, sem_fine), dim=-2)
        sem_shape = sem_mix.shape
        sem_mix = sem_mix[inds_1, inds_2, :].view(sem_shape)

    feat_mix = None
    if feat is not None:
        feat_mix = torch.cat((feat, feat_fine), dim=-2)
        feat_shape = feat_mix.shape
        feat_mix = feat_mix[inds_1, inds_2, :].view(feat_shape)

    return z_vals_mix, alpha_mix, color_mix, sem_mix #, feat_mix


def sample_z_vals_fine(termination,
                       min_depth,
                       max_depth,
                       n_bins_fine
                       ):
    bin_limits = torch.linspace(
        min_depth,
        max_depth,
        termination.shape[-1] + 2,
        device=termination.device,
    )

    bin_limits = bin_limits.expand(
        [termination.shape[0], bin_limits.shape[0]])

    mid_limits = 0.5 * (bin_limits[..., 1:] + bin_limits[..., :-1])

    termination = termination.detach()
    z_vals_fine = sample_pdf(mid_limits, termination, n_bins_fine)

    return z_vals_fine


def weight_from_termination(termination):
    termination_det = termination.detach()

    termination_pad = torch.cat([
        termination_det[..., :1],
        termination_det,
        termination_det[..., -1:],
    ],
        axis=-1)
    termination_max = torch.max(
        termination_pad[..., :-1], termination_pad[..., 1:])
    termination_blur = 0.5 * \
        (termination_max[..., :-1] + termination_max[..., 1:])
    weights = termination_blur + 0.01

    return weights


def sample_fine(weights,
                bin_limits,
                min_depth,
                max_depth,
                n_bins,
                n_bins_fine,
                origins,
                dirs_W,
                n_embed_funcs,
                occ_map,
                B_layer=None,
                noise_std=None,
                do_color=False,
                do_sem=False,
                do_features=False
                ):

    z_vals_fine = sample_pdf(bin_limits, weights, n_bins_fine)

    pc_fine = origins[:, None, :] + \
        (dirs_W[:, None, :] * z_vals_fine[:, :, None])
    points_embedding_fine = embedding.positional_encoding(
        pc_fine, B_layer, num_encoding_functions=n_embed_funcs)

    feat_fine = None
    if do_features:
        alpha_fine, color_fine, sem_fine, feat_fine = occ_map(
            points_embedding_fine, noise_std=noise_std,
            do_color=do_color, do_sem=do_sem, return_features=True
        )
    else:
        alpha_fine, color_fine, sem_fine = occ_map(
            points_embedding_fine, noise_std=noise_std,
            do_color=do_color, do_sem=do_sem, return_features=False
        )
    alpha_fine = alpha_fine.squeeze(dim=-1)

    if color_fine is not None:
        color_fine = color_fine.squeeze(dim=-1)
    if sem_fine is not None:
        sem_fine = sem_fine.squeeze(dim=-1)
    if feat_fine is not None:
        feat_fine = feat_fine.squeeze(dim=-1)

    return z_vals_fine, alpha_fine, color_fine, sem_fine #, feat_fine


def render_images_chunks(T_WC,
                         min_depth, max_depth,
                         n_embed_funcs, n_bins,
                         occ_map,
                         B_layer=None,
                         H=None, W=None, fx=None, fy=None, cx=None, cy=None,
                         grad=False, dirs_C=None,
                         do_fine=False,
                         do_var=True,
                         do_color=False,
                         do_sem=False,
                         n_bins_fine=None,
                         noise_std=None,
                         chunk_size=10000):

    if dirs_C is None:
        B_cam = T_WC.shape[0]
        dirs_C = ray_dirs_C(
            B_cam, H, W, fx, fy, cx, cy, T_WC.device, depth_type='z')
        dirs_C = dirs_C.view(B_cam, -1, 3)

    n_pts = dirs_C.shape[1]
    n_chunks = int(np.ceil(n_pts / chunk_size))
    depths = []
    vars = []
    cols = []
    sems = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = dirs_C[:, start:end, :]
        depth, var, col, sem = render_images(
            T_WC,
            min_depth, max_depth,
            n_embed_funcs, n_bins,
            occ_map,
            B_layer=B_layer,
            H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
            grad=grad,
            dirs_C=chunk,
            do_fine=do_fine,
            do_var=do_var,
            do_color=do_color,
            do_sem=do_sem,
            n_bins_fine=n_bins_fine,
            noise_std=noise_std
        )
        depths.append(depth.detach())
        if do_var:
            vars.append(var.detach())
        if do_sem:
            sems.append(sem.detach())
        if do_color:
            cols.append(col.detach())

    depths = torch.cat(depths, dim=0)
    if do_var:
        vars = torch.cat(vars, dim=0)
    if do_sem:
        sems = torch.cat(sems, dim=0)
    if do_color:
        cols = torch.cat(cols, dim=0)

    return depths, vars, cols, sems


def render_normals(T_WC,
                   render_depth,
                   occ_map,
                   dirs_C,
                   n_embed_funcs,
                   origins_dirs,
                   B_layer=None,
                   noise_std=None,
                   do_mip=False,
                   radius=None
                   ):
    origins, dirs_W = origins_dirs
    origins = origins.view(-1, 3)
    dirs_W = dirs_W.view(-1, 3)

    if do_mip:
        z_vals_limits = torch.stack((render_depth - 0.005,
                                     render_depth + 0.005,), dim=1)
        pc, cov_diag = mip_mean_cov(
            z_vals_limits, radius, dirs_W, origins)
        pc.requires_grad_(True)
        points_embedding = mip_embed(pc, cov_diag, n_embed_funcs)
    else:
        pc = origins + (dirs_W * (render_depth[:, None]))
        pc.requires_grad_(True)
        points_embedding = embedding.positional_encoding(
            pc, B_layer, num_encoding_functions=n_embed_funcs)

    alpha, _, _ = occ_map(
        points_embedding, noise_std=noise_std,
        do_color=False, do_sem=False
    )

    d_points = torch.ones_like(
        alpha, requires_grad=False, device=alpha.device)
    points_grad = torch.autograd.grad(
        outputs=alpha,
        inputs=pc,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if do_mip:
        points_grad = points_grad.squeeze(1)

    surface_normals_W = -points_grad / \
        (points_grad.norm(dim=1, keepdim=True) + 1e-4)
    R_CW = T_WC[:, :3, :3].inverse()
    surface_normals_C = (R_CW * surface_normals_W[..., None, :]).sum(dim=-1)

    L = dirs_C.squeeze(0)
    L = -L / L.norm(dim=1, keepdim=True)
    diffuse = (L * surface_normals_C).sum(1)

    return surface_normals_W, diffuse


def render_images(T_WC,
                  min_depth, max_depth,
                  n_embed_funcs, n_bins,
                  occ_map,
                  B_layer=None,
                  H=None, W=None, fx=None, fy=None, cx=None, cy=None,
                  grad=False, dirs_C=None,
                  do_fine=False,
                  do_var=True,
                  do_color=False,
                  do_sem=False,
                  n_bins_fine=None,
                  noise_std=None,
                  z_vals_limits=None,
                  normalise_weights=False,
                  do_sem_activation=False,
                  do_hierarchical=False,
                  do_mip=False,
                  render_coarse=False,
                  radius=None,
                  origins_dirs=None):
    B_cam = T_WC.shape[0]
    with torch.set_grad_enabled(grad):
        if dirs_C is None and origins_dirs is None:
            dirs_C = ray_dirs_C(
                B_cam, H, W, fx, fy, cx, cy, T_WC.device, depth_type='z')
            dirs_C = dirs_C.view(B_cam, -1, 3)

        # rays in world coordinate
        if origins_dirs is None:
            origins, dirs_W = origin_dirs_W(T_WC, dirs_C)
        else:
            origins, dirs_W = origins_dirs

        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        n_rays = dirs_W.shape[0]

        if z_vals_limits is None:
            # back project samples
            z_vals_limits = stratified_bins(
                min_depth, max_depth,
                n_bins, n_rays, T_WC.device)
        z_vals = 0.5 * (z_vals_limits[..., 1:] + z_vals_limits[..., :-1])

        if do_mip:
            mean, cov_diag = mip_mean_cov(
                z_vals_limits, radius, dirs_W, origins)
            points_embedding = mip_embed(mean, cov_diag, n_embed_funcs)

        else:
            pc = origins[:, None, :] + \
                (dirs_W[:, None, :] * z_vals[:, :, None])

            points_embedding = embedding.positional_encoding(
                pc, B_layer, num_encoding_functions=n_embed_funcs)

        alpha, color, sem = occ_map(
            points_embedding, noise_std=noise_std,
            do_color=do_color, do_sem=do_sem
        )
        is_color = color is not None

        alpha = alpha.squeeze(dim=-1)
        if is_color:
            color = color.squeeze(dim=-1)
        if do_sem:
            sem = sem.squeeze(dim=-1)

        occupancy = alpha_to_occupancy(
            z_vals_limits, dirs_W, alpha
        )
        termination = occupancy_to_termination(occupancy)
        if normalise_weights:
            termination = termination + 0.0001
            termination = termination / termination.sum(1, keepdim=True)

        if render_coarse or do_fine is False:
            render_depth_c = render(termination, z_vals)

            var_c = None
            if do_var:
                diff_sq = (z_vals - render_depth_c[:, None]) ** 2
                var_c = render(termination, diff_sq)

            render_color_c = None
            if is_color:
                render_color_c = render(
                    termination[..., None], color, dim=-2)
            render_sem_c = None
            if do_sem:
                render_sem_c = render(
                    termination[..., None], sem, dim=-2)
                if do_sem_activation:
                    if do_hierarchical:
                        render_sem_c = torch.sigmoid(render_sem_c)
                    else:
                        render_sem_c = torch.nn.functional.softmax(
                            render_sem_c, dim=1)

            if do_fine is False:
                return render_depth_c, var_c, render_color_c, render_sem_c

        if do_fine:
            weights = weight_from_termination(termination)
            if do_mip:
                z_vals_fine = sample_z_vals_fine(
                    weights,
                    min_depth,
                    max_depth,
                    32
                )
                z_vals_mix, _ = torch.sort(z_vals_fine, dim=-1)

                z_mids_mix = 0.5 * (z_vals_mix[..., :-1] + z_vals_mix[..., 1:])

                mean_mix, cov_diag_mix = mip_mean_cov(
                    z_vals_mix, radius, dirs_W, origins)
                embed_mix = mip_embed(mean_mix, cov_diag_mix, n_embed_funcs)

                alpha_mix, color_mix, sem_mix = occ_map(
                    embed_mix, noise_std=noise_std,
                    do_color=do_color, do_sem=do_sem
                )
                is_color = color is not None

                alpha_mix = alpha_mix.squeeze(dim=-1)
                if is_color:
                    color_mix = color_mix.squeeze(dim=-1)
                if do_sem:
                    sem_mix = sem_mix.squeeze(dim=-1)

                occupancy_mix = alpha_to_occupancy(
                    z_vals_mix, dirs_W, alpha_mix)
                termination_mix = occupancy_to_termination(occupancy_mix)
                z_vals_mix = z_mids_mix

            else:
                z_vals_fine, alpha_fine, color_fine, sem_fine = sample_fine(
                    weights,
                    z_vals_limits,
                    min_depth,
                    max_depth,
                    n_bins,
                    n_bins_fine,
                    origins,
                    dirs_W,
                    n_embed_funcs,
                    occ_map,
                    B_layer,
                    noise_std=noise_std,
                    do_color=do_color,
                    do_sem=do_sem
                )

                z_vals_mix, alpha_mix, color_mix, sem_mix = mix_zvals_alpha(
                    alpha,
                    alpha_fine,
                    z_vals,
                    z_vals_fine,
                    color,
                    color_fine,
                    sem,
                    sem_fine
                )

                occupancy_mix = alpha_to_occupancy(
                    z_vals_mix, dirs_W, alpha_mix, add_last=True
                )
                termination_mix = occupancy_to_termination(occupancy_mix)

        render_depth = render(termination_mix, z_vals_mix)

        var = None
        if do_var:
            diff_sq = (z_vals_mix - render_depth[:, None]) ** 2
            var = render(termination_mix, diff_sq)

        render_color = None
        if is_color:
            render_color = render(
                termination_mix[..., None], color_mix, dim=-2)
        render_sem = None
        if do_sem:
            render_sem = render(
                termination_mix[..., None], sem_mix, dim=-2)
            if do_sem_activation:
                if do_hierarchical:
                    render_sem = torch.sigmoid(render_sem)
                else:
                    render_sem = torch.nn.functional.softmax(render_sem, dim=1)

        if render_coarse is False:
            return render_depth, var, render_color, render_sem
        else:
            return (
                [render_depth, render_depth_c],
                [var, var_c],
                [render_color, render_color_c],
                [render_sem, render_sem_c]
            )


def render_features(T_WC,
                  min_depth, max_depth,
                  n_embed_funcs, n_bins, n_bins_fine,
                  occ_map,
                  do_fine=True,
                  B_layer=None,
                  H=None, W=None, fx=None, fy=None, cx=None, cy=None,
                  grad=False, dirs_C=None,
                  noise_std=None):

    B_cam = T_WC.shape[0]
    with torch.set_grad_enabled(grad):
        if dirs_C is None:
            dirs_C = ray_dirs_C(
                B_cam, H, W, fx, fy, cx, cy, T_WC.device, depth_type='z')
            dirs_C = dirs_C.view(B_cam, -1, 3)

        # rays in world coordinate
        origins, dirs_W = origin_dirs_W(T_WC, dirs_C)

        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        n_rays = dirs_W.shape[0]

        # back project samples
        z_vals = stratified_bins(
            min_depth, max_depth,
            n_bins, n_rays, T_WC.device)

        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])

        points_embedding = embedding.positional_encoding(
            pc, B_layer, num_encoding_functions=n_embed_funcs)

        alpha, color, sem, feat = occ_map(
            points_embedding, noise_std=noise_std,
            do_color=True, do_sem=True, return_features=True
        )

        alpha = alpha.squeeze(dim=-1)
        color = color.squeeze(dim=-1)
        sem = sem.squeeze(dim=-1)
        feat = feat.squeeze(-1)

        occupancy = alpha_to_occupancy(z_vals, dirs_W, alpha)
        termination = occupancy_to_termination(occupancy)

        if do_fine:
            z_vals_fine, alpha_fine, color_fine, sem_fine, feat_fine = sample_fine(
                termination,
                min_depth,
                max_depth,
                n_bins,
                n_bins_fine,
                origins,
                dirs_W,
                n_embed_funcs,
                occ_map,
                B_layer,
                noise_std=noise_std,
                do_color=True,
                do_sem=True,
                do_features=True
            )
            z_vals_mix, alpha_mix, color_mix, sem_mix, feat_mix = mix_zvals_alpha(
                alpha,
                alpha_fine,
                z_vals,
                z_vals_fine,
                color,
                color_fine,
                sem,
                sem_fine,
                feat,
                feat_fine
            )

            occupancy_mix = alpha_to_occupancy(z_vals_mix, dirs_W, alpha_mix)
            termination_mix = occupancy_to_termination(occupancy_mix)

            features = render(termination_mix[..., None], feat_mix, dim=-2)
            #features = feat_mix.sum(dim=-2)
        else:
            features = feat.sum(dim=-2)
        return features

def entropy(dist):
    return -(dist * torch.log(dist)).sum(2)


def map_color(sem, colormap):
    return (colormap[..., :, :] * sem[..., None]).sum(-2)


def sample_pdf(bin_limits, weights, num_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )

    u = torch.rand(
        list(cdf.shape[:-1]) + [num_samples],
        dtype=weights.dtype,
        device=weights.device,
    )

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bin_limits.unsqueeze(
        1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def loss_approx(full_loss, binary_masks, W, H, factor=8):
    w_block = W // factor
    h_block = H // factor
    loss_approx = full_loss.view(-1, factor, h_block, factor, w_block)
    loss_approx = loss_approx.sum(dim=(2, 4))
    actives = binary_masks.view(-1, factor, h_block, factor, w_block)
    actives = actives.sum(dim=(2, 4))
    actives[actives == 0] = 1.0
    loss_approx = loss_approx / actives

    return loss_approx


def mip_mean_cov(z_vals, radius, dirs_W, origins):
    t0 = z_vals[..., :-1]
    t1 = z_vals[..., 1:]
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                      (3 * mu**2 + hw**2)**2)
    r_var = radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                         (hw**4) / (3 * mu**2 + hw**2))

    mean = dirs_W[..., None, :] * t_mean[..., None]
    mean = mean + origins[..., None, :]

    d_mag_sq = torch.max(torch.tensor(
        [1e-10], device=z_vals.device), torch.sum(dirs_W**2, axis=-1, keepdims=True))
    d_outer_diag = dirs_W**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq

    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag

    return mean, cov_diag


def mip_embed(mean, cov_diag, n_embed_funcs, scale=10):
    scales = torch.tensor(
        [2**i for i in range(0, n_embed_funcs)], device=mean.device)
    y = mean[..., None, :] * scales[:, None]
    y_var = cov_diag[..., None, :] * scales[:, None]**2

    shape = list(mean.shape[:-1]) + [-1]
    y = y.reshape(shape)
    y_var = y_var.reshape(shape)

    y = torch.cat([y, y + 0.5 * np.pi], axis=-1)
    y_var = torch.cat([y_var] * 2, axis=-1)

    scale_var = scale**2
    embed = torch.exp(-0.5 * y_var / scale_var) * torch.sin(y / scale)
    return embed


if __name__ == "__main__":

    bins_limits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    weights = torch.tensor([[0.2, 0.3, 0.1, 0.1, 0.5],
                            [0.2, 0.3, 0.1, 0.1, 0.5]])
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    n_samples = 20

    sample_pdf(bins_limits, weights, n_samples)

    import ipdb
    ipdb.set_trace()
