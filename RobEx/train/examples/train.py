#!/usr/bin/env python
import torch

# from torch.utils.data import DataLoader
import cv2
import json
import os
from datetime import datetime
import argparse
import numpy as np
import random

from RobEx import visualisation
from RobEx.train import trainer
from RobEx.mapping import occupancy

# torch.manual_seed(0)
# random.seed(0)

def train(
    save_ims,
    show_freq,
    device,
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
):
    # init trainer-------------------------------------------------------------
    show_ims = show_freq is not None
    render_trainer = trainer.RenderTrainer(
        device,
        config_file,
        load_path=load_path,
        load_epoch=load_epoch,
        do_mesh=show_mesh,
        incremental=incremental,
        do_track=do_track,
        do_color=do_color,
    )

    # saving init--------------------------------------------------------------
    save_chckpts = save_path is not None
    if save_chckpts:
        with open(save_path + "/" + config_file, "w") as outfile:
            json.dump(render_trainer.config, outfile, indent=4)
        # writer = SummaryWriter(save_path)
        # dummy_input = torch.rand(
        #     (1, render_trainer.embedding_size), device=device)
        # writer.add_graph(render_trainer.fc_occ_map, (dummy_input,))

    # main  loop---------------------------------------------------------------
    save_count = 0
    # size_dataset = len(render_trainer.scene_dataset)
    for t in range(render_trainer.t_init, render_trainer.N_epochs):
        # if t == 4000:
        #     render_trainer.grid_dim = 512
        #     import ipdb
        #     ipdb.set_trace()

        render_trainer.frames_since_add += 1
        # get/add data---------------------------------------------------------
        finish_optim = (
            render_trainer.frames_since_add == render_trainer.optim_frames
        )

        if finish_optim:
            render_trainer.has_initialised = True
            if incremental is False:
                render_trainer.noise_std = render_trainer.noise_frame

        if incremental and (finish_optim or t == 0):
            if t == 0:
                new_frame = True

            else:
                if do_track:
                    T_WC = render_trainer.frames.T_WC_track[-1]
                else:
                    T_WC = render_trainer.frames.T_WC_batch[-1]

                T_WC = T_WC.unsqueeze(0)
                new_frame = render_trainer.check_keyframe(T_WC)

            if new_frame:
                frame_data = render_trainer.get_data([render_trainer.frame_id])
                render_trainer.add_frame(frame_data)
                render_trainer.frame_id += 1

                # if render_trainer.frame_id == 311:
                #     np.save("tree_save/bounds_T.npy",
                #             render_trainer.transform)
                #     np.save("tree_save/bounds_extents.npy",
                #             render_trainer.extents)

                if do_track:
                    if t == 0:
                        if render_trainer.gt_traj:
                            T_WC_track_init = render_trainer.frames.T_WC_batch
                        else:
                            T_WC_track_init = torch.eye(
                                4, device=render_trainer.device
                            ).unsqueeze(0)

                        render_trainer.init_pose_vars(T_WC_track_init)
                        render_trainer.init_trajectory(render_trainer.T_WC)

                    else:
                        latest_depth = render_trainer.frames.depth_batch[-1].unsqueeze(
                            0)

                        latest_im = None
                        if render_trainer.do_color and render_trainer.track_color:
                            latest_im = render_trainer.frames.im_batch[-1].unsqueeze(
                                0)
                        track_time = render_trainer.track(
                            latest_depth,
                            render_trainer.do_information,
                            render_trainer.do_fine,
                            render_trainer.frames.depth_batch_np[-1],
                            latest_im,
                            do_vis=True
                        )
                        print("track_time: ", track_time)
                        render_trainer.add_track_pose(render_trainer.T_WC)
                    # SAVE---------------------------------------------------------------
                    # save_pad = f'{(render_trainer.frame_id-1):03}'
                    # chechpoint_file = "tree_save/epoch_" + save_pad + ".pth"
                    # B_layer_dict = render_trainer.B_layer.state_dict()
                    # torch.save(
                    #     {
                    #         "epoch": render_trainer.frame_id - 1,
                    #         "model_state_dict": render_trainer.fc_occ_map.state_dict(),
                    #         "B_layer_state_dict": B_layer_dict,
                    #     },
                    #     chechpoint_file,
                    # )
                    # np.save("tree_save/T_" + save_pad + ".npy",
                    #         render_trainer.T_WC.cpu().numpy())
                    # SAVE---------------------------------------------------------------

                print("frame______________________", render_trainer.frame_id)
                # if(render_trainer.frame_id == size_dataset):
                    # if save_chckpts:
                    #     chechpoint_file = save_path + \
                    #         "/checkpoints" "/epoch_" + str(t) + ".pth"

                    #     B_layer_dict = None
                    #     if render_trainer.B_layer is not None:
                    #         B_layer_dict = render_trainer.B_layer.state_dict()
                    #     torch.save(
                    #         {
                    #             "epoch": t,
                    #             "model_state_dict": render_trainer.fc_occ_map.state_dict(),
                    #             "B_layer_state_dict": B_layer_dict,
                    #             "optimizer_state_dict": render_trainer.optimiser.state_dict(),
                    #             "loss": loss.item(),
                    #         },
                    #         chechpoint_file,
                    #     )

                    # if render_trainer.dataset_format == "TUM":
                    #     trainer.save_trajectory(
                    #         render_trainer.poses, "./", "traj_tum.txt",
                    #         format="TUM",
                    #         timestamps=render_trainer.scene_dataset.timestamps)
                    # elif render_trainer.dataset_format == "replica":
                    #     trainer.save_trajectory(
                    #         render_trainer.poses_gt, "./", "traj_gt.txt")

                if render_trainer.do_tsdf:
                    if do_track:
                        T_WC = render_trainer.frames.T_WC_track[-1]
                    else:
                        T_WC = render_trainer.frames.T_WC_batch[-1]
                    render_trainer.integrate_tsdf(
                        render_trainer.frames.im_batch_np[-1],
                        render_trainer.frames.depth_batch_np[-1],
                        T_WC.cpu().numpy(),
                        render_trainer.intrinsic_open3d)

                if t == 0:
                    render_trainer.last_is_keyframe = True
                else:
                    render_trainer.last_is_keyframe = False

            # if (new_frame is False and render_trainer.last_is_keyframe) or t == 0:
            #     col_save = render_trainer.frames.im_batch_np[-1]
            #     col_save = cv2.cvtColor(col_save, cv2.COLOR_RGB2BGR)
            #     save_pad = f'{(render_trainer.frame_id-1):03}'
            #     file_name = 'tree_save/keyframe_' + save_pad + '.png'
            #     cv2.imwrite(file_name, col_save)

        render_trainer.step_pyramid(t)

        # window optimisation step---------------------------------------------
        depth_batch_raw, T_WC_batch_gt, im_RGB = render_trainer.get_batch()
        depth_batch_raw_np = render_trainer.frames.depth_batch_np

        # pyramid at init.
        if t == 0:
            pyramid_batch_np = None
            pyramid_batch = None
            for idx in range(render_trainer.batch_size):
                pyramid_np = trainer.get_pyramid(
                    depth_batch_raw_np[idx],
                    render_trainer.n_levels,
                    render_trainer.kernel_init,
                )[None, ...]
                pyramid = torch.from_numpy(
                    pyramid_np).float().to(render_trainer.device)
                pyramid_batch_np = render_trainer.expand_data(
                    pyramid_batch_np, pyramid_np
                )
                pyramid_batch = render_trainer.expand_data(
                    pyramid_batch, pyramid)

        if render_trainer.has_initialised is False:
            depth_batch = pyramid_batch[:, render_trainer.pyramid_level, :, :]
            depth_batch_np = pyramid_batch_np[:,
                                              render_trainer.pyramid_level, :, :]
        else:
            depth_batch = depth_batch_raw
            depth_batch_np = depth_batch_raw_np
            pyramid_batch_np = None
            pyramid_batch = None

        if incremental and do_track:
            T_WC_batch = render_trainer.frames.T_WC_track
        else:
            T_WC_batch = T_WC_batch_gt

        im_batch = None
        if render_trainer.do_color:
            im_batch = im_RGB

        loss, step_time = render_trainer.step(
            depth_batch,
            T_WC_batch,
            render_trainer.do_information,
            render_trainer.do_fine,
            do_active=do_active,
            im_batch=im_batch,
            color_scaling=render_trainer.color_scaling,
        )
        print("step_time: ", step_time)

        # saving----------------------------------------------------------------
        # if save_chckpts:
        #     writer.add_scalar("training loss", loss.item(), t)

        if save_chckpts and (t % checkpoint_freq == 0):
            chechpoint_file = save_path + \
                "/checkpoints" "/epoch_" + str(t) + ".pth"

            B_layer_dict = None
            if render_trainer.B_layer is not None:
                B_layer_dict = render_trainer.B_layer.state_dict()
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": render_trainer.fc_occ_map.state_dict(),
                    "B_layer_state_dict": B_layer_dict,
                    "optimizer_state_dict": render_trainer.optimiser.state_dict(),
                    "loss": loss.item(),
                },
                chechpoint_file,
            )

        # visualisation----------------------------------------------------------
        if show_ims and (t % show_freq == 0):
            view_depths, view_vars, view_cols, _ = trainer.render_vis(
                render_trainer.batch_size,
                render_trainer,
                T_WC_batch,
                render_trainer.do_fine,
                do_var=True,
                do_color=render_trainer.do_color,
            )
            if render_trainer.has_initialised is False:
                render_trainer.gt_depth_vis = None
                render_trainer.gt_im_vis = None
            render_trainer.update_vis_vars(
                depth_batch_np, render_trainer.frames.im_batch_np
            )

            viz = trainer.image_vis(
                render_trainer.batch_size,
                render_trainer.gt_depth_vis,
                view_depths,
                view_vars,
                render_trainer.gt_im_vis,
                view_cols,
            )
            if save_ims:
                file_name = "res/im_" + str(t) + ".jpg"
                cv2.imwrite(file_name, viz)

            if (show_pc or show_mesh) and (t % scene_draw_freq == 0):
                if t < 100:
                    draw_mesh = False
                    level = 0.
                else:
                    draw_mesh = show_mesh
                    level = 0.45
                T_WC_batch_np = T_WC_batch.detach().cpu().numpy()

                scene = render_trainer.draw_3D(
                    show_pc,
                    draw_mesh,
                    level,
                    render_trainer.gt_depth_vis,
                    render_trainer.gt_im_vis,
                    render_trainer.gt_depth_vis,
                    T_WC_batch_np,
                    draw_cameras=True,
                )

                if render_trainer.do_tsdf:
                    mesh_tsdf = render_trainer.tsdf_mesh(T_WC_batch_np[-1])

            display = {}
            if show_pc or show_mesh:
                display["scene"] = scene
                # png = scene.save_image()
                # file_name = 'meshing/render_' + f'{save_count:03}' + '.png'
                # save_count += 1
                # with open(file_name, 'wb') as f:
                #     f.write(png)
                #     f.close()

            if render_trainer.do_tsdf:
                display["tsdf"] = mesh_tsdf

            display["viz"] = viz
            yield display

        print(t, loss.item())


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    parser = argparse.ArgumentParser(description="Neural Scene SLAM.")
    parser.add_argument("--config", type=str, help="input json config")
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    parser.add_argument(
        "-na",
        "--no_active",
        action="store_false",
        help="disable active sampling option",
    )
    parser.add_argument(
        "-nt", "--no_track", action="store_false", help="disable camera tracking"
    )
    parser.add_argument(
        "-nc", "--no_color", action="store_false", help="disable color optimisation"
    )
    args = parser.parse_args()

    config_file = args.config
    incremental = args.no_incremental
    do_active = args.no_active
    do_track = args.no_track
    do_color = args.no_color

    save = False
    load = False
    show_pc = False
    show_mesh = True
    save_ims = False
    show_freq = 100
    scene_draw_freq = 200
    checkpoint_freq = 500

    if save:
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        save_path = "experiments/" + time_str
        os.mkdir(save_path)
        checkpoint_path = save_path + "/checkpoints"
        os.mkdir(checkpoint_path)
    else:
        save_path = None

    if load:
        load_path = (
            ""
        )
        load_epoch = 3500
    else:
        load_path = None
        load_epoch = None

    scenes = train(
        save_ims,
        show_freq,
        device,
        config_file,
        scene_draw_freq=scene_draw_freq,
        checkpoint_freq=checkpoint_freq,
        show_pc=show_pc,
        show_mesh=show_mesh,
        save_path=save_path,
        load_path=load_path,
        load_epoch=load_epoch,
        incremental=incremental,
        do_active=do_active,
        do_track=do_track,
        do_color=do_color,
    )

    tiling = None
    if show_pc or show_mesh:
        tiling = (1, 2)
    visualisation.display_scenes.display_scenes(
        scenes, height=int(680 * 0.8), width=int(1200 * 0.8), tile=tiling
    )
