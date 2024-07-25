import os
from pathlib import Path
import re
import open3d as o3d
import numpy as np
from RobEx.train import trainer
from RobEx.label.ilabel import iLabel
from RobEx.render import render_rays
from RobEx.mapping import occupancy
from RobEx import visualisation
import torch
import argparse
import cv2
import trimesh

def gen_mesh(
        render_trainer, ilabel, mesh_geo,
        mesh_frame, vis, view_ctl, extrinsic, h_level,
        save=False, save_dir='./save_live/', save_idx=0):

    if render_trainer.grid_pc is not None:
        with torch.set_grad_enabled(False):
            alphas, _, sems = occupancy.chunks(
                render_trainer.grid_pc,
                render_trainer.chunk_size,
                render_trainer.fc_occ_map,
                render_trainer.n_embed_funcs,
                render_trainer.B_layer,
                do_sem=True,
                do_hierarchical=render_trainer.do_hierarchical,
                to_cpu=True
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

        do_sem = render_trainer.do_sem
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
            do_sem=do_sem,
            do_hierarchical=render_trainer.do_hierarchical,
            colormap=colormap,
            h_level=h_level
        )

        if save:
            # save mesh
            output_scene_file = f"{save_dir}/scene_{save_idx}.ply"
            trimesh.exchange.export.export_mesh(occ_mesh, output_scene_file)
            return None

        # occ_mesh.show()
        mesh = visualisation.draw.trimesh_to_open3d(occ_mesh)

        if mesh_geo is not None:
            mesh_geo.clear()
            reset_bound = False
        else:
            reset_bound = True

        mesh_geo = mesh
        vis.add_geometry(mesh_geo, reset_bounding_box=reset_bound)

        cam = view_ctl.convert_to_pinhole_camera_parameters()
        cam.extrinsic = extrinsic
        view_ctl.convert_from_pinhole_camera_parameters(cam)
        if mesh_frame is not None:
            mesh_frame.clear()

        return mesh_geo


def gen_mesh_render(device,
                    config_file,
                    load_folder,
                    grid_dim=512,
                    follow=False,
                    freq=10):
    render_trainer = trainer.RenderTrainer(
        device,
        config_file,
        load_path=None,
        load_epoch=None,
        do_mesh=True,
        incremental=True,
        do_track=True,
        do_color=True,
        do_sem=True
    )

    ilabel = iLabel(render_trainer.do_hierarchical,
                    render_trainer.do_names,
                    render_trainer.label_cam,
                    render_trainer.device,
                    render_trainer.n_classes,
                    None)

    paths = sorted(Path(load_folder).iterdir(), key=os.path.getmtime)

    latest_date = paths[-1]

    transform_path = os.path.join(latest_date, "bounds_T.npy")
    transform = np.load(transform_path)
    extents_path = os.path.join(latest_date, "bounds_extents.npy")
    extents = np.load(extents_path)
    render_trainer.grid_dim = grid_dim
    render_trainer.set_scene_bounds(extents, transform)

    save_mesh_path = os.path.join(latest_date, "mesh_res")
    if not os.path.isdir(save_mesh_path):
        os.mkdir(save_mesh_path)

    viz_path = os.path.join(latest_date, "live_ims")
    poses_path = os.path.join(latest_date, "poses")
    network_path = os.path.join(latest_date, "network")
    sem_path = os.path.join(latest_date, "semantics")

    viz_names = os.listdir(viz_path)
    viz_names.sort(key=lambda f: int(re.sub('\D', '', f)))

    mesh_geo = None
    mesh_frame = None
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(render_trainer.W),
                      height=int(render_trainer.H),
                      left=600, top=50, visible=False)
    view_ctl = vis.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        render_trainer.W, render_trainer.H,
        render_trainer.fx, render_trainer.fy,
        render_trainer.cx, render_trainer.cy)

    split_name = re.split('[._]', viz_names[-2])
    number = split_name[1]

    kf_Ts_name = "kf_Ts_" + number + ".npy"
    kf_Ts_file = os.path.join(poses_path, kf_Ts_name)

    kf_Ts = np.load(kf_Ts_file)

    T_WC_name = "T_" + number + ".npy"
    T_WC_file = os.path.join(poses_path, T_WC_name)
    T_WC = np.load(T_WC_file)

    checkpoint_name = "epoch_" + number + ".pth"
    chechpoint_load_file = os.path.join(
        network_path, checkpoint_name
    )
    checkpoint = torch.load(chechpoint_load_file)
    render_trainer.fc_occ_map.load_state_dict(
        checkpoint["model_state_dict"])
    if render_trainer.B_layer:
        render_trainer.B_layer.load_state_dict(
            checkpoint["B_layer_state_dict"])

    T_CW_np = np.linalg.inv(T_WC[0])
    save_idx = 0
    mesh_geo = gen_mesh(
        render_trainer, ilabel, mesh_geo,
        mesh_frame, vis, view_ctl, T_CW_np, 0,
        save=True, save_dir=save_mesh_path, save_idx=number)

    try:
        cam = view_ctl.convert_to_pinhole_camera_parameters()
        fixed_extrinsic = cam.extrinsic
        while True and mesh_geo is not None:
            vis.poll_events()
            vis.update_renderer()
    except KeyboardInterrupt:
        pass

    mesh_render_idx = 0
    for viz_name in viz_names:
        split_name = re.split('[._]', viz_name)
        number = split_name[1]
        kf_Ts_name = "kf_Ts_" + number + ".npy"
        kf_Ts_file = os.path.join(poses_path, kf_Ts_name)

        if os.path.isfile(kf_Ts_file):
            kf_Ts = np.load(kf_Ts_file)
            T_WC_name = "T_" + number + ".npy"
            T_WC_file = os.path.join(poses_path, T_WC_name)
            T_WC = np.load(T_WC_file)

            class_kf_name = "class_kf_" + number + ".npy"
            class_kf_file = os.path.join(sem_path, class_kf_name)
            class_kf = np.load(class_kf_file)

            if render_trainer.do_hierarchical:
                h_level = class_kf[2]
            else:
                h_level = None

            if mesh_render_idx % freq == 0:
                checkpoint_name = "epoch_" + number + ".pth"
                chechpoint_load_file = os.path.join(
                    network_path, checkpoint_name
                )
                checkpoint = torch.load(chechpoint_load_file)
                render_trainer.fc_occ_map.load_state_dict(
                    checkpoint["model_state_dict"])
                render_trainer.B_layer.load_state_dict(
                    checkpoint["B_layer_state_dict"])

                if follow:
                    T = np.linalg.inv(T_WC[0])
                else:
                    T = fixed_extrinsic

                mesh_geo = gen_mesh(
                    render_trainer, ilabel, mesh_geo,
                    mesh_frame, vis, view_ctl, T, h_level,
                    save=True, save_dir=save_mesh_path, save_idx=number)

            if mesh_geo is None:
                continue

            if follow is False:
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.3)
                mesh_frame.transform(T_WC[0])
                vis.add_geometry(mesh_frame, reset_bounding_box=False)

            mesh_render_idx += 1
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(False)
            image = (np.asarray(image) * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            save_mesh_file = os.path.join(
                save_mesh_path, "mesh_" + number + ".png")
            cv2.imwrite(save_mesh_file, image)

            if mesh_frame is not None:
                mesh_frame.clear()


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    parser = argparse.ArgumentParser(description='gen ilabel vid')
    parser.add_argument('--config', type=str, help='input json config')
    parser.add_argument('-i',
                        '--input_dir',
                        type=str,
                        default='./live_save',
                        help='Directory where live clicks were saved')

    args = parser.parse_args()
    config_file = args.config
    input_dir = args.input_dir

    gen_mesh_render(
        device,
        config_file,
        input_dir
    )
