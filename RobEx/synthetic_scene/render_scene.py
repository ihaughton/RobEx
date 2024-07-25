import numpy as np
import trimesh
import pyrender
import cv2
import imgviz
from RobEx.geometry import transform

if __name__ == "__main__":
    save = True
    path = ("")
    mesh_tri = trimesh.load(path, process=False)
    mesh = pyrender.Mesh.from_trimesh(mesh_tri)
    scene = pyrender.Scene()
    scene.add(mesh)

    traj_file = ("")
    Ts = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

    H = 96
    W = 128
    fx, fy = (
        W / 2.0,
        W / 2.0,
    )
    cx, cy = (W - 1.0) / 2.0, (H - 1.0) / 2.0
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    nc = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(nc)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    nl = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(nl)
    for idx, T in enumerate(Ts):
        scene.set_pose(nc, pose=T @ transform.to_trimesh())

        scene.set_pose(nl, pose=T @ transform.to_trimesh())
        r = pyrender.OffscreenRenderer(W, H)
        color, depth = r.render(scene)
        depth_viz = imgviz.depth2rgb(depth, min_value=0.1, max_value=5.)

        depth = depth * 65535.0 * 0.1
        depth = depth.astype(np.uint16)
        num = f"{idx:06}"

        cv2.imshow("depth", depth_viz)
        cv2.imshow("color", color)
        cv2.waitKey(1)

        if save:
            cv2.imwrite("results/depth" + num + ".png", depth)
            cv2.imwrite("results/frame" + num + ".jpg", color)
