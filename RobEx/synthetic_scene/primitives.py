import trimesh
import numpy as np
import pyrender
import imgviz
import cv2
import random

def bottom_point(mesh, dump=True):
    if dump:
        mesh_dump = mesh.dump().sum()
    else:
        mesh_dump = mesh
    mesh_vertices = mesh_dump.vertices
    vertices_z = mesh_vertices[:, 1]
    bottom = np.min(vertices_z)

    return bottom


class ObjectMesh(object):
    def __init__(self):
        super(ObjectMesh, self).__init__()


if __name__ == "__main__":
    mesh_objs = []
    types = ['c', 's', 'c', 's', 'c', 's', 'c', 's']
    random.shuffle(types)
    colors = [[255, 0, 0],
              [0, 255, 0],
              [255, 0, 0],
              [0, 255, 0],
              [255, 0, 0],
              [0, 255, 0],
              [255, 0, 0],
              [0, 255, 0]]

    for i in range(8):
        obj = ObjectMesh()
        obj.translate = np.array([-2.2 + 0.6 * i, 0.0, 1.0])
        obj_type = types[i]
        if obj_type == 'b':
            obj.mesh = trimesh.primitives.Box(
                extents=[.3, .3, .3])
        elif obj_type == 's':
            obj.mesh = trimesh.primitives.Sphere(radius=0.2)
        elif obj_type == 'c':
            rot = trimesh.transformations.rotation_matrix(
                angle=np.pi / 2, direction=[1, 0, 0])
            obj.mesh = trimesh.primitives.Cylinder(
                radius=0.2, height=0.6, transform=rot)
        obj.mesh.visual.vertex_colors = colors[i]
        mesh_objs.append(obj)

    tri_scene = trimesh.Scene()
    room_file = ("")
    room_mesh = trimesh.load(room_file, process=False)
    room_mesh = room_mesh.dump().sum()
    room_mesh.visual.vertex_colors = [255, 255, 255, 255]
    bottom_room = bottom_point(room_mesh, dump=False)
    tri_scene.add_geometry(room_mesh)

    for mesh_obj in mesh_objs:
        bottom_mesh = bottom_point(mesh_obj.mesh, dump=False)
        distance = bottom_room - bottom_mesh
        translation = mesh_obj.translate + np.array([0, distance, 0])
        mesh_obj.mesh.apply_translation(translation)
        tri_scene.add_geometry(mesh_obj.mesh)

    H = 720
    W = 1280
    fx, fy = (
        W / 2.0,
        W / 2.0,
    )

    tri_scene.set_camera()
    tri_scene.camera.focal = (fx, fy)
    tri_scene.camera.resolution = (W, H)
    print(tri_scene.camera.K)

    T_WC = np.eye(4)
    T_WC[:3, 3] = [0., 2.5, 2.3]
    rot = trimesh.transformations.rotation_matrix(
        angle=-np.pi / 3, direction=[1, 0, 0])
    T_WC = T_WC @ rot
    tri_scene.camera_transform = T_WC
    tri_scene.show()
    path = ("scene.obj")
    png = tri_scene.save_image()
    export = tri_scene.export("scene.obj")
    mesh_tri = trimesh.load(path, 'obj', process=False)
    mesh = pyrender.Mesh.from_trimesh(mesh_tri)
    scene = pyrender.Scene()
    scene.add(mesh)

    cx, cy = (W) / 2.0, (H) / 2.0
    print(fx, fy, cx, cy)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    nc = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(nc)

    for _ in range(50):
        light = pyrender.PointLight(color=np.ones(3), intensity=20.0)
        T = np.eye(4)
        T[:3, 3] = np.random.uniform(size=[3], low=-4, high=4)
        scene.add(light, pose=T)

    scene.set_pose(nc, pose=T_WC)

    r = pyrender.OffscreenRenderer(W, H)
    _, depth = r.render(scene)
    depth_viz = imgviz.depth2rgb(depth)

    depth = depth * 65535.0 * 0.1
    depth = depth.astype(np.uint16)

    cv2.imshow("depth", depth_viz)
    cv2.waitKey(1)

    for idx in range(100):
        num = f"{idx:06}"
        cv2.imwrite("" + num + ".png", depth)

        file_name = "" + num + ".png"
        with open(file_name, 'wb') as f:
            f.write(png)
            f.close()
