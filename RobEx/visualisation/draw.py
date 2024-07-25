import trimesh
from RobEx import geometry
import numpy as np
import open3d as o3d
import cv2


def draw_segment(t1, t2, color=(1., 1., 0.)):
    line_segment = trimesh.load_path([t1, t2])
    line_segment.colors = (color, ) * len(line_segment.entities)

    return line_segment


def draw_trajectory(trajectory, scene, color=(1., 1., 0.)):
    for i in range(trajectory.shape[0] - 1):
        if (trajectory[i] != trajectory[i + 1]).any():
            segment = draw_segment(trajectory[i], trajectory[i + 1], color)
            scene.add_geometry(segment)


def draw_camera(camera, transform, color=(0., 1., 0., 0.8), marker_height=0.2):
    marker = trimesh.creation.camera_marker(
        camera, marker_height=marker_height)
    marker[0].apply_transform(transform)
    marker[1].apply_transform(transform)
    marker[1].colors = (color, ) * len(marker[1].entities)

    return marker


def draw_cameras(eyes, ats, up, scene):
    for eye, at in zip(eyes, ats):
        R, t = geometry.transform.look_at(eye, at, up)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        transform = T @ geometry.transform.to_replica()
        camera = trimesh.scene.Camera(fov=scene.camera.fov,
                                      resolution=scene.camera.resolution)
        marker = draw_camera(camera, transform)
        scene.add_geometry(marker)


def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst


def draw_tree(H, W, levels, colors, select, class_names=None):
    im = np.zeros([H, W, 3]).astype(np.uint8) + 255
    increment_y = H // levels
    circle_size = W // 20

    y = increment_y // 2
    counter = -1
    for i in range(levels):
        nodes = 2**i
        increment_x = W // nodes
        x = increment_x // 2
        for j in range(nodes):
            if counter == -1:
                color = [100, 100, 100]
                size = 5
            else:
                color = colors[counter][::-1].tolist()
                size = circle_size

            if i < (levels - 1):
                cv2.line(im, (x, y), (x - increment_x // 4, y + increment_y),
                         (100, 100, 100), 1)
                cv2.line(im, (x, y), (x + increment_x // 4, y + increment_y),
                         (100, 100, 100), 1)

            if counter == select:
                cv2.circle(im, (x, y), size + 5, [100, 100, 100], -1)
            cv2.circle(im, (x, y), size, color, -1)

            if counter != -1 and class_names is not None:
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 0.5
                thick = 1
                text = class_names[counter]
                text_size, _ = cv2.getTextSize(text, font, scale, thick)
                text_origin = (x - text_size[0] // 2, y + text_size[1] // 2)
                cv2.putText(im, text, text_origin, font, scale,
                            (200, 200, 200), thick, cv2.LINE_AA)

            counter += 1
            x += increment_x

        y += increment_y

    return im
