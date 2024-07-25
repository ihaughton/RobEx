import trimesh
import numpy as np
import scipy.spatial

from RobEx import geometry
from RobEx import visualisation


def bound_box(bounding_box_extents, transform):
    box = trimesh.creation.box(
        extents=bounding_box_extents, transform=transform)

    return box


class Scene(object):
    """scene for trajectory generation"""

    def __init__(self, path, visual=True, draw_bounds=True,
                 up=[0., 0., 1.],
                 trans_eyes=[0.0, 0.0, 0.3],
                 scale_eyes=[0.5, 0.5, 0.5],
                 trans_ats=[0.0, 0.0, -0.4],
                 scale_ats=[1, 1, 0.5]):
        super(Scene, self).__init__()
        self.up = up
        self.path = path
        self.visual = visual
        self.scene_mesh = trimesh.exchange.load.load(self.path, process=False)
        self.T_extent_to_scene, self.bound_scene_extents = trimesh.bounds.oriented_bounds(
            self.scene_mesh)
        self.T_extent_to_scene = np.linalg.inv(self.T_extent_to_scene)
        if visual:
            self.scene = trimesh.Scene()
            self.scene.add_geometry(self.scene_mesh)

        self.transform_eyes = np.eye(4)
        self.transform_eyes[:3, 3] = trans_eyes
        self.transform_eyes[:3, :3] *= np.array(scale_eyes)[:, None]
        self.transform_eyes = self.T_extent_to_scene @ self.transform_eyes
        self.box_eyes = bound_box(
            self.bound_scene_extents, self.transform_eyes)

        self.transform_ats = np.eye(4)
        self.transform_ats[:3, 3] = trans_ats
        self.transform_ats[:3, :3] *= np.array(scale_ats)[:, None]
        self.transform_ats = self.T_extent_to_scene @ self.transform_ats
        self.box_ats = bound_box(
            self.bound_scene_extents, self.transform_ats)
        if visual and draw_bounds:
            self.box_eyes.visual.face_colors = ((1.0, 0.0, 0.0, 0.2),) * 12
            self.scene.add_geometry(self.box_eyes)

            self.box_ats.visual.face_colors = ((0.0, 1.0, 1.0, 0.2),) * 12
            self.scene.add_geometry(self.box_ats)

    def sort(self, points):
        assert points.ndim == 2, "points must be 2 dimensional"
        assert points.shape[1] == 3, "points shape must be (N, 3)"

        points_left = points.copy()[1:]
        points_sorted = [points[0]]
        while len(points_sorted) < (len(points) - 1):  # drop last point
            kdtree = scipy.spatial.KDTree(points_left)
            _, index = kdtree.query(points_sorted[-1])
            points_sorted.append(kdtree.data[index])
            points_left = points_left[np.arange(len(points_left)) != index]
        points_sorted = np.array(points_sorted, dtype=float)
        return points_sorted

    def generate_trajectory(self, n_anchors, n_points):
        eye_anchors = np.random.uniform(
            low=-self.bound_scene_extents / 2.0,
            high=self.bound_scene_extents / 2.0,
            size=[n_anchors, 3],
        )
        eye_anchors = trimesh.transform_points(
            eye_anchors, self.transform_eyes)
        eye_anchors = self.sort(eye_anchors)
        eyes = geometry.transform.interpolation(eye_anchors, n_points)

        at_anchors = np.random.uniform(
            low=-self.bound_scene_extents / 2.0,
            high=self.bound_scene_extents / 2.0,
            size=[n_anchors, 3],
        )
        at_anchors = trimesh.transform_points(
            at_anchors, self.transform_ats)
        at_anchors = self.sort(at_anchors)
        ats = geometry.transform.interpolation(at_anchors, n_points)

        if self.visual:
            visualisation.draw.draw_trajectory(eyes, self.scene)
            visualisation.draw.draw_cameras(
                eye_anchors, at_anchors, self.up, self.scene)

        self.eyes, self.ats = eyes, ats

    def generate_ellipse_trajectory(self, n_points, turns=2):
        angles = np.linspace(0, turns * 2 * np.pi, num=n_points)
        x1 = self.box_eyes.extents[0] / 2 * np.cos(angles)
        y1 = self.box_eyes.extents[1] / 2 * np.sin(angles)

        x2 = self.box_ats.extents[0] / 2 * np.cos(angles)
        y2 = self.box_ats.extents[1] / 2 * np.sin(angles)

        z = np.full(n_points, 0)

        self.eyes = np.stack((x1, y1, z), axis=1)
        self.ats = np.stack((x2, y2, z), axis=1)

        self.eyes += self.box_eyes.centroid
        self.ats += self.box_ats.centroid

        if self.visual:
            visualisation.draw.draw_trajectory(self.eyes, self.scene)
            n_views = 20
            cam_view_inds = np.arange(n_views) * (n_points // n_views)
            visualisation.draw.draw_cameras(
                self.eyes[cam_view_inds],
                self.ats[cam_view_inds],
                self.up, self.scene)

    def save(self, path):
        with open(path, 'w') as out:
            for eye, at in zip(self.eyes, self.ats):
                R, t = geometry.transform.look_at(eye, at, self.up)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t

                transform = T @ geometry.transform.to_replica()
                np.savetxt(out, transform.reshape([1, 16]))

    def show(self):
        self.scene.show()
