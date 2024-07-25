#!/usr/bin/env python

import argparse
import itertools
import tempfile

import cv2
import imgviz
import IPython
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp
import trimesh
from scipy.spatial.transform import Rotation

import mercury
from mercury.examples.phys_imap import _reorient
from mercury.examples.phys_imap import _utils
from mercury.geometry.coordinate import Coordinate

import cv_bridge
import rospy as rp
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from ._message_subscriber import MessageSubscriber
from ._tsdf_from_depth import tsdf_from_depth
from .base_task_interface import BaseTaskInterface
from RobEx.ros.ros_bridge import RosBridge
import time

import matplotlib.pyplot as plt
from datetime import datetime
from path import Path

class ImapTaskInterface(BaseTaskInterface):
    def __init__(self, sim=False):

        rp.init_node("imap_task_interface")
        super().__init__(sim)

        self.ros_bridge = RosBridge(init_node=False, sim=sim)
        self.scene_id = None

        self.use_force = False

    def init_task(self):
        # target place
        visual_file = mercury.datasets.ycb.get_visual_file(class_id=3)

        rgba_color= (1,1,1,0.5)
        if self.sim:
            rgba_color= (1,1,1,0.)

        obj = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            quaternion=_utils.get_canonical_quaternion(class_id=3),
            rgba_color=rgba_color,
            mesh_scale=(0.99, 0.99, 0.99),  # for virtual rendering
        )
        pp.set_pose(
            obj,
            (
                (0.44410000000000166, 0.5560999999999995, 0.02929999999999988),
                (
                    -0.5032839784369476,
                    -0.4819772480647679,
                    -0.4778992452799924,
                    0.5348041517765217,
                ),
            ),
        )
        self._obj_goal = obj
        self._env.PLACE_POSE = pp.get_pose(self._obj_goal)
        c = mercury.geometry.Coordinate(*self._env.PLACE_POSE)
        c.translate([0, 0, 0.2], wrt="world")
        self._env.PRE_PLACE_POSE = c.pose
        # highlight target pose
        mesh = trimesh.load(visual_file)
        mesh.apply_transform(
            mercury.geometry.transformation_matrix(*self._env.PLACE_POSE)
        )
        pp.draw_aabb(mesh.bounds, color=(1, 0, 0, 1))

    def run(self):
        self.init_workspace()

    def scan(self):
        r_eye = 2.
        r_target = 1.5

        translation = np.eye(4, dtype=float)
        translation[:3,3] = [0.2, 0., -1.5]
        rot = Rotation.from_euler('y', 0, degrees=True)
        rotation = np.eye(4, dtype=float)
        rotation[:3,:3] = rot.as_matrix()
        transform = np.matmul(translation, rotation)
        # print("transform: ", transform)

        ## Original
        theta = np.linspace(-90, 90, num=5)
        phi = np.linspace(15, 10, num=2)

        for i in range(0,2):
            for p in phi:
                for t in theta:
                    xyz_eye = self.spher_to_cart_coord([r_eye, t, p], transform=transform)
                    xyz_target = self.spher_to_cart_coord([r_target, t, p], transform=transform)

                    print('theta: {}, phi: {}'.format(t, p))
                    print('Look at target: {}, eye: {}'.format(xyz_target, xyz_eye))
                    print('Look at target: {}, eye: {}'.format(xyz_target, xyz_eye))
                    self.look_at_target(target=xyz_target, eye=xyz_eye)

                theta = np.flip(theta, 0)

            r_eye = 0.5
            r_target = 1.6

            translation = np.eye(4, dtype=float)
            translation[:3,3] = [0.3, 0., 1.]
            rot = Rotation.from_euler('y', -15, degrees=True)
            rotation = np.eye(4, dtype=float)
            rotation[:3,:3] = rot.as_matrix()
            transform = np.matmul(translation, rotation)

            ## Original
            theta = np.linspace(-130, 130, num=5)
            phi = np.linspace(170, 170, num=1)
            theta = np.flip(theta, 0)

        print("Scan complete...")

    def wait_for_key(self):

        pressed_keys = []
        while len(pressed_keys) == 0:
            events = p.getKeyboardEvents()
            key_codes = events.keys()
            for key in key_codes:
                pressed_keys.append(key)

            if self.sim:
                self.step_simulation()

            else:
                pp.step_simulation()
                time.sleep(pp.get_time_step())

        print("pressed_keys: ", pressed_keys)

        return

    def draw_obstacles(self, obstacles):
        for obstacle in obstacles:
            bounds = pp.get_aabb(obstacle)
            print("bounds: ", bounds)
            pp.draw_aabb(bounds, color=(1, 0, 0, 1))

    def scan_pile(self):
        self.look_at_pile()
        self._scan_singleview()

    def scan_target(self):
        self.look_at_target()
        self._scan_singleview()

    def spher_to_cart_coord(self, rtp, transform=None):
        x = rtp[0]*np.sin(np.deg2rad(rtp[2]))*np.cos(np.deg2rad(rtp[1]))
        y = rtp[0]*np.sin(np.deg2rad(rtp[2]))*np.sin(np.deg2rad(rtp[1]))
        z = rtp[0]*np.cos(np.deg2rad(rtp[2]))

        if transform is None:
            return [x,y,z]

        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3,3] = [x, y, z]
        pose_matrix = np.matmul(transform, pose_matrix)
        return pose_matrix[:3,3]

    def look_at_target(self, target=None, eye=None, time_scale=1.):
        if target is None:
            if self._env.fg_object_id is None:
                # default
                target = [0.2, -0.5, 0.1]
            else:
                target = pp.get_pose(self._env.fg_object_id)[0]

        if eye is None:
            eye=[target[0] - 0.1, target[1], target[2] + 0.5]

        self.look_at(
            eye=eye,
            target=target,
            rotation_axis="z",
            time_scale=time_scale,
        )

    def add_collision_mesh(self, mesh):

        ## Reduce mesh to valid workspace
        T_bbx = Coordinate(position=[0.5,0.,0.15], quaternion=[0.,0.,0.,1.]).matrix
        workspace_bbx = trimesh.creation.box(extents=[0.6, 1.0, 0.3], transform=T_bbx)
        mesh = mesh.slice_plane(workspace_bbx.facets_origin, -workspace_bbx.facets_normal)

        f_mesh = "/home/data/scene.obj"
        trimesh.exchange.export.export_mesh(mesh, f_mesh, file_type="obj")

        mesh = Path(f_mesh)
        collision_file = mesh.stripext() + ".convex" + mesh.ext
        log_file = mesh.stripext() + "_log.txt"
        p.vhacd(mesh,
            collision_file,
            log_file,
            alpha=0.04,
            resolution=50000
        )

        scene_id = mercury.pybullet.create_mesh_body(
            visual_file=collision_file,
            collision_file=collision_file,
            position=[0.,0.,0.],
            quaternion=[0.,0.,0.,1.],
            rgba_color=[255,255,0,0.2],
        )

        if self.sim:
            for object_id in self._env.object_ids: ## Turn off collisions with objects in the scene
                p.setCollisionFilterPair(scene_id,
                    object_id,
                    -1,
                    -1,
                    0,
                )

            ## Turn off collisions between scene and robot links
            for n_joint in range(0, p.getNumJoints(self.pi.robot)):
                p.setCollisionFilterPair(scene_id,
                    self.pi.robot,
                    -1,
                    n_joint,
                    0,
                )
            ## Turn off collisions between scene and robot tipLink
            p.setCollisionFilterPair(scene_id, ## Turn off collisions between scene and robot tipLink
                self.pi.robot,
                -1,
                self.pi.ee,
                0,
            )
        self.scene_id = scene_id
        return scene_id


    def move_to_classify(self, c_target, offset=0.1, time_scale=1., obstacles=None):
        n_ik_attempts = 9

        mercury.pybullet.utils.add_debug_visualizer_pose(c_target.matrix, size=0.1)
        c_approach = c_target.copy()
        c_approach.translate([0.,0.,offset], wrt="local") ## Offset for grasping
        j = self.pi.solve_ik(
            c_approach.pose,
            move_target=self.pi.robot_model.tipLink,
            n_init=n_ik_attempts, # first attempt is from current state --> use setj if neccessary. Subsequent are random states.
            thre=None,
            rthre=[1., 1., 360.],
            rotation_axis='z',
            validate=True,
        )
        js_approach = self.pi.planj(
            j,
            obstacles=obstacles,
            min_distances=None,
            min_distances_start_goal=None,
        )
        if js_approach is None:
            return None

        mercury.pybullet.utils.add_debug_visualizer_pose(c_approach.matrix, size=0.1)
        self.pi.setj(js_approach[-1])

        ## check positions match
        tip_pose = self.pi.get_pose("tipLink")
        delta_d = np.sqrt((tip_pose[0][0] - c_approach.matrix[0,3])**2 + \
                (tip_pose[0][1] - c_approach.matrix[1,3])**2 + \
                (tip_pose[0][2] - c_approach.matrix[2,3])**2)
        if delta_d > 0.01:
            print("Error -- plan and request xyz do not match!")
            return None

        j = self.pi.solve_ik(
            c_target.pose,
            move_target=self.pi.robot_model.tipLink,
            n_init=1, # first attempt is from current state --> use setj if neccessary. Subsequent are random states.
            thre=None,
            rthre=[1., 1., 360.],
            rotation_axis='z',
            validate=True,
        )
        js_contact = self.pi.planj(
            j,
            obstacles=obstacles,
            ignore_links=[self.pi.ee],
            min_distances=None,
            min_distances_start_goal=None,
        )
        if js_contact is None:
            print("js_contact is not found, unable to contact")
            return None

        return js_approach

    def create_force_plt(self, z_arr, f_ext_arr):
        timeStr = datetime.now().strftime("%H%M%S%d%b%y")
        f_name = "/home/data/" + "f_ext_" + timeStr + ".png"

        ## Plot f_ext
        plt.clf()
        plt.plot(z_arr, f_ext_arr, 'ko')
        plt.axis([0.01, np.max(z_arr), np.min(f_ext_arr), np.max(f_ext_arr)])
        f_name = "/home/data/" + "f_ext_raw" + timeStr + ".png"
        plt.savefig(f_name)

        return True

    def classify_force(self, z_arr, f_ext_arr):
        n_max = np.argmax(f_ext_arr) + 1
        f_ext_signal = f_ext_arr[n_max-10:n_max]
        z_signal = z_arr[n_max-10:n_max]

        coef_signal = np.polyfit(z_signal,f_ext_signal,1)

        z_base_ = -coef_signal[1]/coef_signal[0] #when poly = 0
        less = np.less(z_arr, z_base_)
        f_ext_base = f_ext_arr[less]
        z_base = z_arr[less]

        coef_base = np.polyfit(z_base,f_ext_base,1)

        z_inter = (coef_signal[1] - coef_base[1])/(coef_base[0] - coef_signal[0])
        greater = np.greater_equal(z_arr, z_inter)

        f_ext_signal = f_ext_arr[greater]
        z_signal = z_arr[greater]

        n_max = np.argmax(f_ext_signal) + 1
        f_ext_signal = f_ext_signal[:n_max]
        z_signal = z_signal[:n_max]

        coef_signal = np.polyfit(z_signal,f_ext_signal,1)

        base_fn = np.poly1d(coef_base)
        signal_fn = np.poly1d(coef_signal)

        label = coef_signal[0]
        if coef_signal[0] < 1000:
            class_id = 1
            title = "soft -- m = "
        else:
            class_id = 2
            title = "hard -- m = "

        ## Plot
        timeStr = datetime.now().strftime("%H%M%S%d%b%y")
        f_name = "/home/data/" + "f_ext_" + timeStr + ".png"

        plt.clf()
        plt.plot(z_base, f_ext_base, 'go')
        plt.plot(z_signal, f_ext_signal, 'ro')

        plt.plot(z_arr, base_fn(z_arr), '--k')
        plt.plot(z_arr, signal_fn(z_arr), '--r')

        plt.title(title + str(coef_signal[0]))
        plt.savefig(f_name)

        return class_id

    def classify(self, min_dz=None, max_dz=None, time_scale=1.):
        self.real2robot()
        j_start = self.pi.getj()

        if self.sim:

            c_from = mercury.geometry.Coordinate(
                *pp.get_link_pose(self.pi.robot, self.pi.ee)
            )
            c_to = c_from.copy()
            c_to.translate([0, 0, 0.03])
            mercury.pybullet.utils.add_debug_visualizer_pose(c_to.matrix, size=0.1)
            collisions = p.rayTest(rayFromPosition=c_from.position,
                                                rayToPosition=c_to.position,
            )
            print("collisions: ", collisions)
            for collision in collisions:
                print("self.scene_id: ", self.scene_id)
                print("collision[0]: ", collision[0])
                if collision[0] != self.scene_id:
                    class_id = collision[0]
                    break
            print(class_id)
            return class_id

        else:
            self.real2robot()
            j_start = self.pi.getj()

            js = []
            c = mercury.geometry.Coordinate(
                *pp.get_link_pose(self.pi.robot, self.pi.ee)
            )
            dz_done = 0
            while True:
                c.translate([0, 0, 0.005])
                dz_done += 0.005
                j = self.pi.solve_ik(c.pose, rotation_axis="z")
                if j is None:
                    continue
                js.append(j)
                if min_dz is not None and dz_done < min_dz:
                    continue
                if max_dz is not None and dz_done >= max_dz:
                    break

            self.ros_bridge.set_f_ext_record(True)
            self.movejs(js, time_scale=time_scale, wait=True)
            self.ros_bridge.set_f_ext_record(False)

            label = -99 + 49
            # class_id = None
            class_id = label
            if self.use_force:

                ## Force measurement
                f_ext_start = self.ros_bridge.f_ext[0]
                tip_z_start = self.ros_bridge.tip_z[0]

                z_arr = np.asarray(self.ros_bridge.tip_z)
                f_ext_arr = np.asarray(self.ros_bridge.f_ext)

                if z_arr.shape[0] > f_ext_arr.shape[0]:
                    z_arr = z_arr[0:f_ext_arr.shape[0]]
                else:
                    f_ext_arr = f_ext_arr[0:z_arr.shape[0]]
                z_arr = np.full(len(z_arr), tip_z_start) - z_arr

                if len(z_arr) > 0:
                    class_id = self.classify_force(z_arr, f_ext_arr)
                    label = class_id

            else:

                ## Spectrometer
                spec_class = self.ros_bridge.get_spec_class(mode=2)
                label = spec_class
                class_id = spec_class

        self.movejs([j_start], time_scale=time_scale)
        return class_id

    def rosmsgs_to_env(self, tsdf=True):
        cam_msg = rp.wait_for_message(
            "/camera/color/camera_info", CameraInfo
        )
        (
            depth_msg,
            cls_msg,
            label_msg,
            obj_poses_msg,
        ) = self._subscriber_reorientbot.msgs

        K = np.array(cam_msg.K).reshape(3, 3)

        bridge = cv_bridge.CvBridge()
        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan
        label = bridge.imgmsg_to_cv2(label_msg)

        camera_to_base = self.lookup_transform(
            "panda_link0",
            obj_poses_msg.header.frame_id,
            time=obj_poses_msg.header.stamp,
        )

        for obj_pose_msg in obj_poses_msg.poses:
            class_id = obj_pose_msg.class_id
            if class_id != _utils.get_class_id(self._obj_goal):
                continue

            pose = obj_pose_msg.pose
            obj_to_camera = (
                (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ),
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ),
            )
            obj_to_base = pp.multiply(camera_to_base, obj_to_camera)

            visual_file = mercury.datasets.ycb.get_visual_file(class_id)
            collision_file = mercury.pybullet.get_collision_file(visual_file)
            obj = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=collision_file,
                position=obj_to_base[0],
                quaternion=obj_to_base[1],
            )
            break
        else:
            raise RuntimeError("Target object is not found")

        if tsdf:
            target_instance_id = obj_pose_msg.instance_id

            mask = label == target_instance_id
            mask = (
                cv2.dilate(
                    imgviz.bool2ubyte(mask),
                    kernel=np.ones((8, 8)),
                    iterations=3,
                )
                == 255
            )
            depth_masked = depth.copy()
            depth_masked[mask] = np.nan
            tsdf = tsdf_from_depth(depth_masked, camera_to_base, K)
            with tempfile.TemporaryDirectory() as tmp_dir:
                visual_file = path.Path(tmp_dir) / "tsdf.obj"
                tsdf.export(visual_file)
                collision_file = mercury.pybullet.get_collision_file(
                    visual_file, resolution=10000
                )
                bg_structure = mercury.pybullet.create_mesh_body(
                    visual_file=visual_file,
                    collision_file=collision_file,
                    rgba_color=(0.5, 0.5, 0.5, 1),
                )
            self._env.bg_objects.append(bg_structure)

        if self._env.fg_object_id is not None:
            pp.remove_body(self._env.fg_object_id)
        self._env.fg_object_id = obj
        self._env.object_ids = [obj]
        self._env.update_obs()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", dest="cmd")
    args = parser.parse_args()

    rp.init_node("imap_task_interface")
    self = ImapTaskInterface()  # NOQA

    print("Before reset pose...")
    self.reset_pose(time_scale=time_scale)
    print("After reset pose...")

    self.scan()


if __name__ == "__main__":
    main()
