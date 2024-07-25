#!/usr/bin/env python

import argparse
import time

import IPython
import numpy as np
import pybullet as p
import pybullet_planning as pp
import rospy as rp
from scipy.spatial.transform import Rotation
import scipy

import mercury

import actionlib
from actionlib_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from franka_msgs.msg import ErrorRecoveryAction
from franka_msgs.msg import ErrorRecoveryGoal
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointField
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty
from std_srvs.srv import SetBool
import tf

from mercury.examples.phys_imap import _env

from ._message_subscriber import MessageSubscriber
from ._panda import Panda
from ._panda_ros_robot_interface import PandaROSRobotInterface
from . import _pybullet

class BaseTaskInterface:
    def __init__(self, sim=False):
        self._tf_listener = tf.listener.TransformListener(
            cache_time=rospy.Duration(60)
        )

        self._ri = PandaROSRobotInterface(robot=Panda())
        self.class_ids = [1,2,3,4,11,16]

        self.sim = sim
        self._env = _env.Env(
            gui=True,
            class_ids=self.class_ids,
            real=(not self.sim),
            robot_model="franka_panda/panda_drl",
            debug=False,
        )
        self._env.reset()

        if not self.sim:
            self.real2robot()

        self._subscriber_base = MessageSubscriber(
            [
                ("/camera/color/camera_info", CameraInfo),
                ("/camera/color/image_rect_color", Image),
                ("/camera/aligned_depth_to_color/image_raw", Image),
            ],
            callback=self._subscriber_base_callback,
        )
        self._subscriber_base_points_stamp = None
        self._subscriber_base_points = None


        if self.sim:
            self.delta_t = 0.
            self.camera_freq = 20
            self.delta_t_joint = 0.
            self.joint_freq = 100

            self.bridge = CvBridge()
            camera_name = "camera"
            color_topic = '/{0}/color/image_rect_color'.format(camera_name)
            aligned_depth_topic = '/{0}/aligned_depth_to_color/image_raw'.format(camera_name)

            self.tf_broadcaster = tf.TransformBroadcaster()
            self.pub_rgb = rp.Publisher(color_topic, Image, queue_size=10)
            self.pub_depth = rp.Publisher(aligned_depth_topic, Image, queue_size=10)
            self.pub_joint_state = rp.Publisher("franka_state_controller/joint_states", JointState, queue_size=10)
            self.pub_pointcloud = rp.Publisher("/camera/depth_registered/points", PointCloud2, queue_size=10)

    def _subscriber_base_callback(self, info_msg, rgb_msg, depth_msg):
        HZ = 5
        if self._subscriber_base_points_stamp is not None and (
            info_msg.header.stamp - self._subscriber_base_points_stamp
        ) < rospy.Duration(1 / HZ):
            return

        K = np.array(info_msg.K).reshape(3, 3)
        bridge = CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan

        try:
            camera_to_base = self.lookup_transform(
                "panda_link0",
                info_msg.header.frame_id,
                time=info_msg.header.stamp,
            )
        except tf.ExtrapolationException:
            return

        pcd = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        pcd = mercury.geometry.transform_points(
            pcd, mercury.geometry.transformation_matrix(*camera_to_base)
        )

        subscriber_base_points = None #_pybullet.draw_points(pcd, rgb, size=1)

        if self._subscriber_base_points is not None:
            pp.remove_debug(self._subscriber_base_points)
        self._subscriber_base_points = subscriber_base_points
        self._subscriber_base_points_stamp = info_msg.header.stamp

    @property
    def pi(self):
        return self._env.ri

    @property
    def ri(self):
        return self._ri

    def start_passthrough(self):
        passthroughs = [
            "/camera/color/image_rect_color_passthrough",
        ]
        for passthrough in passthroughs:
            client = rospy.ServiceProxy(passthrough + "/request", Empty)
            client.call()

    def stop_passthrough(self):
        passthroughs = [
            "/camera/color/image_rect_color_passthrough",
        ]
        for passthrough in passthroughs:
            client = rospy.ServiceProxy(passthrough + "/stop", Empty)
            client.call()

    def lookup_transform(self, target_frame, source_frame, time, timeout=None):
        if timeout is not None:
            self._tf_listener.waitForTransform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=time,
                timeout=timeout,
            )
        return self._tf_listener.lookupTransform(
            target_frame=target_frame, source_frame=source_frame, time=time
        )

    def recover_from_error(self):
        print("Waiting for server")
        client = actionlib.SimpleActionClient(
            "/franka_control/error_recovery", ErrorRecoveryAction
        )
        client.wait_for_server()
        print("Connected to server")

        print("client.get_state(): ", client.get_state())
        if client.get_state() == GoalStatus.SUCCEEDED:
            return True

        goal = ErrorRecoveryGoal()

        print("Sending goal")
        state = client.send_goal(goal)
        rp.sleep(0.5)
        print("Complete goal")
        return True

    def real2robot(self):
        self.ri.update_robot_state()
        self.pi.setj(self.ri.potentio_vector())
        for attachment in self.pi.attachments:
            attachment.assign()

    def matrix_to_trans_quart(self, matrix):
        trans = matrix[0:3,3]
        rot = Rotation.from_matrix(matrix[:3,:3])
        quat = rot.as_quat()

        return trans, quat

    def broadcast_tf_frame(self, matrix, child_frame, parent_frame, time=None):
        trans, quat = self.matrix_to_trans_quart(matrix)

        if time is None:
            time = rp.Time.now()

        self.tf_broadcaster.sendTransform(
            translation=trans,
            rotation=quat,
            time=time,
            child=child_frame,
            parent=parent_frame)

        return True

    def create_and_publish_pointcloud(self, rgb, depth, transform=None):
        K = self.pi.get_opengl_intrinsic_matrix()
        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        if transform is None:
            camera_T_WC = self.pi.get_pose("camera_link")
            camera_T_WC = mercury.geometry.transformation_matrix(camera_T_WC[0], camera_T_WC[1])
        else:
            camera_T_WC = transform

        pcd_in_map = mercury.geometry.transform_points(
            pcd_in_camera,
            camera_T_WC,
        )

        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "map"

        msg.height = pcd_in_map.shape[1]
        msg.width = pcd_in_map.shape[0]

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * pcd_in_map.shape[0]
        msg.is_dense = False
        msg.data = np.asarray(pcd_in_map, np.float32).tostring()

        self.pub_pointcloud.publish(msg)
        return

    def step_simulation(self):

        self.pi.update_robot_model()
        time_now = rp.Time.now()

        if (self.delta_t_joint > 1./self.joint_freq):
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = time_now
            joint_state_msg.name = self.pi.getj_name()
            joint_state_msg.position = self.pi.getj()

            self.pub_joint_state.publish(joint_state_msg)
            self.delta_t_joint = 0

        if (self.delta_t > 1./self.camera_freq):

            rgb_img, depth_raw, segm = self.pi.get_camera_image()
            depth_raw *= 1000 #convert from m to mm
            depth_raw = depth_raw.astype(np.uint16)

            ## Masking robot from depth
            robot_mask_link_low = np.full(segm.shape, self.pi.robot)
            robot_mask_link_high = np.full(segm.shape, (self.pi.robot + p.getNumJoints(self.pi.robot)))
            robot_mask = np.equal(segm, robot_mask_link_low)

            ## dialate mask
            depth_raw = np.ma.array(depth_raw, mask=robot_mask, fill_value=0.)

            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, "rgb8")
            depth_raw_msg = self.bridge.cv2_to_imgmsg(depth_raw, encoding='16UC1')

            # time_now = rp.Time.now()
            rgb_msg.header.stamp = time_now
            depth_raw_msg.header.stamp = time_now

            joint_state_msg = JointState()
            joint_state_msg.header.stamp = time_now
            joint_state_msg.name = self.pi.getj_name()
            joint_state_msg.position = self.pi.getj()

            self.pub_rgb.publish(rgb_msg)
            self.pub_depth.publish(depth_raw_msg)
            self.pub_joint_state.publish(joint_state_msg)

            self.delta_t = 0

        pp.step_simulation()
        time.sleep(pp.get_time_step())

        self.delta_t += pp.get_time_step()
        self.delta_t_joint += pp.get_time_step()
        return

    def visjs(self, js):
        for j in js:
            for _ in self.pi.movej(j):
                self.step_simulation()

    def movejs(self, js, time_scale=None, wait=True, retry=False, wait_callback=None):

        if time_scale is None:
            time_scale = 3
        js = np.asarray(js)

        if self.sim:
            for j in js:
                for _ in self.pi.movej(j, speed=0.01/time_scale):
                    self.step_simulation()
        else:
            if not self.recover_from_error():
                return

            self.real2robot()
            j_init = self.pi.getj()

            self.ri.angle_vector_sequence(
                js, time_scale=time_scale, max_pos_accel=0.8
            )
            if wait:
                success = self.wait_interpolation(callback=wait_callback)
                if success or not retry:
                    return

                self.real2robot()
                j_curr = self.pi.getj()

                js = np.r_[[j_init], js]

                for i in range(len(js) - 1):
                    dj1 = js[i + 1] - j_curr
                    dj2 = js[i + 1] - js[i]
                    dj1[abs(dj1) < 0.01] = 0
                    dj2[abs(dj2) < 0.01] = 0
                    if (np.sign(dj1) == np.sign(dj2)).all():
                        break
                else:
                    return
                self.movejs(
                    js[i + 1 :], time_scale=time_scale, wait=wait, retry=False
                )

    def wait_interpolation(self, callback=None):
        self._subscriber_base.subscribe()
        controller_actions = self.ri.controller_table[self.ri.controller_type]

        t_ = 0
        j = np.asarray(self.pi.getj())
        pre_j = j
        while True:
            states = [action.get_state() for action in controller_actions]
            if all(s >= GoalStatus.SUCCEEDED for s in states):
                break
            self.real2robot()
            if callback is not None:
                callback()
            rospy.sleep(0.01)

            t_ += 1
            if t_ == 200:
                j = np.asarray(self.pi.getj())
                j_close = np.allclose(j, pre_j, rtol=0., atol=1E-4) #absolute(a - b) <= (atol + rtol * absolute(b))
                print("j_close: ", j_close)
                if j_close:
                    break
                pre_j = j
                t_ = 0

        self._subscriber_base.unsubscribe()
        if not all(s == GoalStatus.SUCCEEDED for s in states):
            rospy.logwarn("Some joint control requests have failed")
            return False
        return True

    def start_grasp(self):
        client = rospy.ServiceProxy("/set_suction", SetBool)
        client.call(data=True)

    def stop_grasp(self):
        client = rospy.ServiceProxy("/set_suction", SetBool)
        client.call(data=False)

    def reset_pose(self, *args, **kwargs):
        print("Resetting pose...")
        self.movejs([self.pi.homej], *args, **kwargs)
        print("Resetting pose done...")


    def _solve_ik_for_look_at(self, eye, target, rotation_axis=True):
        c = mercury.geometry.Coordinate.from_matrix(
            mercury.geometry.look_at(eye, target)
        )
        if rotation_axis is True:
            for _ in range(4):
                c.rotate([0, 0, np.deg2rad(90)])
                if abs(c.euler[2] - np.deg2rad(-90)) < np.pi / 4:
                    break
        j = self.pi.solve_ik(
            c.pose,
            move_target=self.pi.robot_model.camera_link,
            n_init=20,
            thre=0.05,
            rthre=np.deg2rad(15),
            rotation_axis=rotation_axis,
            validate=True,
        )
        if j is None:
            rospy.logerr("j is not found")
            return
        return j

    def look_at(self, eye, target, rotation_axis=True, *args, **kwargs):
        j = self._solve_ik_for_look_at(eye, target, rotation_axis)
        if j is None:
            return False

        print("j: ", j)
        js = self.pi.planj(
            j,
            obstacles=None,
            min_distances=None,
            min_distances_start_goal=None,
        )
        # js = [j]
        if js is None:
            return False
        else:
            self.movejs(js, *args, **kwargs)

    def look_at_pile(self, *args, **kwargs):
        self.look_at(eye=[0.5, 0, 0.7], target=[0.5, 0, 0], *args, **kwargs)

    def init_workspace(self):
        # light
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, True, lightPosition=(100, -100, 0.5)
        )

        # table
        pp.set_texture(self._env.plane)

        # left wall
        obj = pp.create_box(w=3, l=0.01, h=1.05, color=(0.6, 0.6, 0.6, 1))
        pp.set_pose(
            obj,
            (
                (-0.0010000000000000002, 0.6925000000000028, 0.55),
                (0.0, 0.0, 0.0194987642109932, 0.9998098810245096),
            ),
        )
        self._env.bg_objects.append(obj)

        # back wall
        obj = pp.create_box(w=0.01, l=3, h=1.05, color=(0.7, 0.7, 0.7, 1))
        pp.set_pose(obj, ([-0.4, 0, 1.05 / 2], [0, 0, 0, 1]))
        self._env.bg_objects.append(obj)

        # ceiling
        obj = pp.create_box(w=3, l=3, h=0.5, color=(1, 1, 1, 1))
        pp.set_pose(obj, ([0, 0, 0.25 + 1.05], [0, 0, 0, 1]))
        self._env.bg_objects.append(obj)

        # bin
        obj = mercury.pybullet.create_bin(
            X=0.3, Y=0.3, Z=0.11, color=(0.7, 0.7, 0.7, 1)
        )
        pp.set_pose(
            obj,
            (
                (0.4495000000000015, 0.5397000000000006, 0.059400000000000126),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        self._env.bg_objects.append(obj)
        self._bin = obj


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", dest="cmd")
    args = parser.parse_args()

    rospy.init_node("base_task_interface")
    self = BaseTaskInterface()  # NOQA

    if args.cmd:
        exec(args.cmd)

    IPython.embed(header="base_task_interface")


if __name__ == "__main__":
    main()
