#!/usr/bin/env python3

from scipy.spatial.transform import Rotation
import numpy as np
import json
import cv2
import imgviz

import rospy as rp
import rospkg
import tf
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import UInt16
from std_srvs.srv import SetBool, SetBoolRequest

import mercury
from . import spectrometer

import skrobot

class RosBridge():

    def __init__(self, **kwargs):

        self.sim = kwargs.get("sim", True)
        self.rviz = kwargs.get("rviz", True)
        self.init_node = kwargs.get("init_node", True)
        self.name = kwargs.get("name", "ros_bridge")
        self.config = kwargs.get("config", None)

        self.bridge = CvBridge()
        camera_name = "camera"
        color_topic = '/{0}/color/image_rect_color'.format(camera_name)
        aligned_depth_topic = '/{0}/aligned_depth_to_color/image_raw'.format(camera_name)
        f_ext_topic = '/franka_state_controller/F_ext'
        f_ext_record = '/franka_state_controller/F_ext/record'

        self.joint_states_topic = "/joint_states"
        self.pose_topic = "/camera_pose"

        if self.init_node:
            print("initiating node: ", self.name)
            rp.init_node(self.name)

        self.color = None
        self.aligned_depth = None
        self.joint_state = None
        color = message_filters.Subscriber(color_topic, Image)
        aligned_depth = message_filters.Subscriber(aligned_depth_topic, Image)
        joint_state = message_filters.Subscriber("/franka_state_controller/joint_states", JointState)
        ts = message_filters.ApproximateTimeSynchronizer([color, aligned_depth, joint_state], 5, 1/50) #queue size, window secs
        ts.registerCallback(self.callback)

        self.tf_listener = tf.TransformListener()

        ## Robot model for fk
        urdf_file = ""

        self.robot_model = skrobot.models.urdf.RobotModelFromURDF(
            urdf_file=urdf_file
        )
        T_camera = None
        while T_camera is None:
            if self.sim:
                T_camera = self.lookup_tf_frame("camera_link", "panda_link8")
            else:
                T_camera = mercury.geometry.Coordinate([0.01831, 0.06372534, 0.02532344], [ 0.26376335, -0.65537354,  0.26756764,  0.65522666]).matrix
            print("Waiting for camera transform...")
        camera_pose = mercury.geometry.transformations.pose_from_matrix(T_camera)
        self.add_camera("camera_link", camera_pose)

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.pub = rp.Publisher(self.pose_topic, Pose, queue_size=10)
        self.pub_pointcloud = rp.Publisher("/imap/keyframe/points", PointCloud2, queue_size=10)
        self.pub_fext_record = rp.Publisher(f_ext_record, Bool, queue_size=10)
        self.imap_pause = False
        self.pub_imap_key = rp.Publisher("/imap/key", Int32, queue_size=10)
        self.pub_imap_label = rp.Publisher("/imap/label", Float32MultiArray, queue_size=10)
        self.pub_spec = rp.Publisher("/spec/capture", UInt16, queue_size=10)

        self.spectrometer = spectrometer.Spectrometer()
        self.spec_measurement = [None, rp.Time.now()]
        self.spec_measurement_t_prev = self.spec_measurement[1]
        self.imap_label = [-99, np.array([0.,0.,0.]), rp.Time.now()]
        self.imap_label_time_prev = self.imap_label[-1]
        self.imap_label_target_prev = None
        self.imap_key = [0, rp.Time.now()]
        self.imap_key_time_prev = self.imap_key[1]
        self.f_ext_record = False
        self.f_ext = [] ## Sliding window of force measurements length 10
        self.tip_z = []
        if not self.sim:
            f_ext_subscriber = rp.Subscriber(f_ext_topic, WrenchStamped, self.f_ext_callback)
            fext_record_subscriber = rp.Subscriber(f_ext_record, Bool, self.f_ext_record_callback)
            imap_key_subscriber = rp.Subscriber("/imap/key", Int32, self.imap_key_callback)
            imap_label_subscriber = rp.Subscriber("/imap/label", Float32MultiArray, self.imap_label_callback)
            sub_spec = rp.Subscriber("/spec/measurement", UInt16MultiArray, self.spec_callback)

    def pause_rosbag(self, pause):
        print("Pausing rosbag: ", pause)
        rp.wait_for_service('/rosbag_play/pause_playback')
        try:
            rosbag_pause = rp.ServiceProxy('/rosbag_play/pause_playback', SetBool)

            req = SetBoolRequest()
            req.data = pause

            res = rosbag_pause(req)
            return True
        except rp.ServiceException as e:
            print("Service call failed: %s"%e)

        return False

    def callback(self, color, aligned_depth, joint_state):
        self.color = color
        self.aligned_depth = aligned_depth
        self.joint_state = joint_state
        return

    def f_ext_callback(self, data):
        if self.f_ext_record:
            self.f_ext.append(data.wrench.force.z)
            self.tip_z.append(self.lookup_tf_frame("map", "panda_link8", lookup_time=data.header.stamp)[2,3])#[0,3]
        return

    def set_f_ext_record(self, record):
        self.f_ext_record = record

        if record:
            self.f_ext = []
            self.tip_z = []

        msg = Bool()
        msg.data = record
        self.pub_fext_record.publish(msg)
        return

    def f_ext_record_callback(self, data):
        self.f_ext_record = data.data
        return

    def imap_key_callback(self, data):
        self.imap_key = [data.data, rp.Time.now()]
        return

    def imap_label_callback(self, data):
        self.imap_label = [data.data[-1],
                        data.data[:3],
                        rp.Time.now()]
        return

    def get_imap_label(self):
        if self.imap_label[-1] == self.imap_label_time_prev:
            return None

        self.imap_label_time_prev = self.imap_label[-1]
        return self.imap_label[:2]

    def set_imap_key(self, key):
        msg = Int32()
        msg.data = key
        self.pub_imap_key.publish(msg)
        return

    def get_imap_key(self):
        if self.imap_key[1] == self.imap_key_time_prev:
            return None

        self.imap_key_time_prev = self.imap_key[1]
        return self.imap_key[0]

    def publish_label(self, label):
        msg = Float32MultiArray()
        msg.data = label
        self.pub_imap_label.publish(msg)
        return

    def spec_callback(self, data):
        self.spec_measurement = [data.data, rp.Time.now()]
        return

    def get_spec_measurement(self):
        if self.spec_measurement[1] == self.spec_measurement_t_prev:
            return None

        self.spec_measurement_t_prev = self.spec_measurement[1]
        return self.spec_measurement[0]

    def get_spec_class(self, mode=0):
        classes = []
        for i in range(0, 1): #3):
            classes.append(self.spectrometer.classify([], mode=mode))

        classes = np.array(classes)
        print("classes: ", classes)

        vals, counts = np.unique(classes, return_counts=True)
        mode_value = np.argwhere(counts == np.max(counts))

        mode = vals[mode_value].flatten().tolist()[0]
        return mode

    def undistort(self, depth_np, color_np):

        w=self.config["camera"]["w"]
        h=self.config["camera"]["h"]
        mw=self.config["camera"]["mw"]
        mh=self.config["camera"]["mh"]
        fx=self.config["camera"]["fx"]
        fy=self.config["camera"]["fy"]
        cx=self.config["camera"]["cx"]
        cy=self.config["camera"]["cy"]
        k1=self.config["camera"]["k1"]
        k2=self.config["camera"]["k2"]
        k3=self.config["camera"]["k3"]
        k4=self.config["camera"]["k4"]
        k5=self.config["camera"]["k5"]
        k6=self.config["camera"]["k6"]
        p1=self.config["camera"]["p1"]
        p2=self.config["camera"]["p2"]


        undistort = False
        if k1 != 0:
            undistort = True

        if not undistort:
            return depth_np, color_np

        K = np.array([[fx, 0., cx],
                    [0., fy, cy],
                    [0., 0., 1.]])

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            K,
            np.array([k1, k2, p1, p2, k3, k4, k5, k6]),
            np.eye(3),
            K,
            (w, h),
            cv2.CV_32FC1)

        # undistort
        depth_np = cv2.remap(depth_np, self.map1x, self.map1y,
                                cv2.INTER_NEAREST)
        color_np = cv2.remap(color_np, self.map1x, self.map1y,
                                cv2.INTER_LINEAR)

        depth_np = np.nan_to_num(depth_np)
        color_np = np.nan_to_num(color_np)

        # crop
        if mw > 0 or mh > 0:
            w = depth_np.shape[1]
            h = depth_np.shape[0]
            depth_np = depth_np[mh:(h - mh), mw:(w - mw)]
            color_np = color_np[mh:(h - mh), mw:(w - mw)]

        return depth_np, color_np[:, :, :3]

    def add_camera(self, name, pose):

        if self.sim:
            euler = [np.deg2rad(90.),np.deg2rad(90.),0.]
        else:
            euler = [np.deg2rad(-90.),np.deg2rad(90.),0.]

        c_pose = mercury.geometry.Coordinate(pose[0], pose[1])
        c_pose.rotate(euler)
        pose = c_pose.pose

        parent_name = "panda_link8"
        link_list = self.robot_model.link_list.copy()
        joint_list = self.robot_model.joint_list.copy()
        parent_link = getattr(self.robot_model, parent_name)
        link = skrobot.model.Link(
            parent=parent_link,
            pos=pose[0],
            rot=mercury.geometry.quaternion_matrix(pose[1])[:3, :3],
            name=name,
        )
        joint = skrobot.model.FixedJoint(
            child_link=link,
            parent_link=parent_link,
            name=f"{parent_name}_to_{name}_joint",
        )
        link.joint = joint
        link_list.append(link)
        joint_list.append(joint)
        self.robot_model = skrobot.model.RobotModel(
            link_list=link_list,
            joint_list=joint_list,
        )

    def update_robot_model(self, joint_state):
        for n_joint in range(0, len(joint_state.position)):
            joint_name = joint_state.name[n_joint]
            joint_position = joint_state.position[n_joint]
            getattr(self.robot_model, joint_name).joint_angle(joint_position)

    def get_camera_data(self, mw=0, mh=0):

        color_msg = self.color
        depth_msg = self.aligned_depth

        if color_msg is None:
            return None, None, None

        self.update_robot_model(self.joint_state)
        T_camera = self.robot_model.camera_link.worldcoords().T()

        rgb_img = np.asanyarray(self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')) #desired_encoding='rgb8'))
        depth_raw = np.asanyarray(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'))

        depth_raw, rgb_img = self.undistort(depth_raw, rgb_img)

        return depth_raw, rgb_img, T_camera

    def create_and_publish_pointcloud(self, pc, transform=None, rgb=None, depth=None, camera_pose=None, intrinsics=None):

        height = 1
        if depth is None:
            pcd_in_map = pc
            height = pcd_in_map.shape[1]
        else:
            pcd_in_camera = mercury.geometry.pointcloud_from_depth(
                depth, fx=intrinsics[0], fy=intrinsics[1], cx=intrinsics[2], cy=intrinsics[3]
            )
            pcd_in_camera = pcd_in_camera.reshape(-1,3)
            pcd_in_map = mercury.geometry.transform_points(
                pcd_in_camera,
                camera_pose,
            )

        if transform is not None:
            pcd_in_map = mercury.geometry.transform_points(
                pcd_in_map,
                transform,
            )

        msg = PointCloud2()
        msg.header.frame_id = "map"

        msg.height = height
        msg.width = pcd_in_map.shape[0]

        pcd_in_map = pcd_in_map.flatten()
        pcd_in_map = np.float32(pcd_in_map)

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

    def trans_quart_to_matrix(self, trans, quat):
        m = np.identity(4)
        m[0:3,3] = trans
        rot = Rotation.from_quat(quat)
        m[:3,:3] = rot.as_matrix()

        return m

    def matrix_to_trans_quart(self, matrix):
        trans = matrix[0:3,3]
        rot = Rotation.from_matrix(matrix[:3,:3])
        quat = rot.as_quat()

        return trans, quat

    def lookup_tf_frame(self, source_frame, target_frame, lookup_time=None, timeout=0.2, attempts=5):
        c = 0
        while c < int(attempts):
            try:
                t = rp.Time(0) if lookup_time is None else lookup_time

                self.tf_listener.waitForTransform(
                    target_frame=target_frame,
                    source_frame=source_frame,
                    time=t,
                    timeout=rp.Duration.from_sec(timeout))

                trans, quat = self.tf_listener.lookupTransform(
                    target_frame=target_frame,
                    source_frame=source_frame,
                    time=t)

                return self.trans_quart_to_matrix(trans=trans, quat=quat)

            except tf.Exception:
                d = rp.Duration(0.1)
                rp.sleep(d)
                c += 1

        m = 'Unable to lookup TF frame, {0} -> {1}'.format(source_frame, target_frame)
        print(m)
        return None

    def broadcast_tf_frame(self, matrix, child_frame, parent_frame):
        trans, quat = self.matrix_to_trans_quart(matrix)

        self.tf_broadcaster.sendTransform(
            translation=trans,
            rotation=quat,
            time=rp.Time.now(),
            child=child_frame,
            parent=parent_frame)

        return True

    def get_camera_pose(self):
        t = rp.Time(self.color.header.stamp.secs, self.color.header.stamp.nsecs)
        return self.lookup_tf_frame('camera_link', 'map', lookup_time=t)

    def publish_pose(self, **kwargs):
        pose = kwargs["pose"]

        pose_msg = Pose()
        pose_msg.position.x = pose[3,0]
        pose_msg.position.y = pose[3,1]
        pose_msg.position.z = pose[3,2]

        rot = Rotation.from_matrix(pose[:3,:3])
        pose_msg.orientation.x = rot.as_quat()[0]
        pose_msg.orientation.y = rot.as_quat()[1]
        pose_msg.orientation.z = rot.as_quat()[2]
        pose_msg.orientation.w = rot.as_quat()[3]

        self.pub.publish(pose_msg)
        return True

    def get_pose(self, **kwargs):
        try:
            pose_msg = rp.wait_for_message(self.pose_topic,
                        Pose,
                        timeout=rp.Duration.from_sec(0.5)
            )
        except Exception as e:
            print("Error: Unable to get pose.")
            return []

        pose = np.zeros((4,4))
        trans = np.array([pose_msg.position.x,
                    pose_msg.position.y,
                    pose_msg.position.z]
        )
        quat = np.array([pose_msg.orientation.x,
                    pose_msg.orientation.y,
                    pose_msg.orientation.z,
                    pose_msg.orientation.w]
        )

        return self.trans_quart_to_matrix(trans=trans, quat=quat)

    def get_joint_state(self, **kwargs):
        try:
            joint_stage_msg = rp.wait_for_message(self.joint_states_topic,
                        JointState,
                        timeout=rp.Duration.from_sec(0.5)
            )
        except Exception as e:
            print("Error: Unable to get joint_state.")
            return []

        return joint_stage_msg


def main():

    config_folder = ""
    config_file = ""

    with open(config_folder + "/" + config_file) as json_file:
        config = json.load(json_file)

    ros_bridge = RosBridge(init_node=True, sim=False, name="ros_bridge_test", config=config)

    rate = rp.Rate(1) # 10hz
    while not rp.is_shutdown():
        depth_np, im_np, pose_np = ros_bridge.get_camera_data()

        rate.sleep()

    print("End ros_bridge test.")


if __name__ == "__main__":
    main()
