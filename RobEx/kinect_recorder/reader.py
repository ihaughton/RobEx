import cv2
import open3d as o3d
import numpy as np
import timeit
import pyk4a
from pyk4a import Config, PyK4A
import socket
import struct
import pickle
import pyrealsense2 as rs

class KinectReaderLight:

    def __init__(self, config, device,
                 w, h, fx, fy, cx, cy,
                 k1, k2, k3, k4, k5, k6, p1, p2,
                 align_depth_to_color=True,
                 undistort=True
                 ):
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
                camera_fps=pyk4a.FPS.FPS_30,
                synchronized_images_only=True,
            )
        )
        self.k4a.start()
        self.k4a.exposure = 9000

        self.undistort = undistort
        if undistort:
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

    def get(self, mw=0, mh=0, as_bgr=True, undistort=True):
        capture = self.k4a.get_capture()
        if np.any(capture.color) and np.any(capture.depth):
            color_np = capture.color
            depth_np = capture.transformed_depth

        else:
            return None

        if self.undistort:
            # undistort
            depth_np = cv2.remap(depth_np, self.map1x, self.map1y,
                                 cv2.INTER_NEAREST)
            color_np = cv2.remap(color_np, self.map1x, self.map1y,
                                 cv2.INTER_LINEAR)

        # crop
        if mw > 0 or mh > 0:
            w = depth_np.shape[1]
            h = depth_np.shape[0]
            depth_np = depth_np[mh:(h - mh), mw:(w - mw)]
            color_np = color_np[mh:(h - mh), mw:(w - mw)]

        return depth_np, color_np[:, :, :3]


class KinectReader:
    def __init__(self, config, device,
                 w, h, fx, fy, cx, cy,
                 k1, k2, k3, k4, k5, k6, p1, p2,
                 align_depth_to_color=True
                 ):
        self.align_depth_to_color = align_depth_to_color

        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')

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

    def get(self, mw=0, mh=0, as_bgr=True):
        start_time = timeit.default_timer()
        rgbd = self.sensor.capture_frame(self.align_depth_to_color)
        if rgbd is None:
            return rgbd
        elapsed = timeit.default_timer() - start_time
        print("capture", elapsed * 1000)

        depth_np = np.asarray(rgbd.depth)
        color_np = np.asarray(rgbd.color)
        elapsed = timeit.default_timer() - start_time

        if as_bgr:
            color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

        start_time = timeit.default_timer()
        # undistort
        depth_np = cv2.remap(depth_np, self.map1x, self.map1y,
                             cv2.INTER_NEAREST)
        color_np = cv2.remap(color_np, self.map1x, self.map1y,
                             cv2.INTER_LINEAR)
        elapsed = timeit.default_timer() - start_time
        print("remap ", elapsed * 1000)

        # crop
        w = depth_np.shape[1]
        h = depth_np.shape[0]
        depth_np = depth_np[mh:(h - mh), mw:(w - mw)]
        color_np = color_np[mh:(h - mh), mw:(w - mw)]

        return depth_np, color_np


class DataReaderServer:
    def __init__(self,
                 w, h, fx, fy, cx, cy,
                 k1, k2, k3, k4, k5, k6, p1, p2,
                 align_depth_to_color=True,
                 port=8485
                 ):

        self.align_depth_to_color = align_depth_to_color

        # set up port
        self.host = ''
        self.port = port

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        print('Socket created')

        self.s.bind((self.host, self.port))
        print('Data capture socket bind complete')
        self.s.listen(10)
        print('Data capture socket now listening')

        self.conn, self.addr = self.s.accept()
        self.data = b""
        self.payload_size = struct.calcsize(">L")

        self.width = w
        self.height = h
        self.client_has_closed = False

    def get(self, mw=0, mh=0):
        """

        """

        try:
            while len(self.data) < self.payload_size:
                self.data += self.conn.recv(4096)

            packed_msg_size = self.data[:self.payload_size]
            self.data = self.data[self.payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(self.data) < msg_size:
                self.data += self.conn.recv(4096)

            frame_data = self.data[:msg_size]
            self.data = self.data[msg_size:]

            data = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            client_closing_flag = data['client_closing_flag']

            if client_closing_flag:
                return None

            save_mesh_flag = data['save_mesh']

            frame = data['rgbd_data']
            frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
            color_np = frame[:, :, :3]
            color_np = color_np.astype(np.uint8)
            depth_np = frame[:, :, -1].squeeze()

            # crop
            if mw > 0 or mh > 0:
                w = depth_np.shape[1]
                h = depth_np.shape[0]
                depth_np = depth_np[mh:(h - mh), mw:(w - mw)]
                color_np = color_np[mh:(h - mh), mw:(w - mw)]

            # Keyframe info
            request_new_keyframe = data['request_new_kf']
            keyframe_idx = data['keyframe_idx']

            # Check for annotation
            annotation_data = None
            new_annotation = data['new_annotation']
            if new_annotation:
                annotation_data = data['mouse_callback_params']

            return depth_np, color_np, annotation_data, keyframe_idx, request_new_keyframe, save_mesh_flag

        except Exception as e:
            print(f"Client has closed. Terminating read. {e}")
            return None

    def send(self, image):
        """
        """

        # populate queue
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        rgb = np.copy(image[:, :, :3])
        if image.shape[-1] > 3:
            label_kf = np.copy(image[:, :, 3:])
            images, kf_buffer = cv2.imencode('.jpg', label_kf, params=encode_param)
        else:
            kf_buffer = None

        images, vis_buffer = cv2.imencode('.jpg', rgb, params=encode_param)

        # create send object and pickle
        send_object = dict()
        send_object['seg_vis'] = vis_buffer
        send_object['keyframe'] = kf_buffer
        send_object['save_mesh'] = False

        data = pickle.dumps(send_object, 0)

        data_size = len(data)
        try:
            self.conn.sendall(struct.pack(">L", data_size) + data)
            return True
        except:
            print("Client has closed. Terminating send.")
            self.client_has_closed = True
            return False


class RealsenseReaderLight:

    def __init__(self,
                 w, h,
                 align_depth_to_color=True
                 ):

        self.align = align_depth_to_color
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices()
        dev = devices[0]

        # clamp max depth
        advanced_mode = rs.rs400_advanced_mode(dev)
        depth_table = advanced_mode.get_depth_table()
        depth_table.depthClampMax = 1500 #1.5m
        advanced_mode.set_depth_table(depth_table)

        # Create a pipeline
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        # Create a config and configure the pipeline to stream
        config = rs.config()
        config.enable_device(dev.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # NB bgr so needs to be converted
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

        # Start streaming
        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        depth_profile = profile.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics

        K = np.array([[intrinsics.fx, 0., intrinsics.ppx],
                      [0., intrinsics.fy, intrinsics.ppy],
                      [0., 0., 1.]])

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            K,
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            np.eye(3),
            K,
            (w, h),
            cv2.CV_32FC1)

    def get(self, mw=0, mh=0):
        frames = self.pipeline.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)
        rgb_img = np.asanyarray(frames.get_color_frame().get_data())
        depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
        return depth_raw, rgb_img
