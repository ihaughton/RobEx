import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QObject
import threading
from queue import Queue
import pyrealsense2 as rs
import pyk4a
from pyk4a import Config, PyK4A
import numpy as np
import cv2
import pickle
import struct
import socket
from typing import Any
from nptyping import NDArray, UInt8
import pdb

RUNNING = False
NUM_CLASSES = 12
SENSOR_TYPE = 'Realsense'
N_POINTS_PER_SCRIBBLE = 10
UNDISTORT = False

current_keyframe = None
request_new_keyframe = False
keyframe_idx = -1
save_mesh_flag = False
client_closing_flag = False

pg.setConfigOptions(imageAxisOrder='row-major')

# set colour pallette. Same as iMAP
colourmap = np.array([[255, 0, 0],
                      [0, 255, 0],
                      [0, 0, 255],
                      [255, 255, 0],
                      [0, 255, 255],
                      [255, 0, 255],
                      [255, 128, 0],
                      [101, 67, 33],
                      [230, 0, 126],
                      [120, 120, 120],
                      [235, 150, 135],
                      [0, 128, 0]],
                     dtype=np.uint8)

# Mouse scribble parameters
mouse_callback_params = {}
mouse_callback_params["indices_w"] = []
mouse_callback_params["indices_h"] = []
mouse_callback_params["h_labels"] = []
mouse_callback_params["h_masks"] = []
mouse_callback_params["classes"] = []
mouse_callback_params["class"] = 0
mouse_callback_params["drawing"] = False
mouse_callback_params["new_scribble"] = False
mouse_callback_params["request_new_kf"] = False
mouse_callback_params["n_points_per_scribble"] = N_POINTS_PER_SCRIBBLE  # Number of points to sample from each scribble

scribble_params = {}
# Scribble parameters
scribble_params["indices_w"] = []
scribble_params["indices_h"] = []
scribble_params["class"] = []

def visualise_segmentation(im: NDArray[(Any, ...), Any], masks: NDArray[(Any, ...), Any], nc: int = None) -> NDArray[(Any, ...), UInt8]:
    """
    Visualize segmentations nicely. Based on code from:
    https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

    :param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
    :param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., nc-1}
    :param nc: total number of colors. If None, this will be inferred by masks

    :return: a [H x W x 3] numpy array of dtype np.uint8
    """
    global colourmap, NUM_CLASSES

    colors = colourmap[:NUM_CLASSES, [2, 0, 1]]
    masks = masks.astype(int)
    im = im.copy()

    # Mask
    imgMask = np.zeros(im.shape)

    # Draw color masks
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)

    return im


class ClientWorkerThread(QObject):
    global SENSOR_TYPE, UNDISTORT
    changeKeyframePixmap = pyqtSignal(QtGui.QImage)
    changeCurrentFramePixmap = pyqtSignal(QtGui.QImage)

    camera_type = SENSOR_TYPE
    if camera_type == 'Realsense':
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices()
        dev = devices[0]

        # Clamp max depth
        advanced_mode = rs.rs400_advanced_mode(dev)
        depth_table = advanced_mode.get_depth_table()
        depth_table.depthClampMax = 1500 #1.5m
        advanced_mode.set_depth_table(depth_table)

        # Create a pipeline
        pipeline = rs.pipeline()
        align = rs.align(rs.stream.color)
        # Create a config and configure the pipeline to stream
        config = rs.config()
        config.enable_device(dev.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # NB bgr so needs to be converted
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

        im_width = 640
        im_height = 480

        # Start streaming
        profile = pipeline.start(config)

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    elif camera_type == 'Azure':
        k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_720P,
                           depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
                           camera_fps=pyk4a.FPS.FPS_15,
                           synchronized_images_only=True, ))

        k4a.start()

        # getters and setters directly get and set on device
        im_width = 1280
        im_height = 720
        k4a.exposure = 5000
        # getters and setters directly get and set on device
        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500
        k4a.whitebalance = 4510
        assert k4a.whitebalance == 4510
        k4a.backlight_compensation = 1

        # Remapping and cropping parameters
        fx = 610.274414
        fy = 610.202942
        cx = 639.089478
        cy = 367.630157
        mw = 0
        mh = 0
        k1 = 0.657326
        k2 = -2.806669
        k3 = 1.560503
        k4 = 0.530181
        k5 = -2.625473
        k6 = 1.487417
        p1 = 0.000529
        p2 = -0.000321


        if UNDISTORT:
            K = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.]])

            map1x, map1y = cv2.initUndistortRectifyMap(
                K,
                np.array([k1, k2, p1, p2, k3, k4, k5, k6]),
                np.eye(3),
                K,
                (im_width, im_height),
                cv2.CV_32FC1)

    # set up Azure queue
    rgbd_queue = Queue(maxsize=1)

    # set up socket
    port = 8485
    # host = '10.48.83.15'
    host = '127.0.0.1'

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    server_address = (host, port)
    client_socket.connect(server_address)
    connection = client_socket.makefile('wb')

    current_keyframe = None
    keyframe_count = -1

    def stream_frames(self):
        """

        :return:
        """
        global RUNNING

        # set up threads
        publish_thread = threading.Thread(target=self.data_publish_thread, daemon=True)
        receive_thread = threading.Thread(target=self.data_receive_thread, daemon=True)

        publish_thread.start()
        receive_thread.start()

        # Populate the queue
        frame_count = 0
        while RUNNING:
            if self.rgbd_queue.full():
                continue

            if self.camera_type == 'Realsense':
                frames = self.pipeline.wait_for_frames()
                rgb = np.asanyarray(frames.get_color_frame().get_data())
                depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
                # depth = depth_raw * self.depth_scale

                if frame_count < 50:
                    frame_count += 1
                    continue


            elif self.camera_type == 'Azure':
                capture = self.k4a.get_capture()
                if np.any(capture.color) and np.any(capture.depth):
                    rgb = capture.color
                    depth = capture.transformed_depth
                    # undistort
                    if UNDISTORT:
                        depth = cv2.remap(depth, self.map1x, self.map1y,
                                             cv2.INTER_NEAREST)
                        rgb = cv2.remap(rgb, self.map1x, self.map1y,
                                             cv2.INTER_LINEAR)

                    # crop
                    if self.mh > 0 or self.mw > 0:
                        w = depth.shape[1]
                        h = depth.shape[0]
                        depth = depth[self.mh:(h - self.mh), self.mw:(w - self.mw)]
                        rgb = rgb[self.mh:(h - self.mh), self.mw:(w - self.mw)]

                    rgb = rgb[:, :, :3]

            images = np.concatenate((rgb, depth[:, :, np.newaxis]), axis=-1)
            self.rgbd_queue.put(images)

        # shut down threads
        publish_thread.join()
        receive_thread.join()
        self.client_socket.close()

    def data_publish_thread(self):
        """
        Writes data socket

        :return:
        """
        global RUNNING, mouse_callback_params, request_new_keyframe, keyframe_idx, save_mesh_flag, client_closing_flag

        while RUNNING:
            try:
                if self.rgbd_queue.empty():
                    continue

                publish_data = {}

                publish_data['client_closing_flag'] = client_closing_flag

                if client_closing_flag:
                    data = pickle.dumps(publish_data, 0)
                    data_size = len(data)
                    self.client_socket.sendall(struct.pack(">L", data_size) + data)
                    continue

                # Send esc signal to server and shutdown
                publish_data['save_mesh'] = save_mesh_flag

                images = self.rgbd_queue.get()
                # your code here
                images, buffer = cv2.imencode('.png', images)
                publish_data['rgbd_data'] = buffer
                publish_data['new_annotation'] = False
                publish_data['request_new_kf'] = request_new_keyframe
                publish_data['keyframe_idx'] = keyframe_idx

                #reset keyframe request
                request_new_keyframe = False

                # Check if new scribble available
                if mouse_callback_params['new_scribble']:
                    publish_data['new_annotation'] = True
                    # Publish scribble data and corresponding keyframe
                    publish_data['mouse_callback_params'] = mouse_callback_params

                    # Remove scribble to allow for new annotation
                    mouse_callback_params['new_scribble'] = False

                data = pickle.dumps(publish_data, 0)
                data_size = len(data)
                self.client_socket.sendall(struct.pack(">L", data_size) + data)

                # Only reset if flag was set in time
                if publish_data['save_mesh']:
                    save_mesh_flag = False

            except Exception as e:
                print(f"Client terminating with {e}")
                RUNNING = False
                break


    def data_receive_thread(self):
        """
        Receives response from server

        :return:
        """
        global RUNNING, mouse_callback_params, current_keyframe, keyframe_idx

        while RUNNING:
            try:

                data_back = b""
                payload_size = struct.calcsize(">L")

                # wait for response
                while len(data_back) < payload_size:
                    data_back += self.client_socket.recv(4096)

                packed_msg_size = data_back[:payload_size]
                data_back = data_back[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]

                while len(data_back) < msg_size:
                    data_back += self.client_socket.recv(4096)
                frame_data = data_back[:msg_size]

                data = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

                vis = np.asarray(data['seg_vis'])
                if data['keyframe'] is not None:
                    keyframe = np.asarray(data['keyframe'])
                    keyframe = cv2.imdecode(keyframe, cv2.IMREAD_COLOR)
                    keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2BGR)
                    # Check if new keyframe
                    change_keyframe = False
                    if current_keyframe is None:
                        change_keyframe = True
                        current_keyframe = np.copy(keyframe)
                    else:
                        if not np.allclose(keyframe, current_keyframe):
                            change_keyframe = True
                            current_keyframe = np.copy(keyframe)

                    # emit keyframe
                    if change_keyframe:
                        qtKeyframe = QtGui.QImage(keyframe.data, keyframe.shape[1], keyframe.shape[0], keyframe.strides[0],
                                                  QtGui.QImage.Format_RGB888)
                        keyframe_emit = qtKeyframe.scaled(self.im_width, self.im_height, Qt.KeepAspectRatio)
                        self.changeKeyframePixmap.emit(keyframe_emit)
                        keyframe_idx += 1
                        print("Keyframe changed")

                    self.current_keyframe = np.copy(keyframe)

                vis = cv2.imdecode(vis, cv2.IMREAD_COLOR)

                # emit visualisation
                qtCurrentframe = QtGui.QImage(vis.data, vis.shape[1], vis.shape[0], vis.strides[0],
                                       QtGui.QImage.Format_RGB888)
                current_frame_emit = qtCurrentframe.scaled(self.im_width, self.im_height, Qt.KeepAspectRatio)
                self.changeCurrentFramePixmap.emit(current_frame_emit)


            except Exception as e:
                print(f"Client terminating with {e}")
                RUNNING = False
                break


# GUI
class App(QtGui.QDialog):

    def __init__(self):
        global NUM_CLASSES, SENSOR_TYPE, colourmap
        super(App, self).__init__()
        if SENSOR_TYPE == 'Realsense':
            self.window_size = (640, 480)
        else:
            self.window_size = (1280, 720)
        self.initUI()
        self.setup_thread()
        self.colourmap = colourmap[:NUM_CLASSES, :]

    def initUI(self):
        self.originalPalette = QtGui.QApplication.palette()
        self.main_tab = QtGui.QWidget()
        self.main_tab.layout = QtGui.QHBoxLayout()

        # add class buttons
        buttons_layout = QtGui.QHBoxLayout()
        self.class_one_button = QtGui.QPushButton('1')
        self.class_one_button.setStyleSheet("background-color:rgb(255, 0, 0); border: native")
        buttons_layout.addWidget(self.class_one_button)

        self.class_two_button = QtGui.QPushButton('2')
        self.class_two_button.setStyleSheet("background-color:rgb(0, 255, 0); border: none")
        buttons_layout.addWidget(self.class_two_button)

        self.class_three_button = QtGui.QPushButton('3')
        self.class_three_button.setStyleSheet("background-color:rgb(0, 0, 255); border: none")
        buttons_layout.addWidget(self.class_three_button)

        self.class_four_button = QtGui.QPushButton('4')
        self.class_four_button.setStyleSheet("background-color:rgb(255, 255, 0); border: none")
        buttons_layout.addWidget(self.class_four_button)

        self.class_five_button = QtGui.QPushButton('5')
        self.class_five_button.setStyleSheet("background-color:rgb(0, 255, 255); border: none")
        buttons_layout.addWidget(self.class_five_button)

        # add nearest keyframe button
        misc_buttons_layout = QtGui.QVBoxLayout()
        self.keyframe_button = QtGui.QPushButton('Next KF')
        misc_buttons_layout.addWidget(self.keyframe_button, alignment=QtCore.Qt.AlignBottom)

        self.save_button = QtGui.QPushButton('Save Mesh')
        misc_buttons_layout.addWidget(self.save_button, alignment=QtCore.Qt.AlignTop)

        # image layout: rgb image and keyframe image
        self.image_widget = pg.GraphicsLayoutWidget()
        self.keyframe_widget = pg.GraphicsLayoutWidget()
        self.current_view_box = self.image_widget.addViewBox(lockAspect=True)
        self.current_view_box.invertY(True)
        self.current_view_box.setMouseEnabled(x=False, y=False)
        self.current_view_image = pg.ImageItem(np.zeros([self.window_size[0], self.window_size[1], 3], dtype=np.int32))
        self.current_view_box.addItem(self.current_view_image)
        self.keyframe_view_box = self.keyframe_widget.addViewBox(lockAspect=True)
        self.keyframe_view_box.invertY(True)
        self.keyframe_view_box.setMouseEnabled(x=False, y=False)
        self.keyframe_image = pg.ImageItem(np.zeros([self.window_size[0], self.window_size[1], 3], dtype=np.int32))
        self.keyframe_view_box.addItem(self.keyframe_image)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addLayout(buttons_layout, 0, 1)
        mainLayout.addWidget(self.image_widget, 1, 0)
        mainLayout.addWidget(self.keyframe_widget, 1, 1)
        mainLayout.addLayout(misc_buttons_layout, 1, 2)
        self.setLayout(mainLayout)

        self.show()

    def mouse_press_event(self, event):
        global mouse_callback_params, scribble_params

        drawing = mouse_callback_params["drawing"]
        scribble_exists = mouse_callback_params["new_scribble"]

        # Don't allow new annotation until previous one has been sent to server
        if event.button() == Qt.LeftButton and not drawing and not scribble_exists:
            mouse_callback_params['drawing'] = True
            mouse_callback_params['new_scribble'] = False
            pos = event.pos()
            pos = self.keyframe_image.mapFromScene(pos)
            scribble_params["indices_h"].append(int(pos.y()))
            scribble_params["indices_w"].append(int(pos.x()))

    def mouse_move_event(self, event):
        global mouse_callback_params, scribble_params
        if (event.buttons() and Qt.LeftButton) and mouse_callback_params['drawing']:
            sem_class = mouse_callback_params["class"]
            colour = self.colourmap[sem_class]
            keyframe_image = np.copy(self.keyframe_image.image)
            pos = event.pos()
            pos = self.keyframe_image.mapFromScene(pos)
            cv2.line(keyframe_image, (scribble_params["indices_w"][-1], scribble_params["indices_h"][-1]), (int(pos.x()), int(pos.y())), colour.astype(np.uint8).tolist(), 5)
            # update position

            scribble_params["indices_h"].append(int(pos.y()))
            scribble_params["indices_w"].append(int(pos.x()))
            qt_kf = QtGui.QImage(keyframe_image.data, keyframe_image.shape[1], keyframe_image.shape[0],
                                   keyframe_image.strides[0], QtGui.QImage.Format_RGB888)

            kf_emit = qt_kf.scaled(self.window_size[0], self.window_size[1], Qt.KeepAspectRatio)
            np_image = self.QImageToNP(kf_emit)
            self.keyframe_image.setImage(np_image)

    def mouse_release_event(self, event):
        global mouse_callback_params, scribble_params
        if mouse_callback_params['drawing']:
            mouse_callback_params['drawing'] = False
            mouse_callback_params['new_scribble'] = True

            scribble_length = len(scribble_params["indices_h"])
            if scribble_length > mouse_callback_params["n_points_per_scribble"]:
                sample_idx = np.linspace(0, scribble_length - 1, mouse_callback_params["n_points_per_scribble"], dtype=int)
                mouse_callback_params["indices_h"].extend([scribble_params["indices_h"][i] for i in sample_idx[:mouse_callback_params["n_points_per_scribble"]]])
                mouse_callback_params["indices_w"].extend([scribble_params["indices_w"][i] for i in sample_idx[:mouse_callback_params["n_points_per_scribble"]]])
                mouse_callback_params["classes"].extend(mouse_callback_params["n_points_per_scribble"] * [mouse_callback_params["class"]])
            else:
                # Otherwsie use all points in scribble
                mouse_callback_params["indices_h"].extend(scribble_params["indices_h"])
                mouse_callback_params["indices_w"].extend(scribble_params["indices_w"])
                mouse_callback_params["classes"].extend(scribble_length * [mouse_callback_params["class"]])

            # reset temp scribble params
            scribble_params["indices_w"] = []
            scribble_params["indices_h"] = []
            mouse_callback_params["drawing"] = False

    def QImageToNP(self,image):
        # Convert a QImage to a numpy array
        image = image.convertToFormat(QtGui.QImage.Format_RGB888)
        width = image.width()
        height = image.height()
        ptr = image.constBits()
        return np.frombuffer(ptr.asstring(image.byteCount()), dtype=np.uint8).reshape(height, width, 3)

    @pyqtSlot(QtGui.QImage)
    def set_current_image(self, image):
        #self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
        np_image = self.QImageToNP(image)
        self.current_view_image.setImage(np_image)

    @pyqtSlot(QtGui.QImage)
    def set_keyframe(self, image):
        #self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
        np_image = self.QImageToNP(image)
        self.keyframe_image.setImage(np_image)

    @pyqtSlot()
    def request_new_keyframe(self):
        global request_new_keyframe

        request_new_keyframe = True

    @pyqtSlot()
    def save_mesh(self):
        global save_mesh_flag
        save_mesh_flag = True

    @pyqtSlot()
    def set_class(self, idx: int):
        """
        Sets the class index.

        :param class_idx:
        :return:
        """
        global mouse_callback_params, NUM_CLASSES

        if idx == mouse_callback_params["class"]:
            return

        assert (idx < NUM_CLASSES)

        previous_class = mouse_callback_params["class"]

        mouse_callback_params["class"] = idx

        # Reset appearance of previous class
        if previous_class == 0:
            self.class_one_button.setStyleSheet("background-color:rgb(255, 0, 0); border: none")
        if previous_class == 1:
            self.class_two_button.setStyleSheet("background-color:rgb(0, 255, 0); border: none")
        if previous_class == 2:
            self.class_three_button.setStyleSheet("background-color:rgb(0, 0, 255); border: none")
        if previous_class == 3:
            self.class_four_button.setStyleSheet("background-color:rgb(255, 255, 0); border: none")
        if previous_class == 4:
            self.class_five_button.setStyleSheet("background-color:rgb(0, 255, 255); border: none")

        # highlight button of current class
        if idx == 0:
            self.class_one_button.setStyleSheet("background-color:rgb(255, 0, 0); border: native")
        if idx == 1:
            self.class_two_button.setStyleSheet("background-color:rgb(0, 255, 0); border: native")
        if idx == 2:
            self.class_three_button.setStyleSheet("background-color:rgb(0, 0, 255); border: native")
        if idx == 3:
            self.class_four_button.setStyleSheet("background-color:rgb(255, 255, 0); border: native")
        if idx == 4:
            self.class_five_button.setStyleSheet("background-color:rgb(0, 255, 255); border: native")

    def keyPressEvent(self, event):
        global client_closing_flag, RUNNING
        if event.key() == QtCore.Qt.Key_Q:
            print("Closing client")
            client_closing_flag = True
            # send exit flag
        event.accept()

        while RUNNING:
            continue
        self.close_client()

    def setup_thread(self):
        global RUNNING
        RUNNING = True
        self.thread = QThread()
        self.worker = ClientWorkerThread()

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.stream_frames)
        self.worker.changeCurrentFramePixmap.connect(self.set_current_image)
        self.worker.changeKeyframePixmap.connect(self.set_keyframe)
        self.keyframe_widget.mousePressEvent = self.mouse_press_event
        self.keyframe_widget.mouseMoveEvent = self.mouse_move_event
        self.keyframe_widget.mouseReleaseEvent = self.mouse_release_event

        # Connect buttons
        self.class_one_button.clicked.connect(lambda: self.set_class(0))
        self.class_two_button.clicked.connect(lambda: self.set_class(1))
        self.class_three_button.clicked.connect(lambda: self.set_class(2))
        self.class_four_button.clicked.connect(lambda: self.set_class(3))
        self.class_five_button.clicked.connect(lambda: self.set_class(4))

        self.keyframe_button.clicked.connect(self.request_new_keyframe)
        self.save_button.clicked.connect(self.save_mesh)

        self.thread.start()

    def close_client(self):
        print("Closing client")
        self.close()


if __name__ == '__main__':
    import pdb
    app = QtGui.QApplication(sys.argv)
    ex = App()
    # ex.show()
    sys.exit(app.exec_())
