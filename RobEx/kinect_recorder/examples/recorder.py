#!/usr/bin/env python
import os
import argparse
import open3d as o3d
import numpy as np
import cv2
import imgviz
import json

from RobEx.kinect_recorder import reader

class ViewerWithCallback:

    def __init__(self, config, device, output_dir):

        # create output dir if it doesn't exist
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        self.output_dir = output_dir

        self.flag_exit = False
        self.flag_save = False
        self.save_counter = 0

        config_file = config["kinect"]["config_file"]
        kinect_config = o3d.io.read_azure_kinect_sensor_config(config_file)
        self.kinect_reader = reader.KinectReaderLight(
            kinect_config, device,
            w=config["camera"]["w"],
            h=config["camera"]["h"],
            fx=config["camera"]["fx"],
            fy=config["camera"]["fy"],
            cx=config["camera"]["cx"],
            cy=config["camera"]["cy"],
            k1=config["camera"]["k1"],
            k2=config["camera"]["k2"],
            k3=config["camera"]["k3"],
            k4=config["camera"]["k4"],
            k5=config["camera"]["k5"],
            k6=config["camera"]["k6"],
            p1=config["camera"]["p1"],
            p2=config["camera"]["p2"]
        )
        self.mh = config["camera"]["mh"]
        self.mw = config["camera"]["mw"]

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def run(self):
        print("Sensor initialized. Press [ESC] to exit.")
        print("Press s key to start recording.")

        # vis_geometry_added = False
        while not self.flag_exit:
            data = self.kinect_reader.get(
                mw=self.mw, mh=self.mh)
            if data is None:
                continue
            depth_np, color_np = data

            if self.flag_save:
                cv2.imwrite(self.output_dir + "/depth" +
                            f'{self.save_counter:06}' + ".png", depth_np)
                cv2.imwrite(self.output_dir + "/frame" +
                            f'{self.save_counter:06}' + ".jpg", color_np)
                self.save_counter += 1

            depth_vis = imgviz.depth2rgb(depth_np.astype(
                np.float) / 1000., min_value=0., max_value=3.)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

            vis = np.hstack([color_np, depth_vis])

            vis = imgviz.resize(vis, width=2000)
            cv2.imshow("vis", vis)
            key = cv2.waitKey(1)
            if key == 115:
                print("Start saving...")
                self.flag_save = True

            if key == 27:
                self.flag_exit = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
    parser.add_argument('--config', type=str, help='input config')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('--output_dir',
                        type=str,
                        default="./results",
                        help='Directory where results will be saved')

    args = parser.parse_args()

    with open(args.config) as json_file:
        config_file = json.load(json_file)

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    v = ViewerWithCallback(config_file, device, args.output_dir)
    v.run()
