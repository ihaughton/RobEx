#!/usr/bin/env python
import cv2
import imgviz
import numpy as np

import os.path

if __name__ == "__main__":
    save = False
    show = True
    path = ("")
    num_files = len([f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))])
    n_ims = int(num_files / 2)

    for i in range(n_ims):
        s = f"{i:06}"  # int variable
        depth_file = path + "depth" + s + ".png"
        depth = cv2.imread(depth_file, -1)
        depth = depth.astype(np.float32)
        depth /= 10000.0
        depth_viz = imgviz.depth2rgb(depth, min_value=0.2, max_value=4)

        rgb_file = path + "frame" + s + ".jpg"
        rgb = cv2.imread(rgb_file)
        viz = np.hstack([rgb, depth_viz])
        viz = imgviz.resize(viz, width=1600)

        if show:
            cv2.imshow("viz", viz)
            cv2.waitKey(1)

        if save:
            cv2.imwrite("results/tile_" + s + ".jpg", viz)
