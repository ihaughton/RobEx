import numpy as np
import cv2

def determine_edge_mask(rgb_img):
    L2gradient = False
    apertureSize = 3
    vmin = 50
    vmax = 150

    edges_img_r = cv2.Canny(rgb_img[:,:,0],vmin,vmax, apertureSize=apertureSize, L2gradient=L2gradient)
    edges_img_g = cv2.Canny(rgb_img[:,:,1],vmin,vmax, apertureSize=apertureSize, L2gradient=L2gradient)
    edges_img_b = cv2.Canny(rgb_img[:,:,2],vmin,vmax, apertureSize=apertureSize, L2gradient=L2gradient)

    edges_img = np.logical_or(edges_img_r == 255, edges_img_g == 255)
    edges_img = np.logical_or(edges_img, edges_img_b == 255)
    edges_img = (edges_img * 255).round().astype(np.uint8)

    ## dilate lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges_img = cv2.dilate(edges_img, kernel, iterations=1)

    return edges_img == 255
