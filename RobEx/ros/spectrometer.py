#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import joblib

import rospy as rp
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import UInt16
from std_msgs.msg import String

sns.set_theme(style="ticks", palette="pastel")

DATA_DIR = ""
SCALER_FILE = "model/scaler.sav"
MODEL_FILE = "model/svc_model.sav"

class RosBridge():

    def __init__(self, **kwargs):

        self.measurement = [None, rp.Time.now()]
        self.measurement_t_prev = self.measurement[1]
        self.pub = rp.Publisher("/spec/capture", UInt16, queue_size=10)

        truth_sub = rp.Subscriber("/spec/ground_truth", String, self.callback)

    def callback(self, data):
        self.measurement = [data.data, rp.Time.now()]
        return

    def get_measurement(self):
        if self.measurement[1] == self.measurement_t_prev:
            return None

        self.measurement_t_prev = self.measurement[1]
        return self.measurement[0]

class Spectrometer():

    def __init__(self, **kwargs):
        self.ros_bridge = RosBridge()

        ## Load scaler and model
        self.scaler = joblib.load(DATA_DIR + SCALER_FILE)
        self.svc_model = joblib.load(DATA_DIR + MODEL_FILE)

        self.material = {"wood": 0,
                        "laminate": 1,
                        "card": 2,
                        "cotton": 3,
                        "felt": 4,
                        "polypropylene": 5,
                        "polythene": 6,
                        "polyethylene": 7,
                        "synthetic": 1,
                        "lycra": 1,
                        "wool": 2,
        }


    def classify(self, meas, mode=0):

        if mode == 0:
            print("meas: ", meas)
            meas = self.scaler.transform([meas])
            pred = self.svc_model.predict(meas)
            print("Predict material: ", pred)
            return self.material[pred[0]]

        if mode == 1:
            ## Get ground truth
            truth = None
            while truth is None:
                truth = self.ros_bridge.get_measurement()

            truth = truth.split()
            if truth[0] == 'w':
                return self.material["wood"]
            if truth[0] == 'l':
                return self.material["laminate"]
            if truth[0] == 'c':
                return self.material["card"]
            if truth[0] == 'cn':
                return self.material["cotton"]
            if truth[0] == 'f':
                return self.material["felt"]
            if truth[0] == 'pp':
                return self.material["polypropylene"]
            if truth[0] == 'pt':
                return self.material["polythene"]

            ## fabrics
            if truth[0] == 'sy':
                return self.material["synthetic"]
            if truth[0] == 'wl':
                return self.material["wool"]

        if mode == 2:
            ## Get ground truth
            truth = None
            while truth is None:
                truth = self.ros_bridge.get_measurement()

            truth = truth.split()
            if truth[1] == 'h':
                return 1
            if truth[1] == 's':
                return 2

        return self.material[pred[0]]


def main():
    spectrometer = Spectrometer()

    exit = False
    while not exit:
        input_ = input("Enter p -- predict and e -- exit: ")

        if input_ == "p":

            class_ = spectrometer.classify()
            print("Predict class: ", class_)

        if input_ == "e":
            print("Exiting...")
            exit = True

    print("Measurements complete.")

if __name__ == "__main__":
    main()
