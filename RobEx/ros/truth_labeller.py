#!/usr/bin/env python3

import rospy as rp
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import UInt16
from std_msgs.msg import String

class RosBridge():

    def __init__(self, **kwargs):

        rp.init_node("truth_labeller")
        self.pub = rp.Publisher("/spec/ground_truth", String, queue_size=10)

def main():

    ros_bridge = RosBridge()

    exit = False
    while not exit:
        input_ = input("Enter truth: ")

        truth_msg = String()
        truth_msg.data = input_

        ros_bridge.pub.publish(truth_msg)


if __name__ == "__main__":
    main()
