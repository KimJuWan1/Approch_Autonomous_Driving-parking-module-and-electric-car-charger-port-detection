#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class Mux:
    def __init__(self):
        self.active = False
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/parking/active", Bool, self.cb_active, queue_size=1)
        rospy.Subscriber("/cmd_vel_nav", Twist, self.cb_nav, queue_size=10)
        rospy.Subscriber("/cmd_vel_parking", Twist, self.cb_parking, queue_size=10)

    def cb_active(self, m: Bool):
        self.active = m.data

    def cb_nav(self, msg: Twist):
        if not self.active:
            self.pub.publish(msg)

    def cb_parking(self, msg: Twist):
        if self.active:
            self.pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("cmd_vel_mux")
    Mux()
    rospy.spin()
