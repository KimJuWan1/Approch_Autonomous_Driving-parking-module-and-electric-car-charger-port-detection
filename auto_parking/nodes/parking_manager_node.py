#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, math, actionlib
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from auto_parking.msg import ParkInAction, ParkInFeedback, ParkInResult

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

class Manager:
    def __init__(self):
        self.active_pub = rospy.Publisher("/parking/active", Bool, queue_size=1, latch=True)
        self.best_pub   = rospy.Publisher("/parking/best_slot", PoseStamped, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.cb_odom, queue_size=1)
        rospy.Subscriber("/parking/path", Path, self.cb_path, queue_size=1)
        self.odom = None
        self.path = None

        self.xy_tol = rospy.get_param("~xy_tol", 0.08)
        self.server = actionlib.SimpleActionServer("/parking/park_in", ParkInAction,
                                                   execute_cb=self.execute, auto_start=False)
        self.server.start()
        rospy.loginfo("[auto_parking/manager] action server ready.")

    def cb_odom(self, o: Odometry): self.odom = o
    def cb_path(self, p: Path): self.path = p

    def execute(self, goal):
        self.active_pub.publish(Bool(data=True))
        if goal.target.header.frame_id:
            self.best_pub.publish(goal.target)

        fb = ParkInFeedback(state="PLANNING", progress=0.1)
        rate = rospy.Rate(10)
        success = False
        t0 = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.server.is_preempt_requested():
                self.active_pub.publish(Bool(data=False))
                self.server.set_preempted()
                return

            if self.odom and self.path and len(self.path.poses) > 0:
                px = self.odom.pose.pose.position.x; py = self.odom.pose.pose.position.y
                gx = self.path.poses[-1].pose.position.x; gy = self.path.poses[-1].pose.position.y
                d = dist((px, py), (gx, gy))
                fb.state = "EXECUTING"
                fb.progress = max(0.1, 1.0 - min(1.0, d / 2.0))
                self.server.publish_feedback(fb)
                if d < self.xy_tol:
                    success = True
                    break

            if (rospy.Time.now() - t0).to_sec() > 120.0:
                break

            rate.sleep()

        self.active_pub.publish(Bool(data=False))
        res = ParkInResult(success=success, message=("OK" if success else "Timeout/Fail"))
        (self.server.set_succeeded if success else self.server.set_aborted)(res)

if __name__ == "__main__":
    rospy.init_node("parking_manager")
    Manager()
    rospy.spin()
