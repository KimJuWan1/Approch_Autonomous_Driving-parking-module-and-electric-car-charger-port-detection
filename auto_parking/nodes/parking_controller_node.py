#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, math
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

def angdiff(a, b):
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))

class PurePursuit:
    def __init__(self):
        self.path = None
        self.odom = None
        self.active = False

        self.cmd_pub = rospy.Publisher("/cmd_vel_parking", Twist, queue_size=1)
        rospy.Subscriber("/parking/active", Bool, self.cb_active, queue_size=1)
        rospy.Subscriber("/parking/path", Path, self.cb_path, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.cb_odom, queue_size=1)

        self.v_max   = rospy.get_param("~max_speed", 0.2)
        self.lfh     = rospy.get_param("~lookahead", 0.6)
        self.k_ang   = rospy.get_param("~k_ang", 1.5)
        self.xy_tol  = rospy.get_param("~xy_tol", 0.08)
        self.yaw_tol = rospy.get_param("~yaw_tol_deg", 3.0) * math.pi/180.0

        rospy.Timer(rospy.Duration(0.05), self.loop)

    def cb_active(self, msg: Bool): self.active = msg.data
    def cb_path(self, p: Path): self.path = p
    def cb_odom(self, o: Odometry): self.odom = o

    def loop(self, _):
        if not self.active or self.path is None or self.odom is None or len(self.path.poses) == 0:
            return

        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        q  = self.odom.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

        gx = self.path.poses[-1].pose.position.x
        gy = self.path.poses[-1].pose.position.y
        gq = self.path.poses[-1].pose.orientation
        # FIX: gq.y*gq.y (이전 코드에 q.y 들어가던 오타 수정)
        gyaw = math.atan2(2*(gq.w*gq.z+gq.x*gq.y), 1-2*(gq.y*gq.y+gq.z*gq.z))

        # 종료 조건
        if math.hypot(gx - px, gy - py) < self.xy_tol and abs(angdiff(gyaw, yaw)) < self.yaw_tol:
            self.cmd_pub.publish(Twist())
            return

        # lookahead 목표점
        target = None; acc = 0.0; last = (px, py)
        for ps in self.path.poses:
            x, y = ps.pose.position.x, ps.pose.position.y
            seg = math.hypot(x - last[0], y - last[1])
            acc += seg
            if acc >= self.lfh:
                target = (x, y); break
            last = (x, y)
        if target is None:
            target = (gx, gy)

        dx, dy = target[0] - px, target[1] - py
        tgt = math.atan2(dy, dx)
        err = angdiff(tgt, yaw)

        cmd = Twist()
        cmd.linear.x  = max(0.05, min(self.v_max, self.v_max * (1.0 - abs(err))))
        cmd.angular.z = self.k_ang * err
        self.cmd_pub.publish(cmd)

if __name__ == "__main__":
    rospy.init_node("parking_controller")
    PurePursuit()
    rospy.spin()
