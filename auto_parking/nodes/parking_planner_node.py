#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, math
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

class Planner:
    def __init__(self):
        self.path_pub = rospy.Publisher("/parking/path", Path, queue_size=1, latch=True)
        rospy.Subscriber("/odom", Odometry, self.cb_odom, queue_size=1)
        rospy.Subscriber("/parking/best_slot", PoseStamped, self.cb_target, queue_size=1)
        self.odom = None
        self.target = None
        self.frame = rospy.get_param("~frame_id", "map")
        self.res   = rospy.get_param("~step", 0.1)

    def cb_odom(self, msg: Odometry): self.odom = msg
    def cb_target(self, msg: PoseStamped): self.target = msg; self.plan()

    def plan(self):
        if self.odom is None or self.target is None:
            return

        frame = self.target.header.frame_id or self.frame
        start = self.odom.pose.pose.position
        goal  = self.target.pose.position

        # NOTE: 실제에선 TF로 start를 frame에 맞춰야 함(여긴 단순 더미)
        dx, dy = goal.x - start.x, goal.y - start.y
        dist = max(0.05, math.hypot(dx, dy))
        n = int(dist / self.res)

        poses = []
        for i in range(n+1):
            t = float(i) / max(1, n)
            ps = PoseStamped()
            ps.header.frame_id = frame
            ps.pose.position.x = start.x + dx * t
            ps.pose.position.y = start.y + dy * t
            ps.pose.orientation = self.target.pose.orientation
            poses.append(ps)

        path = Path()
        path.header = Header(stamp=rospy.Time.now(), frame_id=frame)
        path.poses = poses
        self.path_pub.publish(path)
        rospy.loginfo_throttle(2.0, "[parking_planner] frame=%s len=%d", frame, len(poses))

if __name__ == "__main__":
    rospy.init_node("parking_planner")
    Planner()
    rospy.spin()
