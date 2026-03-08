#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from auto_parking.msg import ParkingSlotArray

def yaw_from_quat(q):
    # q is geometry_msgs/Quaternion
    # yaw only
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cos_y_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    return math.atan2(siny_cosp, cos_y_cosp)

def rect_corners(cx, cy, yaw, length, width, z):
    c = math.cos(yaw); s = math.sin(yaw)
    hl = 0.5*length; hw = 0.5*width
    # local corners (front-left, front-right, back-right, back-left)
    pts_local = [( hl,  hw), ( hl, -hw), (-hl, -hw), (-hl,  hw)]
    out = []
    for x,y in pts_local:
        X = cx + c*x - s*y
        Y = cy + s*x + c*y
        out.append(Point(x=X, y=Y, z=z))
    # close loop
    out.append(out[0])
    return out

class DebugViz(object):
    def __init__(self):
        self.frame_id   = rospy.get_param("~frame_id", "map")
        self.topic      = rospy.get_param("~marker_topic", "/parking/debug/markers")
        self.line_width = float(rospy.get_param("~line_width", 0.08))  # m
        self.text_size  = float(rospy.get_param("~text_size", 0.40))   # m
        self.z_up       = float(rospy.get_param("~z_up", 0.08))        # m
        self.color_box  = ColorRGBA(0.0, 0.8, 1.0, 1.0)  # cyan
        self.color_best = ColorRGBA(1.0, 0.2, 0.2, 1.0)  # red
        self.color_txt  = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # white

        # MarkerArray를 래치해서 RViz가 늦게 붙어도 바로 보이게
        self.pub = rospy.Publisher(self.topic, MarkerArray, queue_size=10, latch=True)

        self.last_slots = None
        self.best = None

        rospy.Subscriber("/parking/slots", ParkingSlotArray, self.cb_slots, queue_size=1)
        rospy.Subscriber("/parking/best_slot", PoseStamped, self.cb_best, queue_size=1)

        rospy.loginfo("[debug_viz] ready: line_width=%.2f text_size=%.2f z_up=%.2f",
                      self.line_width, self.text_size, self.z_up)

        self.timer = rospy.Timer(rospy.Duration(0.2), self._tick)  # 5 Hz

    def cb_slots(self, msg):
        self.last_slots = msg

    def cb_best(self, msg):
        self.best = msg

    def _tick(self, _):
        arr = MarkerArray()
        # 항상 깨끗이 지우고 다시 그림
        mclr = Marker()
        mclr.header.frame_id = self.frame_id
        mclr.action = Marker.DELETEALL
        arr.markers.append(mclr)

        mid = 0
        now = rospy.Time.now()

        if self.last_slots is not None:
            for i, s in enumerate(self.last_slots.slots):
                cx = s.pose.position.x
                cy = s.pose.position.y
                yaw = yaw_from_quat(s.pose.orientation)
                L   = float(getattr(s, "length", 1.0))
                W   = float(getattr(s, "width",  0.5))
                z   = self.z_up

                # 박스(라인스트립)
                box = Marker()
                box.header.frame_id = self.frame_id
                box.header.stamp    = now
                box.ns   = "slot_box"
                box.id   = mid; mid += 1
                box.type = Marker.LINE_STRIP
                box.action = Marker.ADD
                box.pose.orientation.w = 1.0
                box.scale.x = max(1e-3, self.line_width)
                box.color = self.color_box
                box.lifetime = rospy.Duration(0.0)
                box.points = rect_corners(cx, cy, yaw, L, W, z)
                arr.markers.append(box)

                # 텍스트
                txt = Marker()
                txt.header.frame_id = self.frame_id
                txt.header.stamp    = now
                txt.ns   = "slot_text"
                txt.id   = mid; mid += 1
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = cx
                txt.pose.position.y = cy
                txt.pose.position.z = z + 0.02
                txt.scale.z = self.text_size
                txt.color = self.color_txt
                txt.lifetime = rospy.Duration(0.0)
                txt.text = "score=%.2f  occ=%s" % (getattr(s, "score", 0.0),
                                                   str(getattr(s, "occupied", False)))
                arr.markers.append(txt)

        if self.best is not None:
            # best 슬롯 중앙에 화살표(고정 길이)
            arrow = Marker()
            arrow.header.frame_id = self.frame_id
            arrow.header.stamp    = now
            arrow.ns   = "best"
            arrow.id   = mid; mid += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose = self.best.pose
            arrow.pose.position.z += self.z_up
            # 화살표 스케일: x 길이, y/ z 두께
            arrow.scale.x = 0.8
            arrow.scale.y = max(1e-3, self.line_width*1.2)
            arrow.scale.z = max(1e-3, self.line_width*1.2)
            arrow.color = self.color_best
            arrow.lifetime = rospy.Duration(0.0)
            arr.markers.append(arrow)

        self.pub.publish(arr)

def main():
    rospy.init_node("parking_debug_viz")
    DebugViz()
    rospy.spin()

if __name__ == "__main__":
    main()
