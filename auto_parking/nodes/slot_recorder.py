#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RViz 2D Nav Goal( /move_base_simple/goal )을 클릭-드래그로 찍어서
[ cx, cy, yaw_deg, width, length ] 항목을 모아 YAML로 출력/저장.

- RViz에서 '2D Nav Goal' 도구로 슬롯 중심을 클릭하고 드래그해 방향(yaw)을 지정.
- width/length는 파라미터로 설정(기본 0.60 x 1.00 m).
- 클릭할 때마다 Marker로 박스 미리보기 표시.
- 종료(Ctrl+C) 시 /tmp/fixed_slots.yaml로 저장하고, 콘솔에 launch에 붙여넣을 YAML도 출력.
"""

import rospy, math, os, yaml
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def quat_from_yaw(yaw):
    x,y,z,w = quaternion_from_euler(0,0,yaw)
    return Quaternion(x=x,y=y,z=z,w=w)

def make_box_marker(frame_id, mid, cx, cy, yaw, w, l, z=0.02, color=(0.1,0.9,0.1,0.9)):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = "slot_recorder"
    m.id = mid
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = cx; m.pose.position.y = cy; m.pose.position.z = z
    m.pose.orientation = quat_from_yaw(yaw)
    m.scale.x = l; m.scale.y = w; m.scale.z = 0.02
    r,g,b,a = color
    m.color.r, m.color.g, m.color.b, m.color.a = r,g,b,a
    m.lifetime = rospy.Duration(0.0)
    return m

class SlotRecorder:
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.width    = float(rospy.get_param("~width", 0.60))
        self.length   = float(rospy.get_param("~length", 1.00))
        self.out_path = rospy.get_param("~out", "/tmp/fixed_slots.yaml")

        self.slots = []  # dict {cx,cy,yaw_deg,w,l}
        self.pub_mk = rospy.Publisher("/parking/debug/markers", MarkerArray, queue_size=1)
        self.sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=10)

        rospy.loginfo("[slot_recorder] frame_id=%s width=%.2f length=%.2f -> click with RViz '2D Nav Goal'",
                      self.frame_id, self.width, self.length)
        rospy.loginfo("[slot_recorder] Press Ctrl+C to save to %s", self.out_path)

        rospy.on_shutdown(self.on_shutdown)

    def cb_goal(self, msg: PoseStamped):
        if msg.header.frame_id != self.frame_id:
            rospy.logwarn("Received goal in frame '%s' but frame_id is '%s' (will still record).",
                          msg.header.frame_id, self.frame_id)
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        _,_,yaw = euler_from_quaternion([q.x,q.y,q.z,q.w])
        yaw_deg = math.degrees(yaw)

        item = dict(cx=float(x), cy=float(y), yaw_deg=float(yaw_deg),
                    w=self.width, l=self.length)
        self.slots.append(item)

        # draw markers (index color)
        mk = MarkerArray()
        mid = 0
        for i, s in enumerate(self.slots):
            col = (0.10, 0.85, 0.10, 0.9)
            mk.markers.append(make_box_marker(self.frame_id, mid, s["cx"], s["cy"],
                                              math.radians(s["yaw_deg"]), s["w"], s["l"], 0.02, col))
            mid += 1
        self.pub_mk.publish(mk)

        rospy.loginfo("[slot_recorder] added #%d: [%.2f, %.2f, %.1f°, %.2f, %.2f]",
                      len(self.slots), x, y, yaw_deg, self.width, self.length)

        # 콘솔에 현재까지 YAML 라인도 바로 찍어줌
        print(self._yaml_snippet())

    def _yaml_snippet(self):
        arr = [[round(s["cx"],3), round(s["cy"],3), round(s["yaw_deg"],1), round(s["w"],2), round(s["l"],2)] for s in self.slots]
        return "fixed_slots_m: " + yaml.safe_dump(arr, default_flow_style=True).strip()

    def on_shutdown(self):
        try:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            arr = [[s["cx"], s["cy"], s["yaw_deg"], s["w"], s["l"]] for s in self.slots]
            with open(self.out_path, "w") as f:
                yaml.safe_dump(dict(fixed_slots_m=arr), f, default_flow_style=True)
            print("\n\n[slot_recorder] Saved to:", self.out_path)
            print("Paste into launch:\n  <param name=\"fixed_slots_m\" type=\"yaml\" value=\"%s\"/>" %
                  yaml.safe_dump(arr, default_flow_style=True).strip().replace("\n",""))
        except Exception as e:
            rospy.logwarn("Failed to save: %s", e)

def main():
    rospy.init_node("slot_recorder")
    SlotRecorder()
    rospy.spin()

if __name__ == "__main__":
    main()
