#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy, math, os, yaml
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def make_box(frame, mid, cx, cy, yaw, w, l):
    m=Marker(); m.header.frame_id=frame; m.ns="slot_rec"; m.id=mid
    m.type=Marker.CUBE; m.action=Marker.ADD; m.pose.position.x=cx; m.pose.position.y=cy; m.pose.position.z=0.02
    x,y,z,wq = quaternion_from_euler(0,0,yaw)
    m.pose.orientation.x=x; m.pose.orientation.y=y; m.pose.orientation.z=z; m.pose.orientation.w=wq
    m.scale.x=l; m.scale.y=w; m.scale.z=0.02; m.color.r=0.1; m.color.g=0.85; m.color.b=0.1; m.color.a=0.9
    return m

class SlotRec:
    def __init__(self):
        self.frame = rospy.get_param("~frame_id","map")
        self.w = float(rospy.get_param("~width",0.60))
        self.l = float(rospy.get_param("~length",0.50))
        self.out = rospy.get_param("~out","/tmp/fixed_slots.yaml")
        self.slots=[]
        self.pub = rospy.Publisher("/parking/slots_markers", MarkerArray, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb)
        rospy.on_shutdown(self.save)
        rospy.loginfo("[slot_recorder] Click with '2D Nav Goal' in RViz (frame=%s). Ctrl+C to save -> %s", self.frame, self.out)

    def cb(self, msg):
        x=msg.pose.position.x; y=msg.pose.position.y
        q=msg.pose.orientation; yaw= euler_from_quaternion([q.x,q.y,q.z,q.w])[2]
        self.slots.append([float(x), float(y), math.degrees(yaw), self.w, self.l])
        mk=MarkerArray()
        for i,s in enumerate(self.slots):
            mk.markers.append(make_box(self.frame, i, s[0], s[1], math.radians(s[2]), s[3], s[4]))
        self.pub.publish(mk)
        rospy.loginfo("[slot_recorder] #%d  [%.2f, %.2f, %.1f°, %.2f, %.2f]", len(self.slots), x,y,math.degrees(yaw),self.w,self.l)
        print("fixed_slots_m:", yaml.safe_dump(self.slots, default_flow_style=True).strip())

    def save(self):
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        with open(self.out,"w") as f: yaml.safe_dump(dict(fixed_slots_m=self.slots), f, default_flow_style=True)
        print("\nSaved:", self.out)
        print("Paste into launch:\n<param name=\"fixed_slots_m\" type=\"yaml\" value=\"%s\"/>" %
              yaml.safe_dump(self.slots, default_flow_style=True).strip().replace("\n",""))

if __name__=="__main__":
    rospy.init_node("slot_recorder"); SlotRec(); rospy.spin()
