#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, csv, os, math
from auto_parking.msg import ParkingSlotArray

class Logger:
    def __init__(self):
        out = rospy.get_param("~out", "/tmp/parking_slots.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        self.fp = open(out, "w", newline="")
        self.w  = csv.writer(self.fp)
        self.w.writerow(["stamp","frame","idx","x","y","yaw_deg","width","length","score","occupied","n_slots"])
        rospy.Subscriber("/parking/slots", ParkingSlotArray, self.cb, queue_size=10)
        rospy.loginfo("[slots_logger] writing to %s", out)

    def cb(self, msg: ParkingSlotArray):
        frame = msg.header.frame_id
        n = len(msg.slots)
        for i, s in enumerate(msg.slots):
            q = s.pose.orientation
            yaw = math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
            self.w.writerow([
                msg.header.stamp.to_sec(), frame, i,
                s.pose.position.x, s.pose.position.y, yaw,
                s.width, s.length, f"{s.score:.3f}", int(s.occupied), n
            ])
        self.fp.flush()

if __name__ == "__main__":
    rospy.init_node("parking_slots_logger")
    Logger()
    rospy.spin()
