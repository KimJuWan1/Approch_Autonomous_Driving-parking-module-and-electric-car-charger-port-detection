#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import PointCloud2
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

class PointsFrameRelay:
    def __init__(self):
        self.in_topic  = rospy.get_param("~input", "/ouster/points")
        self.out_topic = rospy.get_param("~output", "/ouster/points_map")
        self.target    = rospy.get_param("~target_frame", "map")
        self.timeout   = rospy.get_param("~timeout_sec", 0.2)

        self.buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.lst = tf2_ros.TransformListener(self.buf)

        self.pub = rospy.Publisher(self.out_topic, PointCloud2, queue_size=5)
        rospy.Subscriber(self.in_topic, PointCloud2, self.cb, queue_size=1)

        rospy.loginfo("[points_frame_relay] %s -> %s (target_frame=%s)",
                      self.in_topic, self.out_topic, self.target)

    def cb(self, msg: PointCloud2):
        in_stamp = msg.header.stamp
        in_frame = msg.header.frame_id

        try:
            if in_frame == self.target:
                out = msg  # 이미 타겟 프레임
            else:
                tf = self.buf.lookup_transform(self.target, in_frame, in_stamp,
                                               rospy.Duration(self.timeout))
                out = do_transform_cloud(msg, tf)

            # 드물게 변환 후 stamp가 0으로 오는 드라이버 케이스 보정
            if out.header.stamp.to_sec() == 0.0 and in_stamp.to_sec() > 0.0:
                out.header.stamp = in_stamp

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            # TF 미준비: 원본 그대로 패스스루 (프레임/타임스탬프 모두 보존)
            rospy.logwarn_throttle(2.0,
                '[points_frame_relay] TF %s->%s not ready (%s). passthrough original frame "%s".',
                in_frame, self.target, str(e), in_frame)
            out = msg

        self.pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("points_frame_relay")
    PointsFrameRelay()
    rospy.spin()
