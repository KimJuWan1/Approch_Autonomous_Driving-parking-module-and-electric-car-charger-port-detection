#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA, Int32MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


def quat_from_yaw(yaw):
    x, y, z, w = quaternion_from_euler(0, 0, yaw)
    return Quaternion(x=x, y=y, z=z, w=w)


def yaw_from_quat(qx, qy, qz, qw):
    _, _, yaw = euler_from_quaternion([qx, qy, qz, qw])
    return yaw


def make_box_marker(frame_id, mid, cx, cy, yaw, w, l, z=0.02,
                    color=(0.1, 0.9, 0.1, 0.9), lifetime=0.5):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = "parking_occupancy"
    m.id = mid
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = cx
    m.pose.position.y = cy
    m.pose.position.z = z
    m.pose.orientation = quat_from_yaw(yaw)
    m.scale.x = l
    m.scale.y = w
    m.scale.z = 0.02
    r, g, b, a = color
    m.color = ColorRGBA(r, g, b, a)
    m.lifetime = rospy.Duration(lifetime)
    return m


def make_text_marker(frame_id, mid, x, y, text, z=0.40, lifetime=0.5):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = "parking_occupancy_text"
    m.id = mid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z
    m.scale.z = 0.18
    m.color = ColorRGBA(1.0, 1.0, 0.2, 0.95)
    m.text = text
    m.lifetime = rospy.Duration(lifetime)
    return m


class FixedOccupancy:
    def __init__(self):
        # --- 기본 프레임/슬롯 ---
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.single_slot_force_free = bool(rospy.get_param("~single_slot_force_free", False))

        fs = rospy.get_param("~fixed_slots_m", [])
        self.slots = []
        for i, s in enumerate(fs):
            if len(s) != 5:
                rospy.logwarn("fixed_slots_m[%d] invalid: %s", i, str(s));  continue
            cx, cy, yaw_deg, w, l = [float(v) for v in s]
            self.slots.append(dict(cx=cx, cy=cy, yaw=math.radians(yaw_deg), w=w, l=l))
        if not self.slots:
            rospy.logwarn("No slots given. Using 3 default demo slots.")
            self.slots = [
                dict(cx=2.0, cy=-0.60, yaw=math.radians(90.0), w=0.60, l=1.00),
                dict(cx=2.0, cy= 0.00, yaw=math.radians(90.0), w=0.60, l=1.00),
                dict(cx=2.0, cy= 0.60, yaw=math.radians(90.0), w=0.60, l=1.00),
            ]

        # --- 점군 입력 (자동 선택) ---
        self.use_cloud = bool(rospy.get_param("~use_cloud_for_occupancy", True))
        self.user_cloud_topic = rospy.get_param("~cloud_topic", "")  # 비어있으면 자동 탐색
        self.topic_candidates = rospy.get_param("~cloud_topic_candidates", [
            "/lio_localizer/localization/cloud_deskewed",
            "/lio_sam/mapping/cloud_registered",
            "/lio_sam/mapping/cloud_deskewed",
            "/ouster/points",
        ])
        self.prefer_map_cloud = bool(rospy.get_param("~prefer_map_cloud", True))
        self.topic_scan_period = float(rospy.get_param("~topic_scan_period", 2.0))

        # --- TF/필터/판정 ---
        self.use_tf = bool(rospy.get_param("~use_tf", True))
        self.target_frame = rospy.get_param("~target_frame", self.frame_id)
        self.z_low  = float(rospy.get_param("~obstacle_z_low", 0.00))
        self.z_high = float(rospy.get_param("~obstacle_z_high", 1.60))
        self.slot_inset = float(rospy.get_param("~slot_inset_m", 0.05))
        self.occ_mode = rospy.get_param("~occupied_mode", "count").lower()
        self.occ_min_points = int(rospy.get_param("~occupied_min_points", 8))
        self.occ_ratio_thr  = float(rospy.get_param("~occupied_ratio_thr", 0.12))

        self.best_strategy = rospy.get_param("~best_strategy", "closest_x")
        self.front_min_m = float(rospy.get_param("~front_min_m", 0.30))

        self.show_counts = bool(rospy.get_param("~show_counts", True))
        self.force_red   = bool(rospy.get_param("~force_red_when_cloud_seen", False))

        # --- I/O ---
        self.pub_free = rospy.Publisher("/parking/free_slots", Int32MultiArray, queue_size=1, latch=True)
        self.pub_best = rospy.Publisher("/parking/best_slot", PoseStamped, queue_size=1)
        self.pub_mk   = rospy.Publisher("/parking/slots_markers", MarkerArray, queue_size=1)

        self.tfbuf = tf2_ros.Buffer(rospy.Duration(60.0)) if self.use_tf else None
        self.tflistener = tf2_ros.TransformListener(self.tfbuf) if self.use_tf else None

        self.sub_cloud = None
        self.current_topic = None
        self.last_cloud = None
        self.last_cloud_frame = None
        self.cloud_in_target = False

        # 타이머
        rate_hz = float(rospy.get_param("~update_rate", 8.0))
        self.timer = rospy.Timer(rospy.Duration(1.0/max(1e-3, rate_hz)), self.on_timer)
        self.scan_timer = rospy.Timer(rospy.Duration(self.topic_scan_period), self.on_scan_topics)

        rospy.loginfo("[parking_occupancy] ready. prefer_map_cloud=%s candidates=%s",
                      str(self.prefer_map_cloud), str(self.topic_candidates))

        # 초기 구독 설정
        self.ensure_cloud_subscription(initial=True)

    # ---------- 토픽 자동 선택 ----------
    def choose_best_topic(self, published):
        # published: list of (name, type)
        if self.user_cloud_topic:
            # 사용자가 강제 지정하면 그걸 우선
            return self.user_cloud_topic if any(t[0]==self.user_cloud_topic for t in published) else None
        # 후보 순서대로, 먼저 떠 있는 걸 고른다(리스트 앞쪽에 map 프레임 후보 배치)
        names = [n for (n, _) in published]
        for cand in self.topic_candidates:
            if cand in names:
                return cand
        return None

    def ensure_cloud_subscription(self, initial=False):
        published = rospy.get_published_topics()
        topic = self.choose_best_topic(published)
        if not topic:
            if initial:
                rospy.logwarn("[parking_occupancy] No candidate cloud topics yet. Waiting...")
            return
        if topic == self.current_topic:
            return
        # (재)구독
        if self.sub_cloud is not None:
            try: self.sub_cloud.unregister()
            except Exception: pass
        self.sub_cloud = rospy.Subscriber(topic, PointCloud2, self.cb_cloud, queue_size=1)
        self.current_topic = topic
        rospy.loginfo("[parking_occupancy] Subscribing cloud: %s", topic)

    def on_scan_topics(self, _):
        self.ensure_cloud_subscription(initial=False)

    # ---------- 콜백/TF ----------
    def cb_cloud(self, msg: PointCloud2):
        src = msg.header.frame_id
        self.last_cloud_frame = src
        if not self.use_tf:
            self.last_cloud = msg
            self.cloud_in_target = (src == self.target_frame)
            return
        try:
            if src != self.target_frame:
                try:
                    tf = self.tfbuf.lookup_transform(self.target_frame, src, msg.header.stamp, rospy.Duration(0.2))
                except Exception:
                    tf = self.tfbuf.lookup_transform(self.target_frame, src, rospy.Time(0), rospy.Duration(0.2))
                out = do_transform_cloud(msg, tf)
                out.header.frame_id = self.target_frame
                self.last_cloud = out
                self.cloud_in_target = True
            else:
                self.last_cloud = msg
                self.cloud_in_target = True
            rospy.loginfo_throttle(1.0, "[parking_occupancy] cloud ok: %s -> %s (in_target=%s)",
                                   src, self.target_frame, str(self.cloud_in_target))
        except Exception as e:
            self.last_cloud = msg
            self.cloud_in_target = False
            rospy.logwarn_throttle(1.0, "[parking_occupancy] TF fail %s->%s : %s (raw cloud & slot-fallback)",
                                   src, self.target_frame, str(e))

    # ---------- 좌표/판정 ----------
    @staticmethod
    def world_to_slot_local(px, py, cx, cy, yaw):
        dx, dy = px - cx, py - cy
        c, s = math.cos(-yaw), math.sin(-yaw)
        return (c * dx - s * dy, s * dx + c * dy)

    def transform_slot_to_frame(self, s, to_frame):
        if not self.use_tf:
            return None
        try:
            tf = self.tfbuf.lookup_transform(to_frame, self.frame_id, rospy.Time(0), rospy.Duration(0.2))
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            q = tf.transform.rotation
            tyaw = yaw_from_quat(q.x, q.y, q.z, q.w)
            cx_to = math.cos(tyaw)*s["cx"] - math.sin(tyaw)*s["cy"] + tx
            cy_to = math.sin(tyaw)*s["cx"] + math.cos(tyaw)*s["cy"] + ty
            yaw_to = s["yaw"] + tyaw
            return dict(cx=cx_to, cy=cy_to, yaw=yaw_to, w=s["w"], l=s["l"])
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[parking_occupancy] slot TF fail %s -> %s : %s",
                                   self.frame_id, to_frame, str(e))
            return None

    def slot_occupied(self, s):
        if not (self.use_cloud and self.last_cloud is not None):
            return False, 0
        if self.force_red:
            for _ in pc2.read_points(self.last_cloud, field_names=("x", "y", "z"), skip_nans=True):
                return True, 1
            return False, 0

        slot_for_calc = s
        if not self.cloud_in_target:
            to_frame = self.last_cloud_frame
            tf_slot = self.transform_slot_to_frame(s, to_frame)
            if tf_slot is not None:
                slot_for_calc = tf_slot
            else:
                return False, 0

        w = max(0.01, slot_for_calc["w"] - 2 * self.slot_inset)
        l = max(0.01, slot_for_calc["l"] - 2 * self.slot_inset)
        hx, hy = 0.5 * l, 0.5 * w

        inbox = 0
        occ = 0
        for x, y, z in pc2.read_points(self.last_cloud, field_names=("x", "y", "z"), skip_nans=True):
            xL, yL = self.world_to_slot_local(x, y, slot_for_calc["cx"], slot_for_calc["cy"], slot_for_calc["yaw"])
            if (-hx <= xL <= hx) and (-hy <= yL <= hy):
                inbox += 1
                if self.z_low <= z <= self.z_high:
                    occ += 1

        if self.occ_mode == "ratio":
            ratio = (float(occ) / inbox) if inbox > 0 else 0.0
            return (ratio >= self.occ_ratio_thr), ratio
        else:
            return (occ >= self.occ_min_points), occ

    # ---------- 베스트/퍼블리시 ----------
    def choose_best(self, free_idxs):
        if not free_idxs:
            return None
        cand = [(i, self.slots[i]) for i in free_idxs]
        if self.best_strategy == "closest_x":
            cand = [(i, s) for (i, s) in cand if s["cx"] >= self.front_min_m] or cand
            cand.sort(key=lambda t: (abs(t[1]["cx"]), abs(t[1]["cy"])))
            return cand[0][0]
        else:
            cand.sort(key=lambda t: math.hypot(t[1]["cx"], t[1]["cy"]))
            return cand[0][0]

    def on_timer(self, _evt):
        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        mk = MarkerArray()
        free_idxs = []

        for i, s in enumerate(self.slots):
            occupied, val = self.slot_occupied(s)

            # ★ 테스트 모드: 슬롯이 정확히 1개면 무조건 free 취급 (마커도 초록으로 표시)
            if self.single_slot_force_free and len(self.slots) == 1:
                if occupied:
                    rospy.logwarn_throttle(
                        2.0,
                        "[parking_occupancy] OVERRIDE: single-slot test → force FREE (was RED)"
                    )
                occupied = False

            col = (0.85, 0.15, 0.15, 0.9) if occupied else (0.10, 0.85, 0.10, 0.9)
            mk.markers.append(make_box_marker(self.frame_id, i, s["cx"], s["cy"], s["yaw"], s["w"], s["l"], 0.02, col))
            if self.show_counts:
                txt = f"{int(val)}" if self.occ_mode == "count" else f"{val:.2f}"
                mk.markers.append(make_text_marker(self.frame_id, 1000 + i, s["cx"], s["cy"], txt))
            if not occupied:
                free_idxs.append(i)

        self.pub_mk.publish(mk)
        self.pub_free.publish(Int32MultiArray(data=free_idxs))

        best_idx = self.choose_best(free_idxs)
        if best_idx is not None:
            b = self.slots[best_idx]
            self.pub_best.publish(PoseStamped(header=header,
                                              pose=Pose(Point(b["cx"], b["cy"], 0.0), quat_from_yaw(b["yaw"]))))

        rospy.loginfo_throttle(1.5, "[parking_occupancy] topic=%s cloud_frame=%s in_target=%s free=%s best=%s",
                               str(self.current_topic), str(self.last_cloud_frame), str(self.cloud_in_target),
                               str(free_idxs), "none" if best_idx is None else str(best_idx))


def main():
    rospy.init_node("fixed_parking_occupancy")
    FixedOccupancy()
    rospy.spin()


if __name__ == "__main__":
    main()
