#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Back-in Parking Controller — lock slot after activation + rear obstacle stop

FSM:
  ROTATE_FOR_PREPOSE → REVERSE_TO_PREPOSE → ROTATE_TO_SLOT → REVERSE_TO_SLOT → DONE

핵심
- 활성화 시(best_slot) 슬롯을 고정(LOCK)하고 주차 종료까지 변경 무시
- REVERSE_TO_SLOT에서 v = -min(v_back_insert, max(v_creep_max, k_vx * xL)) (P형)
- xL <= 0.0 이면 절대 전진 금지(v=0) + 중심 통과(+→−) 직후 브레이크 & 홀드
- 뒤쪽 ROI에서 장애물 감지되면 후진(v<0) 명령 강제 0 (DWA 스타일 rear stop)
"""

import math
import rospy
import tf2_ros

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2


# -------- utils --------
def ang_norm(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def yaw_of(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def offset_world(sx, sy, syaw, dx, dy=0.0):
    c, s = math.cos(syaw), math.sin(syaw)
    return (sx + c * dx - s * dy,
            sy + s * dx + c * dy)


def world_to_slot_local(px, py, sx, sy, syaw):
    dx, dy = px - sx, py - sy
    c, s = math.cos(-syaw), math.sin(-syaw)
    return (c * dx - s * dy,
            s * dx + c * dy)


# -------- controller --------
class BackInParking:
    ROTATE_FOR_PREPOSE = "ROTATE_FOR_PREPOSE"
    REVERSE_TO_PREPOSE = "REVERSE_TO_PREPOSE"
    ROTATE_TO_SLOT     = "ROTATE_TO_SLOT"
    REVERSE_TO_SLOT    = "REVERSE_TO_SLOT"
    DONE               = "DONE"
    WAIT               = "WAIT"

    def __init__(self):
        # 프레임/토픽
        self.frame            = rospy.get_param("~frame_id", "map")
        self.slot_topic       = rospy.get_param("~slot_topic", "/parking/best_slot")
        self.nav_twist_topic  = rospy.get_param("~nav_twist_topic", "/cmd_vel_nav")
        self.arrived_topic    = rospy.get_param("~arrived_topic", "/nav/arrived")
        self.cmd_out          = rospy.get_param("~cmd_out", "/cmd_vel_park")

        # 게인/속도
        self.k_yaw            = float(rospy.get_param("~k_yaw", 2.2))
        self.k_y_line         = float(rospy.get_param("~k_y_line", 1.0))
        self.k_vx             = float(rospy.get_param("~k_vx", 0.8))   # xL→속도 비례계수
        self.w_max            = float(rospy.get_param("~w_max", 0.6))
        self.v_back_prepose   = float(rospy.get_param("~v_back_prepose", 0.18))
        self.v_back_insert    = float(rospy.get_param("~v_back_insert", 0.15))
        self.v_creep_max      = float(rospy.get_param("~v_creep_max", 0.06))

        # 기하/공차
        self.prepose_dist_m   = float(rospy.get_param("~prepose_dist_m", 1.0))
        self.prepose_tol_m    = float(rospy.get_param("~prepose_tol_m", 0.10))
        self.tol_x            = float(rospy.get_param("~tol_x", 0.10))
        self.tol_y            = float(rospy.get_param("~tol_y", 0.08))
        self.tol_yaw_deg      = float(rospy.get_param("~tol_yaw_deg", 5.0))
        self.hold_on_done_s   = float(rospy.get_param("~hold_on_done_s", 0.4))

        self.yaw_tol_prepose_deg = float(rospy.get_param("~yaw_tol_prepose_deg", 6.0))
        self.yaw_tol_slot_deg    = float(rospy.get_param("~yaw_tol_slot_deg",    4.0))

        # 백인 기본: 통로를 바라본 상태에서 후진
        self.back_in_face_outward = bool(rospy.get_param("~back_in_face_outward", True))

        # 시작/패스스루
        self.start_on_arrived     = bool(rospy.get_param("~start_on_arrived", True))
        self.trigger_radius_m     = float(rospy.get_param("~trigger_radius_m", 0.0))
        self.passthrough_until_active = bool(rospy.get_param("~passthrough_until_arrived", True))
        self.nav_timeout_s        = float(rospy.get_param("~nav_timeout_s", 1.0))

        # 슬롯 락 설정
        self.slot_lock        = bool(rospy.get_param("~slot_lock", True))  # 활성화 시 고정
        self.slot_lock_phase  = rospy.get_param("~slot_lock_phase", "on_activate")  # on_activate | on_prepose
        self.unlock_on_done   = bool(rospy.get_param("~unlock_on_done", True))

        # === 뒤쪽 장애물 감지 (정밀한 로봇 기하학 반영) ===
        # Robot dimensions: length=65cm, width=60cm, height to lidar=58cm
        # Lidar position: 0.5m above ground, center of robot
        self.cloud_topic      = rospy.get_param("~pointcloud_topic", "/ouster/points")
        
        # Robot geometry (meters)
        self.robot_length = float(rospy.get_param("~robot_length", 0.65))           # 65cm
        self.robot_width = float(rospy.get_param("~robot_width", 0.60))             # 60cm
        self.robot_height_to_lidar = float(rospy.get_param("~robot_height_to_lidar", 0.58))  # 58cm
        self.lidar_height = float(rospy.get_param("~lidar_height", 0.50))           # 50cm above ground
        
        # Collision detection parameters (rear)
        # stop_distance: rear buffer for reversing (default: 0.4m)
        self.stop_distance    = float(rospy.get_param("~stop_distance", 0.40))
        
        # stop_width: covers full robot width (60cm)
        self.stop_width       = float(rospy.get_param("~stop_width", 0.60))
        
        # Z-axis range: from ground to robot top (for rear collision)
        self.min_z            = float(rospy.get_param("~min_z", -0.50))   # Ground level (50cm below lidar)
        self.max_z            = float(rospy.get_param("~max_z", 0.0))     # Lidar height (top of robot detection)
        
        self.cloud_downsample = int(rospy.get_param("~cloud_downsample", 4))
        self.block_on_count   = int(rospy.get_param("~block_on_count", 2))
        self.block_off_count  = int(rospy.get_param("~block_off_count", 3))

        self.obstacle_blocked = False
        self.obstacle_min_d   = None
        self.obstacle_on_cnt  = 0
        self.obstacle_off_cnt = 0

        # 상태
        self.phase         = self.WAIT
        self.phase_enter   = rospy.Time.now()
        self.active        = False
        self.arrived_flag  = False

        self.slot_latest   = None   # 최신 수신값
        self.slot_fixed    = None   # 고정해 쓸 값
        self.slot_locked   = False

        self.done_since    = None
        self.prev_xL       = None
        self.nav_twist     = Twist()
        self.nav_last_ts   = rospy.Time(0)

        # TF
        self.tfbuf = tf2_ros.Buffer(rospy.Duration(20.0))
        self.tfl   = tf2_ros.TransformListener(self.tfbuf)

        # IO
        self.pub = rospy.Publisher(self.cmd_out, Twist, queue_size=1)
        rospy.Subscriber(self.slot_topic, PoseStamped, self.cb_slot, queue_size=1)
        rospy.Subscriber(self.nav_twist_topic, Twist, self.cb_nav, queue_size=1)
        if self.start_on_arrived:
            rospy.Subscriber(self.arrived_topic, Bool, self.cb_arrived, queue_size=1)

        # 뒤쪽 장애물용 포인트클라우드
        rospy.Subscriber(self.cloud_topic, PointCloud2, self.cb_cloud, queue_size=1)

        # 타이머
        rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        rospy.Timer(rospy.Duration(1.0 / rate_hz), self.on_timer)

        rospy.loginfo("[back_in_parking] FSM=%s → %s → %s → %s → %s  out=%s",
                      self.ROTATE_FOR_PREPOSE, self.REVERSE_TO_PREPOSE,
                      self.ROTATE_TO_SLOT, self.REVERSE_TO_SLOT, self.DONE, self.cmd_out)
        rospy.loginfo("[back_in_parking] rear obstacle ROI: topic=%s, R=%.2fm, W=%.2fm",
                      self.cloud_topic, self.stop_distance, self.stop_width)

    # --- callbacks ---
    def cb_slot(self, msg: PoseStamped):
        # 항상 최신은 보관
        self.slot_latest = msg
        self.done_since = None

        # 락 전에는 고정값도 최신으로 갱신
        if not self.slot_locked:
            self.slot_fixed = msg

    def cb_nav(self, m: Twist):
        self.nav_twist = m
        self.nav_last_ts = rospy.Time.now()

    def cb_arrived(self, m: Bool):
        self.arrived_flag = bool(m.data)

    def cb_cloud(self, msg: PointCloud2):
        """
        뒤쪽 ROI에서 장애물 감지 (정밀한 로봇 기하학 반영 + 자체-필터링).
        - Robot dimensions: 65cm(L) x 60cm(W) x 58cm(H to lidar)
        - Lidar at: center of robot, 50cm above ground
        - Detection zone: x < 0 (rear), |y| <= 30cm (half width), z in [-50cm, 0cm] (ground to lidar)
        - Self-filter: Exclude points from robot body itself to avoid false positives
        """
        try:
            half_w = 0.5 * self.stop_width  # 30cm
            near = False
            min_d2 = None
            
            # Self-filter zone: robot body envelope in lidar frame
            # Robot extends ~±0.3m (half-width) in y, ~0.33m (half-length) in x
            # Self-filter margin: expand by 5cm safety for sensor noise
            self_filter_radius_y = 0.35  # 0.3m (half-width) + 0.05m margin
            self_filter_radius_x = 0.38  # 0.33m (half-length) + 0.05m margin

            for i, (x, y, z) in enumerate(
                pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True),
                start=1
            ):
                if self.cloud_downsample > 1 and (i % self.cloud_downsample != 0):
                    continue
                
                # 1. Filter: points too close to robot center (self-detection avoidance)
                if abs(x) < self_filter_radius_x and abs(y) < self_filter_radius_y:
                    continue  # Skip points from robot body itself

                # 2. 뒤쪽만 보기 (센서 x+가 앞이라고 가정)
                if x >= 0.0:
                    continue
                
                # 3. Z-axis filtering: from ground to lidar height
                # min_z = -0.5m (ground, 50cm below lidar)
                # max_z = 0.0m (lidar height, top of robot)
                if z < self.min_z or z > self.max_z:
                    continue
                
                # 4. Width filtering: robot is 60cm wide, centered at y=0
                if abs(y) > half_w:
                    continue

                # 5. Distance check: collision if within stop_distance
                d2 = x * x + y * y
                if d2 <= self.stop_distance * self.stop_distance:
                    near = True
                    if min_d2 is None or d2 < min_d2:
                        min_d2 = d2

            if near:
                self.obstacle_on_cnt += 1
                self.obstacle_off_cnt = 0
                if min_d2 is not None:
                    self.obstacle_min_d = math.sqrt(min_d2)
            else:
                self.obstacle_off_cnt += 1
                self.obstacle_on_cnt = 0
                self.obstacle_min_d = None

            # 히스테리시스 on/off
            if not self.obstacle_blocked and self.obstacle_on_cnt >= self.block_on_count:
                self.obstacle_blocked = True
                rospy.logwarn(f"[Rear Collision] OBSTACLE DETECTED at {self.stop_distance:.2f}m (robot width: {self.robot_width:.2f}m, height: ground to {self.lidar_height:.2f}m)")
            elif self.obstacle_blocked and self.obstacle_off_cnt >= self.block_off_count:
                self.obstacle_blocked = False
                rospy.loginfo("[Rear Collision] Rear obstacle cleared: resuming")

        except Exception as e:
            rospy.logwarn("cb_cloud error: %s", str(e))

    # --- helpers ---
    def robot_pose(self):
        try:
            tf = self.tfbuf.lookup_transform(self.frame, "base_link",
                                             rospy.Time(0), rospy.Duration(0.2))
            t = tf.transform.translation
            yaw = yaw_of(tf.transform.rotation)
            return (t.x, t.y, yaw)
        except Exception:
            return None

    def set_phase(self, ph):
        if self.phase != ph:
            self.phase = ph
            self.phase_enter = rospy.Time.now()
            rospy.loginfo("[back_in_parking] PHASE → %s", ph)

            # on_prepose 시점에 락 옵션
            if self.slot_lock and (ph == self.ROTATE_FOR_PREPOSE) and (self.slot_lock_phase == "on_prepose"):
                self._lock_slot_if_possible()

            if ph == self.DONE and self.unlock_on_done:
                self.slot_locked = False  # 다음 주차 대비 해제

    def maybe_passthrough(self):
        if not self.passthrough_until_active:
            return False
        if (rospy.Time.now() - self.nav_last_ts).to_sec() <= self.nav_timeout_s:
            self.pub.publish(self.nav_twist)
            return True
        return False

    def done_check(self, xL, yL, yaw_err):
        return (abs(xL) <= self.tol_x
                and abs(yL) <= self.tol_y
                and abs(math.degrees(ang_norm(yaw_err))) <= self.tol_yaw_deg)

    def _lock_slot_if_possible(self):
        if self.slot_lock and not self.slot_locked and self.slot_latest is not None:
            self.slot_fixed = self.slot_latest
            self.slot_locked = True
            rospy.loginfo("[back_in_parking] SLOT LOCKED at activation (x=%.2f,y=%.2f,yaw=%.1f°)",
                          self.slot_fixed.pose.position.x,
                          self.slot_fixed.pose.position.y,
                          math.degrees(yaw_of(self.slot_fixed.pose.orientation)))

    # --- primitives ---
    def rot_ctrl(self, yaw_err):
        w = clamp(self.k_yaw * ang_norm(yaw_err), -self.w_max, self.w_max)
        return 0.0, w

    def reverse_line_ctrl(self, rx, ry, ryaw, tx, ty, v_back):
        """
        목표점(t)을 등지고 직선 후진.
        """
        yaw_des = math.atan2(ry - ty, rx - tx)
        yaw_err = ang_norm(yaw_des - ryaw)
        v = -v_back
        w = clamp(self.k_yaw * yaw_err, -self.w_max, self.w_max)
        return v, w, yaw_err

    def reverse_slot_ctrl(self, rx, ry, ryaw, sx, sy, yaw_des):
        """
        슬롯 로컬 오차(xL, yL)를 보며 후진.
        - xL>0 : 중심까지 남은 거리 → v = -min(v_back_insert, max(v_creep_max, k_vx*xL))
        - xL<=0: 중심을 넘었으므로 절대 전진하지 않고 v=0 (브레이크)
        """
        xL, yL = world_to_slot_local(rx, ry, sx, sy, yaw_des)
        yaw_err = ang_norm(yaw_des - ryaw)

        if xL <= 0.0:
            v = 0.0
        else:
            v_cmd = min(self.v_back_insert, max(self.v_creep_max, self.k_vx * xL))
            v = -v_cmd  # 후진

        w = self.k_y_line * yL + self.k_yaw * yaw_err
        w = clamp(w, -self.w_max, self.w_max)
        return v, w, xL, yL, yaw_err

    # --- main loop ---
    def on_timer(self, _evt):
        twist = Twist()
        d_debug = -1.0  # 디버그용 거리 값

        # 슬롯 유효성
        if self.slot_fixed is None:
            if self.maybe_passthrough():
                return
            self.pub.publish(twist)
            return

        rp = self.robot_pose()
        if rp is None:
            if self.maybe_passthrough():
                return
            self.pub.publish(twist)
            return
        rx, ry, ryaw = rp

        # 활성화(슬롯 락 타이밍)
        if not self.active:
            dist_to_center = math.hypot(
                rx - self.slot_fixed.pose.position.x,
                ry - self.slot_fixed.pose.position.y
            )
            if self.start_on_arrived and self.arrived_flag:
                self.active = True
                if self.slot_lock_phase == "on_activate":
                    self._lock_slot_if_possible()
                self.set_phase(self.ROTATE_FOR_PREPOSE)
            elif self.trigger_radius_m > 0.0 and dist_to_center <= self.trigger_radius_m:
                self.active = True
                if self.slot_lock_phase == "on_activate":
                    self._lock_slot_if_possible()
                self.set_phase(self.ROTATE_FOR_PREPOSE)

        if not self.active:
            if self.maybe_passthrough():
                return
       

        # 사용 슬롯(고정본)
        sx = self.slot_fixed.pose.position.x
        sy = self.slot_fixed.pose.position.y
        syaw = yaw_of(self.slot_fixed.pose.orientation)

        # 프리포즈 : 슬롯 진행(+x) 쪽으로 prepose_dist 만큼 앞
        Ax, Ay = offset_world(sx, sy, syaw, +self.prepose_dist_m)
        yaw_prepose = math.atan2(ry - Ay, rx - Ax)

        # 최종 백인 yaw
        yaw_slot_back = syaw if self.back_in_face_outward else ang_norm(syaw + math.pi)

        # FSM
        if self.phase == self.WAIT:
            self.set_phase(self.ROTATE_FOR_PREPOSE)

        if self.phase == self.ROTATE_FOR_PREPOSE:
            yaw_err = ang_norm(yaw_prepose - ryaw)
            d_debug = math.hypot(rx - Ax, ry - Ay)
            if abs(math.degrees(yaw_err)) <= self.yaw_tol_prepose_deg:
                self.set_phase(self.REVERSE_TO_PREPOSE)
                v, w = 0.0, 0.0
            else:
                v, w = self.rot_ctrl(yaw_err)

        elif self.phase == self.REVERSE_TO_PREPOSE:
            d = math.hypot(rx - Ax, ry - Ay)
            d_debug = d
            if d <= self.prepose_tol_m:
                self.set_phase(self.ROTATE_TO_SLOT)
                v, w = 0.0, 0.0
            else:
                v, w, _ = self.reverse_line_ctrl(
                    rx, ry, ryaw, Ax, Ay, self.v_back_prepose
                )

        elif self.phase == self.ROTATE_TO_SLOT:
            yaw_err = ang_norm(yaw_slot_back - ryaw)
            d_debug = math.hypot(rx - sx, ry - sy)  # 센터까지 거리
            if abs(math.degrees(yaw_err)) <= self.yaw_tol_slot_deg:
                self.set_phase(self.REVERSE_TO_SLOT)
                v, w = 0.0, 0.0
                self.prev_xL = None
            else:
                v, w = self.rot_ctrl(yaw_err)

        elif self.phase == self.REVERSE_TO_SLOT:
            v, w, xL, yL, yaw_err = self.reverse_slot_ctrl(
                rx, ry, ryaw, sx, sy, yaw_slot_back
            )
            d_debug = xL  # 슬롯 축 방향 오차

            # 중심 통과(+→−) 즉시 브레이크 & 홀드
            if self.prev_xL is not None and (self.prev_xL > 0.0) and (xL <= 0.0):
                v, w = 0.0, 0.0
                if self.done_since is None:
                    self.done_since = rospy.Time.now()

            # 완료 체크
            if self.done_check(xL, yL, yaw_err):
                if self.done_since is None:
                    self.done_since = rospy.Time.now()
                if (rospy.Time.now() - self.done_since).to_sec() >= self.hold_on_done_s:
                    self.set_phase(self.DONE)
                    v, w = 0.0, 0.0
            else:
                # 중심 통과 이벤트가 아닌 경우에는 done_since 리셋
                if not (self.prev_xL is not None and self.prev_xL > 0.0 and xL <= 0.0):
                    self.done_since = None

            self.prev_xL = xL

        elif self.phase == self.DONE:
            v, w = 0.0, 0.0
            d_debug = 0.0

        else:
            v, w = 0.0, 0.0
            d_debug = -1.0

        # ----- 뒤쪽 장애물일 때 후진 정지 (DWA 스타일 gating) -----
        if self.obstacle_blocked and v < 0.0:
            rospy.loginfo_throttle(
                1.0,
                "[back_in_parking] rear obstacle active: v=%.2f -> stop, d=%.2f",
                v,
                self.obstacle_min_d if self.obstacle_min_d is not None else -1.0,
            )
            v = 0.0
            w = 0.0

        twist.linear.x  = v
        twist.angular.z = w
        self.pub.publish(twist)

        rospy.loginfo_throttle(
            0.5,
            "[back_in_parking] PH=%s  v=%.2f w=%.2f  locked=%s  obst=%s d=%.2f",
            self.phase, v, w,
            self.slot_locked,
            "True" if self.obstacle_blocked else "False",
            d_debug,
        )


def main():
    rospy.init_node("parking_controller")
    BackInParking()
    rospy.spin()


if __name__ == "__main__":
    main()
