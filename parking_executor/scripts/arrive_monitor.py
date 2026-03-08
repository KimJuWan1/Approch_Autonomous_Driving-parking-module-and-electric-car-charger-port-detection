#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, re, rospy, tf2_ros
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Log

def yaw_of(q):  # quaternion → yaw (roll/pitch≈0 가정)
    return math.atan2(2*(q.w*q.z), 1 - 2*(q.z*q.z))

class ArriveMonitor:
    def __init__(self):
        # --- 기존 파라미터 ---
        self.frame = rospy.get_param("~frame_id", "map")
        self.path_topic = rospy.get_param("~path_topic", "/astar/path")
        self.pos_tol = float(rospy.get_param("~pos_tol", 0.6))
        self.yaw_tol = math.radians(float(rospy.get_param("~yaw_tol_deg", 20.0)))
        self.hold_s = float(rospy.get_param("~hold_s", 0.5))

        # --- 추가: 로그 기반 도착 감지 옵션 ---
        self.use_rosout_goal = bool(rospy.get_param("~use_rosout_goal", True))
        self.rosout_goal_regex = re.compile(rospy.get_param("~rosout_goal_regex", r"Goal reached!?"), re.IGNORECASE)
        # 특정 노드에서만 받은 로그를 인정 (예: dynamic_window_approach)
        self.rosout_node_filter = rospy.get_param("~rosout_node_filter", "dynamic_window_approach")  # 부분문자열 매칭
        self.rosout_hold_s = float(rospy.get_param("~rosout_hold_s", 1.0))  # 로그 감지 후 True 유지 시간

        # 새 경로가 '새 목표'로 보일 때 arrived False로 리셋할 기준
        self.reset_pos_delta_m = float(rospy.get_param("~reset_pos_delta_m", 0.3))
        self.reset_yaw_delta = math.radians(float(rospy.get_param("~reset_yaw_delta_deg", 15.0)))

        self.goal = None   # (x,y,yaw)
        self.arrived_pub = rospy.Publisher("/nav/arrived", Bool, queue_size=1, latch=True)
        rospy.Subscriber(self.path_topic, Path, self.cb_path, queue_size=1)

        self.tfbuf = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)
        self.last_true = None
        self.force_arrived_until = rospy.Time(0)  # 로그 감지 등으로 강제 True 유지하는 데드라인

        if self.use_rosout_goal:
            rospy.Subscriber("/rosout_agg", Log, self.cb_rosout, queue_size=50)

        rospy.Timer(rospy.Duration(0.05), self.on_timer)  # 20 Hz
        rospy.loginfo("[arrive_monitor] watching %s (frame=%s, rosout=%s)",
                      self.path_topic, self.frame, self.use_rosout_goal)

        # 초기 상태는 False로
        self.arrived_pub.publish(Bool(data=False))

    def cb_path(self, msg: Path):
        if not msg.poses:
            return
        g = msg.poses[-1].pose
        if len(msg.poses) >= 2:
            p0, p1 = msg.poses[-2].pose.position, g.position
            yaw = math.atan2(p1.y - p0.y, p1.x - p0.x)
        else:
            yaw = yaw_of(g.orientation)

        new_goal = (g.position.x, g.position.y, yaw)

        # 새 목표로 판단되면 arrived 리셋
        if self.goal is not None:
            gx, gy, gyaw = self.goal
            nx, ny, nyaw = new_goal
            if math.hypot(nx - gx, ny - gy) > self.reset_pos_delta_m or \
               abs((nyaw - gyaw + math.pi) % (2*math.pi) - math.pi) > self.reset_yaw_delta:
                self.last_true = None
                self.force_arrived_until = rospy.Time(0)
                self.arrived_pub.publish(Bool(data=False))
                rospy.loginfo("[arrive_monitor] goal changed → arrived=False")

        self.goal = new_goal

    def cb_rosout(self, msg: Log):
        # 노드 이름 필터(부분 문자열) 체크
        if self.rosout_node_filter and self.rosout_node_filter not in msg.name:
            return
        # 메시지 텍스트 매칭
        if not self.rosout_goal_regex.search(msg.msg or ""):
            return

        now = rospy.Time.now()
        self.force_arrived_until = now + rospy.Duration(self.rosout_hold_s)
        self.last_true = now  # 즉시 유지조건 충족으로 간주
        self.arrived_pub.publish(Bool(data=True))
        rospy.loginfo("[arrive_monitor] rosout '%s' from %s → arrived=True (hold %.1fs)",
                      msg.msg.strip(), msg.name, self.rosout_hold_s)

    def robot_pose(self):
        try:
            tf = self.tfbuf.lookup_transform(self.frame, "base_link", rospy.Time(0), rospy.Duration(0.2))
            t = tf.transform.translation
            return (t.x, t.y, yaw_of(tf.transform.rotation))
        except Exception:
            return None

    def on_timer(self, _):
        # 로그에 의해 강제 True 유지 중이면 건드리지 않음(마지막 발행 True가 latched로 남음)
        if rospy.Time.now() < self.force_arrived_until:
            return

        if self.goal is None:
            return
        r = self.robot_pose()
        if r is None:
            return

        rx, ry, ryaw = r
        gx, gy, gyaw = self.goal
        dist = math.hypot(rx - gx, ry - gy)
        dyaw = (ryaw - gyaw + math.pi) % (2*math.pi) - math.pi
        ok = (dist <= self.pos_tol) and (abs(dyaw) <= self.yaw_tol)
        now = rospy.Time.now()

        if ok and self.last_true is None:
            self.last_true = now

        if ok and (now - self.last_true).to_sec() >= self.hold_s:
            self.arrived_pub.publish(Bool(data=True))
        elif not ok:
            self.last_true = None
            self.arrived_pub.publish(Bool(data=False))

if __name__ == "__main__":
    rospy.init_node("arrive_monitor")
    ArriveMonitor()
    rospy.spin()
