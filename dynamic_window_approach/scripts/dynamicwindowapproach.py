#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWA local planner — stable path tracking for A* overlap/crossing segments

Fixes vs. original:
  • Ignore /astar/path if it hasn't actually changed (hash signature).
  • Track along-path progress with arc-length s (monotonic, jitter tolerant).
  • Pure-Pursuit style target: point at s + lookahead_distance on the polyline.
  • Progress cost to discourage backwards/sideways motions near junctions.
  • Rotate-only mode with clean hysteresis and path-tangent gating.
  • Minimal forward obstacle stop using front ROI from PointCloud2.

I/O topics kept the same for drop-in replacement.
"""

import math
import numpy as np
import rospy
import tf.transformations as transformations

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from dynamic_window_approach.msg import server_to_robot

# ------------------------------------- utils -------------------------------------
def angdiff(a, b):
    """Wrap-safe angle diff a-b in [-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))

# ----------------------------------- DWA node -----------------------------------
class DWAControl:
    def __init__(self):
        # ===== Dynamics / Sampling =====
        self.max_speed = 0.6
        self.min_speed = -0.4
        self.low_speed = 0.4
        self.max_yaw_rate = math.radians(180.0)
        self.max_accel = 0.1
        self.max_delta_yaw_rate = math.radians(450.0)
        self.v_resolution = 0.00125
        self.yaw_rate_resolution = math.radians(10.0)
        self.dt = 0.1
        self.predict_time = 3.0

        # ===== Costs =====
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.progress_cost_gain = rospy.get_param("~progress_cost_gain", 2.0)
        self.lateral_cost_gain = rospy.get_param("~lateral_cost_gain", 0.0)  # optional
        self.robot_stuck_flag_cons = 0.001

        self.server_cmd_drive_mode = 0

        # ===== Obstacle stop (front ROI) =====
        self.cloud_topic = rospy.get_param("~pointcloud_topic", "/ouster/points")
        self.stop_distance = rospy.get_param("~stop_distance", 0.5)
        self.stop_width = rospy.get_param("~stop_width", 0.3)   # total width (|y|<=width/2)
        self.min_z = rospy.get_param("~min_z", -0.3)
        self.max_z = rospy.get_param("~max_z", 1.5)
        self.cloud_downsample = rospy.get_param("~cloud_downsample", 4)
        self.block_on_count = rospy.get_param("~block_on_count", 2)
        self.block_off_count = rospy.get_param("~block_off_count", 3)
        # Parameters to ignore sensor self-returns (small deadzone around sensor)
        self.self_exclude_forward = rospy.get_param("~self_exclude_forward", 0.15)  # m forward
        self.self_exclude_side = rospy.get_param("~self_exclude_side", 0.15)        # m side
        self.self_exclude_z_min = rospy.get_param("~self_exclude_z_min", -0.2)     # m below lidar
        self.self_exclude_z_max = rospy.get_param("~self_exclude_z_max", 0.2)      # m above lidar
        self.obstacle_near = False
        self._blk_on = 0
        self._blk_off = 0

        # ===== Rotate-only mode =====
        self.rotate_only_deg = rospy.get_param("~rotate_only_deg", 80.0)
        self.rotate_exit_deg = rospy.get_param("~rotate_exit_deg", 30.0)
        self.rotate_kp = rospy.get_param("~rotate_kp", 2.0)
        self.rotate_w_max_deg = rospy.get_param("~rotate_w_max_deg", 90.0)
        self.rotate_ok_count = rospy.get_param("~rotate_ok_count", 3)
        self.rotate_max_spin_deg = rospy.get_param("~rotate_max_spin_deg", 420.0)
        self.rotate_max_time_s = rospy.get_param("~rotate_max_time_s", 6.0)
        self._ROT_HIGH = math.radians(self.rotate_only_deg)
        self._ROT_LOW  = math.radians(self.rotate_exit_deg)
        self._ROT_WMAX = math.radians(self.rotate_w_max_deg)
        self._rot_mode = False
        self._rot_yaw_target = None
        self._rot_prev_yaw = None
        self._rot_accum = 0.0
        self._rot_ok = 0
        self._rot_start_time = None

        # ===== Path tracking (s-based) =====
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.0)
        self.back_jitter_m = rospy.get_param("~back_jitter_m", 0.3)
        self.goal_thresh_m = rospy.get_param("~goal_thresh_m", 0.1)
        self.final_approach_window_m = rospy.get_param("~final_approach_window_m", 2.5)
        self.final_speed_k = rospy.get_param("~final_speed_k", 0.6)
        self.final_speed_min = rospy.get_param("~final_speed_min", 0.12)
        self.lat_goal_slop = rospy.get_param("~lat_goal_slop", 0.6)
        self.current_point_search_radius_m = 5.0  # legacy (kept for /traj_info)

        # Internal path buffers
        self.path_msg = None
        self.path_sig = None
        self.path_pts = []          # [(x,y), ...]
        self.seg_lens = []          # [len_i]
        self.cum_len = [0.0]        # [0, s1, s2, ...]
        self.s_total = 0.0
        self.s_cur = 0.0
        self.s_prev_published_idx = 0

        self.reach_goal_flag = False
        self.prev_goal_flag = False

        # ===== State =====
        self.current_pose = Odometry()
        self.warm_up_flag = False

        # ===== ROS I/O =====
        self.sub_path = rospy.Subscriber('/astar/path', Path, self.path_callback)
        self.sub_pose = rospy.Subscriber('lio_localizer/odometry/optimization', Odometry, self.pose_callback)
        self.sub_server_cmd = rospy.Subscriber("server_to_robot_topic", server_to_robot, self.server_to_robot_callback)
        self.sub_cloud = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.target_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.current_pub = rospy.Publisher('visualization_marker_2', Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('predicted_trajectory', Marker, queue_size=10)
        self.traj_info_pub = rospy.Publisher('/traj_info', Float32MultiArray, queue_size=10)

    # ------------------------------- obstacle stop -------------------------------
    def cloud_callback(self, msg):
        try:
            near = False
            half_w = 0.5 * self.stop_width
            i = 0
            for pt in point_cloud2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
                i += 1
                if self.cloud_downsample > 1 and (i % self.cloud_downsample != 0):
                    continue
                x, y, z = pt
                if x <= 0.0:
                    continue
                if z < self.min_z or z > self.max_z:
                    continue
                if abs(y) > half_w:
                    continue
                # ignore likely self-returns in a small deadzone around the sensor
                try:
                    if (x <= self.self_exclude_forward and abs(y) <= self.self_exclude_side and
                        z >= self.self_exclude_z_min and z <= self.self_exclude_z_max):
                        continue
                except Exception:
                    # be defensive: if params missing or invalid, don't crash
                    pass
                if (x*x + y*y) <= (self.stop_distance * self.stop_distance):
                    near = True
                    break
            if near:
                self._blk_on += 1; self._blk_off = 0
            else:
                self._blk_off += 1; self._blk_on = 0
            if not self.obstacle_near and self._blk_on >= self.block_on_count:
                self.obstacle_near = True
                rospy.logwarn("Obstacle CLOSE: stopping (<= %.2fm)", self.stop_distance)
            elif self.obstacle_near and self._blk_off >= self.block_off_count:
                self.obstacle_near = False
                rospy.loginfo("Obstacle cleared: resuming")
        except Exception as e:
            rospy.logwarn("cloud_callback error: %s", str(e))

    # ------------------------------- rotate-only --------------------------------
    def rotate_only_enter(self, cur_yaw, desired_yaw):
        self._rot_mode = True
        self._rot_yaw_target = desired_yaw
        self._rot_prev_yaw = cur_yaw
        self._rot_accum = 0.0
        self._rot_ok = 0
        self._rot_start_time = rospy.Time.now()
        rospy.loginfo("Rotate-only ENTER: target %.1f°", math.degrees(self._rot_yaw_target))

    def rotate_only_step(self, cur_yaw):
        dyaw = angdiff(cur_yaw, self._rot_prev_yaw)
        self._rot_accum += abs(dyaw)
        self._rot_prev_yaw = cur_yaw
        err = angdiff(self._rot_yaw_target, cur_yaw)
        w_cmd = max(-self._ROT_WMAX, min(self._ROT_WMAX, self.rotate_kp * err))
        u = [0.0, w_cmd]
        if abs(err) < self._ROT_LOW:
            self._rot_ok += 1
        else:
            self._rot_ok = 0
        time_in = (rospy.Time.now() - self._rot_start_time).to_sec()
        if (self._rot_ok >= self.rotate_ok_count or
            self._rot_accum > math.radians(self.rotate_max_spin_deg) or
            time_in > self.rotate_max_time_s):
            self._rot_mode = False
            rospy.loginfo("Rotate-only EXIT: err=%.1f°, accum=%.1f°, t=%.1fs",
                          math.degrees(err), math.degrees(self._rot_accum), time_in)
            return None, True
        return u, False

    # ------------------------------ path handling --------------------------------
    def _path_signature(self, path_msg):
        if not path_msg or not path_msg.poses:
            return None
        n = len(path_msg.poses)
        p0 = path_msg.poses[0].pose.position
        p1 = path_msg.poses[-1].pose.position
        return (n, round(p0.x,3), round(p0.y,3), round(p1.x,3), round(p1.y,3))

    def _rebuild_path_geometry(self):
        self.path_pts = []
        for ps in self.path_msg.poses:
            p = ps.pose.position
            self.path_pts.append((p.x, p.y))
        # seg lengths & cumulative
        self.seg_lens = []
        self.cum_len = [0.0]
        s = 0.0
        for i in range(len(self.path_pts)-1):
            dx = self.path_pts[i+1][0] - self.path_pts[i][0]
            dy = self.path_pts[i+1][1] - self.path_pts[i][1]
            L = math.hypot(dx, dy)
            self.seg_lens.append(L)
            s += L
            self.cum_len.append(s)
        self.s_total = s
        self.s_cur = 0.0
        self.reach_goal_flag = False
        self.prev_goal_flag = False

    def path_callback(self, path_msg):
        sig = self._path_signature(path_msg)
        if sig is not None and sig == self.path_sig:
            # identical path → ignore to avoid re-initialization jitters
            return
        self.path_sig = sig
        self.path_msg = path_msg
        if len(path_msg.poses) < 2:
            return
        self._rebuild_path_geometry()

    def pose_callback(self, msg):
        self.current_pose = msg

    def server_to_robot_callback(self, msg):
        self.server_cmd_drive_mode = msg.Cmd_drive_mode
        if msg.Cmd_use_vel_control:
            if self.max_speed != msg.Cmd_linear_velocity or self.max_yaw_rate != msg.Cmd_angular_velocity:
                self.max_speed = msg.Cmd_linear_velocity
                self.max_yaw_rate = msg.Cmd_angular_velocity
                rospy.loginfo("Updated limits: max_speed=%.3f, max_yaw_rate=%.1fdeg/s",
                              self.max_speed, math.degrees(self.max_yaw_rate))

    # ------------------------------- pure pursuit --------------------------------
    def _project_to_path(self, x, y):
        """Return (s_proj, lateral_err, seg_idx, t) where s is arc-length along path."""
        if len(self.path_pts) < 2:
            return 0.0, 0.0, 0, 0.0
        best_s = 0.0
        best_d2 = 1e18
        best_i = 0
        best_t = 0.0
        for i in range(len(self.path_pts)-1):
            x0, y0 = self.path_pts[i]
            x1, y1 = self.path_pts[i+1]
            vx, vy = x1-x0, y1-y0
            denom = vx*vx + vy*vy
            if denom < 1e-12:
                t = 0.0
                px, py = x0, y0
            else:
                t = ((x-x0)*vx + (y-y0)*vy) / denom
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                px, py = x0 + t*vx, y0 + t*vy
            d2 = (x-px)**2 + (y-py)**2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_t = t
        # arc-length at projection
        s_at_i = self.cum_len[best_i]
        s_proj = s_at_i + best_t * self.seg_lens[best_i]
        lat_err = math.sqrt(best_d2)
        return s_proj, lat_err, best_i, best_t

    def _interp_xy_tangent_at_s(self, s):
        if s <= 0.0 or len(self.path_pts) < 2:
            x, y = self.path_pts[0]
            dx = self.path_pts[1][0] - self.path_pts[0][0]
            dy = self.path_pts[1][1] - self.path_pts[0][1]
            L = math.hypot(dx, dy) + 1e-9
            return x, y, (dx/L, dy/L)
        if s >= self.s_total:
            x, y = self.path_pts[-1]
            dx = self.path_pts[-1][0] - self.path_pts[-2][0]
            dy = self.path_pts[-1][1] - self.path_pts[-2][1]
            L = math.hypot(dx, dy) + 1e-9
            return x, y, (dx/L, dy/L)
        # find segment
        # simple linear scan (path sizes are moderate). Could be binary search.
        i = 0
        while i < len(self.seg_lens) and self.cum_len[i+1] < s:
            i += 1
        ds = s - self.cum_len[i]
        L = self.seg_lens[i]
        t = 0.0 if L < 1e-9 else (ds / L)
        x0, y0 = self.path_pts[i]
        x1, y1 = self.path_pts[i+1]
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        dx = (x1 - x0); dy = (y1 - y0)
        L = math.hypot(dx, dy) + 1e-9
        return x, y, (dx/L, dy/L)

    def _update_progress_and_target(self, pose_x, pose_y, yaw):
        if not self.path_pts:
            return None, None, None, None, False, None, None
        s_proj, lat_err, idx, t = self._project_to_path(pose_x, pose_y)
        # enforce monotonic progress with tiny back jitter allowed
        if s_proj + self.back_jitter_m >= self.s_cur:
            self.s_cur = max(self.s_cur, s_proj)
        # choose target at s+lookahead
        s_target = min(self.s_total, self.s_cur + self.lookahead_distance)
        tx, ty, t_hat = self._interp_xy_tangent_at_s(s_target)
        # goal metrics
        gx, gy = self.path_pts[-1]
        dist_to_goal = math.hypot(gx - pose_x, gy - pose_y)
        arc_rem = max(0.0, self.s_total - self.s_cur)
        at_goal = (min(arc_rem, dist_to_goal) <= self.goal_thresh_m) and (lat_err <= self.lat_goal_slop)
        return (s_proj, lat_err, (tx, ty), t_hat, at_goal, dist_to_goal, arc_rem)

    # ------------------------------- dwa core ------------------------------------
    def dwa_control(self, x, goal_xy, t_hat, lat_err):
        dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal_xy, t_hat, lat_err)
        return u, trajectory

    def moving(self, x, u):
        x[2] = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
        x[0] = self.current_pose.pose.pose.position.x
        x[1] = self.current_pose.pose.pose.position.y
        x[3] = u[0]
        x[4] = u[1]
        return x

    def motion(self, x, u):
        x[2] += u[1] * self.dt
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_dynamic_window(self, x):
        Vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        Vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_delta_yaw_rate * self.dt,
              x[4] + self.max_delta_yaw_rate * self.dt]
        return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    def predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        trajectory = np.array(x)
        t = 0.0
        while t <= self.predict_time:
            x = self.motion(x, [v, y])
            trajectory = np.vstack((trajectory, x))
            t += self.dt
        return trajectory

    def calc_control_and_trajectory(self, x, dw, goal_xy, t_hat, lat_err):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
        gx, gy = goal_xy
        t_hat = np.array(t_hat)

        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):
                traj = self.predict_trajectory(x_init, v, y)
                # to-goal (heading error at final state)
                dx = gx - traj[-1, 0]
                dy = gy - traj[-1, 1]
                error_angle = math.atan2(dy, dx)
                cost_angle = angdiff(error_angle, traj[-1, 2])
                to_goal_cost = self.to_goal_cost_gain * abs(cost_angle)
                # speed cost (prefer faster)
                speed_cost = self.speed_cost_gain * (self.max_speed - traj[-1, 3])
                # progress cost (discourage moving against tangent)
                move_vec = np.array([traj[-1,0] - x[0], traj[-1,1] - x[1]])
                progress = float(np.dot(move_vec, t_hat))
                progress_cost = self.progress_cost_gain * max(0.0, -progress)
                # lateral penalty (optional): use current lat_err snapshot (cheap)
                lateral_cost = self.lateral_cost_gain * lat_err
                final_cost = to_goal_cost + speed_cost + progress_cost + lateral_cost
                if final_cost <= min_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = traj
                    if abs(best_u[0]) < self.robot_stuck_flag_cons and abs(x[3]) < self.robot_stuck_flag_cons:
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory

    # --------------------------------- helpers -----------------------------------
    def visualize_target_point(self, target_point):
        marker = Marker()
        marker.header.frame_id = "map"; marker.header.stamp = rospy.Time.now()
        marker.type = marker.SPHERE; marker.action = marker.ADD
        marker.scale.x = 0.5; marker.scale.y = 0.5; marker.scale.z = 0.0
        marker.color.a = 1.0; marker.color.g = 1.0
        marker.pose.position.x = target_point[0]; marker.pose.position.y = target_point[1]
        self.target_pub.publish(marker)

    def visualize_current_point(self, s_proj):
        # project point on path for viz
        x, y, _t = self._interp_xy_tangent_at_s(max(0.0, min(self.s_total, s_proj)))
        marker = Marker()
        marker.header.frame_id = "map"; marker.header.stamp = rospy.Time.now()
        marker.type = marker.SPHERE; marker.action = marker.ADD
        marker.scale.x = 0.5; marker.scale.y = 0.5; marker.scale.z = 0.0
        marker.color.a = 1.0; marker.color.b = 1.0
        marker.pose.position.x = x; marker.pose.position.y = y
        self.current_pub.publish(marker)

    def visualize_predicted_trajectory(self, predicted_trajectory):
        marker = Marker()
        marker.header.frame_id = "map"; marker.header.stamp = rospy.Time.now()
        marker.type = marker.LINE_STRIP; marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.color.a = 1.0; marker.color.b = 1.0
        for point in predicted_trajectory:
            p = Point(); p.x = float(point[0]); p.y = float(point[1]); p.z = 1.0
            marker.points.append(p)
        self.trajectory_pub.publish(marker)

    def publish_traj_info(self):
        msg = Float32MultiArray()
        cur_idx = 0
        # estimate an index for compatibility (segment start)
        if self.path_pts and len(self.cum_len) > 1:
            while cur_idx+1 < len(self.cum_len) and self.cum_len[cur_idx+1] < self.s_cur:
                cur_idx += 1
        msg.data = [self.s_total, self.s_cur, cur_idx, float(self.reach_goal_flag)]
        self.traj_info_pub.publish(msg)

    def get_yaw_from_quaternion(self, q):
        e = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return e[2]

    # ----------------------------------- main ------------------------------------
    def publish_drive(self, u):
        cmd = Twist(); cmd.linear.x = u[0]; cmd.angular.z = u[1]
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        rospy.loginfo("DWA node (s-tracking) started")
        x = [self.current_pose.pose.pose.position.x,
             self.current_pose.pose.pose.position.y,
             self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation),
             0.0, 0.0]
        rate = rospy.Rate(1.0/self.dt)

        while not rospy.is_shutdown():
            # obstacle stop (block forward when not rotating-only)
            if self.obstacle_near and not self._rot_mode:
                self.publish_drive([0.0, 0.0])
                rate.sleep(); continue

            if not self.path_pts:
                self.publish_drive([0.0, 0.0])
                rate.sleep(); continue

            # current pose snapshot
            yaw = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
            px  = self.current_pose.pose.pose.position.x
            py  = self.current_pose.pose.pose.position.y

            # progress / target compute
            s_proj, lat_err, target_xy, t_hat, at_goal, dist_to_goal, arc_rem = self._update_progress_and_target(px, py, yaw)
            if s_proj is None:
                self.publish_drive([0.0, 0.0])
                rate.sleep(); continue

            self.visualize_current_point(s_proj)
            self.visualize_target_point(target_xy)
            self.publish_traj_info()

            # goal handling
            self.prev_goal_flag = self.reach_goal_flag
            self.reach_goal_flag = at_goal
            if self.reach_goal_flag:
                self.publish_drive([0.0, 0.0])
                if not self.prev_goal_flag:
                    rospy.loginfo("Goal reached!")
                rate.sleep(); continue

            # rotate-only gating with path tangent
            desired = math.atan2(target_xy[1]-py, target_xy[0]-px)
            err = abs(angdiff(desired, yaw))
            # if we're roughly aligned with path tangent (forward progress), avoid entering rotate-only
            heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
            dot_forward = float(np.dot(heading_vec, np.array(t_hat)))
            if not self._rot_mode and (err > self._ROT_HIGH) and (dot_forward < 0.2):
                self.rotate_only_enter(yaw, desired)
            if self._rot_mode:
                u_rot, done = self.rotate_only_step(yaw)
                if not done:
                    x = self.moving(x, u_rot)
                    self.publish_drive(u_rot)
                    rate.sleep(); continue

            # normal DWA towards target
            u, predicted = self.dwa_control(x, target_xy, t_hat, lat_err)
            self.visualize_predicted_trajectory(predicted)
            x = self.moving(x, u)
            # final-approach speed cap
            final_window = self.final_approach_window_m
            if min(arc_rem, dist_to_goal) <= final_window:
                v_cap = max(self.final_speed_min, min(self.max_speed, self.final_speed_k * max(dist_to_goal, 0.0)))
            else:
                v_cap = self.max_speed
            # low-speed clamp with cap in direction of chosen v sign
            u_cmd = list(u)
            if u_cmd[0] >= 0.0:
                u_cmd[0] = min(v_cap, max(self.low_speed, u_cmd[0]))
            else:
                u_cmd[0] = -min(v_cap, max(self.low_speed, -u_cmd[0]))
            self.publish_drive(u_cmd)
            rate.sleep()

# -------------------------------- entry point ------------------------------------
def main():
    rospy.init_node('dwa_node', anonymous=True)
    dwa = DWAControl()
    dwa.run()

if __name__ == "__main__":
    main()
