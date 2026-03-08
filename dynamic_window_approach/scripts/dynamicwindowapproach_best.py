#!/usr/bin/env python
# license removed for brevity

import rospy
from nav_msgs.msg import Path, Odometry
import tf.transformations as transformations
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from dynamic_window_approach.msg import server_to_robot
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

import math
import numpy as np


def angdiff(a, b):
    """Wrap-safe angle difference a-b in [-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class DWAControl:
    def __init__(self):
        # ====== Original parameters (kept as-is) ======
        self.max_speed = 0.6  # [m/s]
        # self.max_speed = 0.8  # [m/s]
        self.min_speed = -0.4  # [m/s]
        self.low_speed = 0.4  # [m/s]
        # self.min_speed = -0.2  # [m/s]
        self.max_yaw_rate = 180.0 * math.pi / 180.0  # [rad/s]
        # self.max_yaw_rate = 3600.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.1  # [m/ss]
        # self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 450.0 * math.pi / 180.0  # [rad/ss]
        # self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.00125  # [m/s]
        self.yaw_rate_resolution = 10 * math.pi / 180.0  # [rad/s]
        # self.v_resolution = 0.01  # [m/s]
        # self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # prevent robot stuck

        self.server_cmd_drive_mode = 0
        
        # ====== Subscribers / Publishers (original) ======
        self.sub_path = rospy.Subscriber('/astar/path', Path, self.path_callback)
        self.sub_pose = rospy.Subscriber('lio_localizer/odometry/optimization', Odometry, self.pose_callback)
        self.sub_server_cmd = rospy.Subscriber("server_to_robot_topic", server_to_robot, self.server_to_robot_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.target_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.current_pub = rospy.Publisher('visualization_marker_2', Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('predicted_trajectory', Marker, queue_size=10)
        self.traj_info_pub = rospy.Publisher('/traj_info', Float32MultiArray, queue_size=10)

        # ====== Obstacle stop (NEW, minimal) ======
        # 전방 ROI에서 stop_distance 이내 점이 있으면 정지 (회전 전용 모드일 때는 회전만 허용)
        self.cloud_topic = rospy.get_param("~pointcloud_topic", "/ouster/points")
        self.stop_distance = rospy.get_param("~stop_distance", 0.7)
        self.stop_width = rospy.get_param("~stop_width", 0.5)  # total width (|y| <= width/2)
        self.min_z = rospy.get_param("~min_z", -0.3)
        self.max_z = rospy.get_param("~max_z", 1.5)
        self.cloud_downsample = rospy.get_param("~cloud_downsample", 4)
        self.block_on_count = rospy.get_param("~block_on_count", 2)
        self.block_off_count = rospy.get_param("~block_off_count", 3)

        self.obstacle_near = False
        self._blk_on = 0
        self._blk_off = 0
        self.sub_cloud = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)

        # ====== Rotate-only mode (NEW, robust) ======
        # 120° 이상이면 진입, 15° 미만 연속 N틱이면 해제. P제어로 감속 회전. 누적 회전/시간 세이프가드.
        self.rotate_only_deg = rospy.get_param("~rotate_only_deg", 120.0)    # enter threshold
        self.rotate_exit_deg = rospy.get_param("~rotate_exit_deg", 15.0)     # exit threshold (small)
        self.rotate_kp = rospy.get_param("~rotate_kp", 2.0)                  # rad/s per rad
        self.rotate_w_max_deg = rospy.get_param("~rotate_w_max_deg", 120.0)  # clamp
        self.rotate_ok_count = rospy.get_param("~rotate_ok_count", 3)        # consecutive ticks
        self.rotate_max_spin_deg = rospy.get_param("~rotate_max_spin_deg", 420.0)  # max accumulated rotation
        self.rotate_max_time_s = rospy.get_param("~rotate_max_time_s", 6.0)        # max time inside mode

        self._ROT_HIGH = math.radians(self.rotate_only_deg)
        self._ROT_LOW  = math.radians(self.rotate_exit_deg)
        self._ROT_WMAX = math.radians(self.rotate_w_max_deg)

        self._rot_mode = False
        self._rot_yaw_target = None
        self._rot_prev_yaw = None
        self._rot_accum = 0.0
        self._rot_ok = 0
        self._rot_start_time = None
        
        # ====== State (original) ======
        self.current_pose = Odometry()
        self.path_msg = None
        self.lookahead_distance = 1.0
        self.current_point_search_radius_m = 5.0
        self.path_callback_flag = False
        self.path_initialize_flag = False
        self.warm_up_flag = False
        self.goal_point = []
        self.target_point = []
        self.start_point = []
        self.current_point_idx = 0
        self.path_length_list = [0]
        self.reach_goal_flag = False
        self.prev_goal_flag = False

    # ================= Obstacle proximity =================
    def cloud_callback(self, msg):
        try:
            near = False
            half_w = self.stop_width * 0.5
            i = 0
            for pt in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                i += 1
                if self.cloud_downsample > 1 and (i % self.cloud_downsample != 0):
                    continue
                x, y, z = pt
                if x <= 0.0:  # 전방만
                    continue
                if z < self.min_z or z > self.max_z:
                    continue
                if abs(y) > half_w:
                    continue
                if (x * x + y * y) <= (self.stop_distance * self.stop_distance):
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

    # ================= Rotate-only helpers =================
    def rotate_only_enter(self, cur_yaw, desired_yaw):
        self._rot_mode = True
        self._rot_yaw_target = desired_yaw  # 목표 각도 고정
        self._rot_prev_yaw = cur_yaw
        self._rot_accum = 0.0
        self._rot_ok = 0
        self._rot_start_time = rospy.Time.now()
        rospy.loginfo("Rotate-only ENTER: target %.1f°", math.degrees(self._rot_yaw_target))

    def rotate_only_step(self, cur_yaw):
        # 누적 회전량 업데이트(절댓값 합)
        dyaw = angdiff(cur_yaw, self._rot_prev_yaw)
        self._rot_accum += abs(dyaw)
        self._rot_prev_yaw = cur_yaw

        # P 제어 각속도 (오차 작아지면 자동 감속)
        err = angdiff(self._rot_yaw_target, cur_yaw)
        w_cmd = max(-self._ROT_WMAX, min(self._ROT_WMAX, self.rotate_kp * err))
        u = [0.0, w_cmd]

        # 해제 조건: 작은 오차 연속 tick 만족
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
            return None, True  # 종료
        return u, False

    # ================= Original callbacks/utils =================
    def path_callback(self, path_msg):
        self.path_msg = path_msg
        self.path_callback_flag = True
        self.path_initialize_flag = False
        self.warm_up_flag = False

        self.path_length_list = [0]
        total_length = 0.0
        if len(path_msg.poses) > 1:
            for i in range(1, len(path_msg.poses)):
                prev = path_msg.poses[i - 1].pose.position
                curr = path_msg.poses[i].pose.position
                dist = math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2 + (curr.z - prev.z) ** 2)
                total_length += dist
                self.path_length_list.append(total_length)
   
    def pose_callback(self, msg):
        self.current_pose = msg

    def server_to_robot_callback(self, msg):
        self.server_cmd_drive_mode = msg.Cmd_drive_mode
        if msg.Cmd_use_vel_control:
            if self.max_speed != msg.Cmd_linear_velocity or self.max_yaw_rate != msg.Cmd_angular_velocity:
                self.max_speed = msg.Cmd_linear_velocity
                self.max_yaw_rate = msg.Cmd_angular_velocity
                print("self.max_speed: ", self.max_speed, ", self.max_yaw_rate: ", self.max_yaw_rate)

    def publish_drive(self, u):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = u[0]
        cmd_vel_msg.angular.z = u[1]
        self.cmd_vel_pub.publish(cmd_vel_msg)
    
    def publish_traj_info(self):
        traj_info_msg = Float32MultiArray()
        traj_info = [self.path_length_list[-1],
                     self.path_length_list[self.current_point_idx],
                     self.current_point_idx,
                     self.reach_goal_flag]
        traj_info_msg.data = traj_info
        self.traj_info_pub.publish(traj_info_msg)
        
    def visualize_target_point(self, target_point):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.pose.position.x = target_point[0]
        marker.pose.position.y = target_point[1]
        self.target_pub.publish(marker)
        
    def visualize_current_point(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0
        marker.color.a = 1.0
        marker.color.b = 1.0
        marker.pose.position.x = self.path_msg.poses[self.current_point_idx].pose.position.x
        marker.pose.position.y = self.path_msg.poses[self.current_point_idx].pose.position.y
        self.current_pub.publish(marker)
        
    def visualize_predicted_trajectory(self, predicted_trajectory):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        for point in predicted_trajectory:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 1.0  
            marker.points.append(p)
        self.trajectory_pub.publish(marker)

    # ================= Original DWA core =================
    def dwa_control(self, x, goal):
        dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal)
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
        Vs = [self.min_speed, self.max_speed,
              -self.max_yaw_rate, self.max_yaw_rate]
        Vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_delta_yaw_rate * self.dt,
              x[4] + self.max_delta_yaw_rate * self.dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw
   
    def predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v,y])
            trajectory = np.vstack((trajectory, x))
            time += self.dt
        return trajectory
   
    def calc_control_and_trajectory(self, x, dw, goal):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
       
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):
                trajectory = self.predict_trajectory(x_init, v, y)
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                final_cost = to_goal_cost + speed_cost
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v,y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.robot_stuck_flag_cons and abs(x[3]) < self.robot_stuck_flag_cons:
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory
   
    def calc_to_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost
   
    def get_yaw_from_quaternion(self, quaternion):
        euler = transformations.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return euler[2]
   
    def reached_goal(self, x, goal):
        distance_threshold = 1.0
        distance_to_goal = math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2)
        return distance_to_goal < distance_threshold

    # ================= Main loop (original + obstacle stop + robust rotate-only) =================
    def run(self):
        print("start!!")
        x = [self.current_pose.pose.pose.position.x,
             self.current_pose.pose.pose.position.y,
             self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation),
             0.0, 0.0]
        while not rospy.is_shutdown():
            rospy.sleep(0.01)

            # ----- Obstacle stop gate -----
            # 회전 전용 모드가 아닐 때만 정지. 회전 전용 모드일 땐 전진 금지이므로 회전만 허용.
            if self.obstacle_near and not self._rot_mode:
                self.publish_drive([0.0, 0.0])
                continue

            if self.path_callback_flag is True:
                self.path_callback_flag = False
                # Goal point reset
                self.goal_point = [self.path_msg.poses[-1].pose.position.x,
                                   self.path_msg.poses[-1].pose.position.y]
                
                # Current point reset
                coarse_current_point_idx = -1
                for i in range(len(self.path_msg.poses)):
                    point_distance = math.hypot(
                        x[0] - self.path_msg.poses[i].pose.position.x,
                        x[1] - self.path_msg.poses[i].pose.position.y)
                    if point_distance < 5:
                        coarse_current_point_idx = i
                        break
                
                min_point_distance = float("inf")
                min_point_idx = -1
                start_i = max(0, coarse_current_point_idx)  # 안전 가드
                for i in range(start_i, len(self.path_msg.poses)):
                    point_distance = math.hypot(
                        x[0] - self.path_msg.poses[i].pose.position.x,
                        x[1] - self.path_msg.poses[i].pose.position.y)
                    if point_distance < min_point_distance:
                        min_point_distance = point_distance
                        min_point_idx = i
                    if point_distance > 8:
                        break
                    
                self.current_point_idx = min_point_idx
                self.start_point = [
                    self.path_msg.poses[min_point_idx].pose.position.x, 
                    self.path_msg.poses[min_point_idx].pose.position.y
                ]
                self.path_initialize_flag = True
            
            if self.path_initialize_flag is True:
                # update current_point_idx around previous index
                min_point_distance = float("inf")
                min_point_idx = -1
                prev_current_point_idx = self.current_point_idx
                for i in range(prev_current_point_idx, len(self.path_msg.poses)):
                    point_distance = math.hypot(
                        x[0] - self.path_msg.poses[i].pose.position.x,
                        x[1] - self.path_msg.poses[i].pose.position.y)
                    if point_distance < min_point_distance:
                        min_point_distance = point_distance
                        min_point_idx = i
                    if point_distance > self.current_point_search_radius_m:
                        break
                self.current_point_idx = min_point_idx
                self.visualize_current_point()
                self.publish_traj_info()
                                
                # DWA control with target point
                if True:
                    self.prev_goal_flag = self.reach_goal_flag
                    self.reach_goal_flag = self.reached_goal(x, self.goal_point)
                    if self.reach_goal_flag:
                        self.publish_drive([0.0, 0.0])
                        if self.prev_goal_flag is False:
                            print("Goal reached!")
                    elif self.target_point == self.goal_point:
                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        target_speed = u[0]
                        if target_speed > 0:
                            u[0] = self.low_speed
                        else:
                            u[0] = -self.low_speed
                        self.visualize_predicted_trajectory(predicted_trajectory)
                        x = self.moving(x, u)
                        self.publish_drive(u)
                        continue
                    elif self.warm_up_flag is False:
                        # Update lookahead point
                        distance = 0.0
                        lookahead_point_idx = None
                        for i in range(self.current_point_idx, len(self.path_msg.poses) - 1):
                            if distance < self.lookahead_distance:
                                dx = self.path_msg.poses[i+1].pose.position.x - self.path_msg.poses[i].pose.position.x
                                dy = self.path_msg.poses[i+1].pose.position.y - self.path_msg.poses[i].pose.position.y
                                distance += math.hypot(dx, dy)
                                lookahead_point_idx = i + 1
                            else:
                                break
                        if lookahead_point_idx is None:
                            lookahead_point_idx = min(self.current_point_idx + 1, len(self.path_msg.poses) - 1)
                        self.target_point = [
                            self.path_msg.poses[lookahead_point_idx].pose.position.x, 
                            self.path_msg.poses[lookahead_point_idx].pose.position.y
                        ]
                        self.visualize_target_point(self.target_point)

                        # ====== Rotate-only gating (robust) ======
                        yaw = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
                        dx = self.target_point[0] - x[0]; dy = self.target_point[1] - x[1]
                        desired = math.atan2(dy, dx)
                        err = angdiff(desired, yaw)

                        if not self._rot_mode and abs(err) > self._ROT_HIGH:
                            self.rotate_only_enter(yaw, desired)

                        if self._rot_mode:
                            u_rot, done = self.rotate_only_step(yaw)
                            if done:
                                pass  # 다음 틱에 DWA 전진 재개
                            else:
                                x = self.moving(x, u_rot)
                                self.publish_drive(u_rot)
                                continue  # 이번 주기엔 전진 스킵

                        # ====== Normal DWA ======
                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        self.visualize_predicted_trajectory(predicted_trajectory)
                        x = self.moving(x, u)
                        target_speed = u[0]
                        if target_speed > 0:
                            u[0] = self.low_speed
                        else:
                            u[0] = -self.low_speed
                        self.publish_drive(u)

                        if self.path_length_list[self.current_point_idx] > self.current_point_search_radius_m:
                            self.warm_up_flag = True

                    else:
                        # Update lookahead point
                        distance = 0.0
                        lookahead_point_idx = None
                        for i in range(self.current_point_idx, len(self.path_msg.poses) - 1):
                            if distance < self.lookahead_distance:
                                dx = self.path_msg.poses[i+1].pose.position.x - self.path_msg.poses[i].pose.position.x
                                dy = self.path_msg.poses[i+1].pose.position.y - self.path_msg.poses[i].pose.position.y
                                distance += math.hypot(dx, dy)
                                lookahead_point_idx = i + 1
                            else:
                                break
                        if lookahead_point_idx is None:
                            lookahead_point_idx = min(self.current_point_idx + 1, len(self.path_msg.poses) - 1)
                        self.target_point = [
                            self.path_msg.poses[lookahead_point_idx].pose.position.x, 
                            self.path_msg.poses[lookahead_point_idx].pose.position.y
                        ]
                        self.visualize_target_point(self.target_point)

                        # ====== Rotate-only gating (robust) ======
                        yaw = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
                        dx = self.target_point[0] - x[0]; dy = self.target_point[1] - x[1]
                        desired = math.atan2(dy, dx)
                        err = angdiff(desired, yaw)

                        if not self._rot_mode and abs(err) > self._ROT_HIGH:
                            self.rotate_only_enter(yaw, desired)

                        if self._rot_mode:
                            u_rot, done = self.rotate_only_step(yaw)
                            if done:
                                pass
                            else:
                                x = self.moving(x, u_rot)
                                self.publish_drive(u_rot)
                                continue

                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        self.visualize_predicted_trajectory(predicted_trajectory)
                        x = self.moving(x, u)
                        self.publish_drive(u)
                else:
                    self.publish_drive([0.0, 0.0])


def main():
    rospy.init_node('dwa_node', anonymous=True) 
    dwa = DWAControl()
    dwa.run()

if __name__ == "__main__":
    main()
