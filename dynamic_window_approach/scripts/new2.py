#!/usr/bin/env python
# license removed for brevity

import rospy
from nav_msgs.msg import Path, Odometry
import tf.transformations as transformations
from geometry_msgs.msg import Twist, Point, PoseStamped
from visualization_msgs.msg import Marker
from dynamic_window_approach.msg import server_to_robot
from std_msgs.msg import Float32MultiArray

import math
import numpy as np
from scipy.interpolate import CubicSpline

class DWAControl:
    def __init__(self):
        self.max_speed = 0.6  # [m/s]
        self.min_speed = -0.4  # [m/s]
        self.low_speed = 0.4  # [m/s]
        self.max_yaw_rate = 180.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.1  # [m/ss]
        self.max_delta_yaw_rate = 450.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.00125  # [m/s]
        self.yaw_rate_resolution = 10 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s]
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001

        self.server_cmd_drive_mode = 0
        self.sub_path = rospy.Subscriber('/astar/path', Path, self.path_callback)
        self.spline_path_pub = rospy.Publisher('/spline_path', Path, queue_size=1, latch=True)
        self.spline_marker_pub = rospy.Publisher('/spline_path_marker', Marker, queue_size=1, latch=True)
        self.sub_pose = rospy.Subscriber('lio_localizer/odometry/optimization', Odometry, self.pose_callback)
        self.sub_server_cmd = rospy.Subscriber("server_to_robot_topic", server_to_robot, self.server_to_robot_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.target_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.current_pub = rospy.Publisher('visualization_marker_2', Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('predicted_trajectory', Marker, queue_size=10)
        self.traj_info_pub = rospy.Publisher('/traj_info', Float32MultiArray, queue_size=10)

        self.current_pose = Odometry()
        self.path_msg = None
        self.lookahead_distance = 3.0
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
        
        # Cubic spline 관련 변수들
        self.spline_x = None
        self.spline_y = None
        self.spline_points = None
        self.spline_length = None

    def path_callback(self, path_msg):
        self.path_msg = path_msg
        self.path_callback_flag = True
        self.path_initialize_flag = False
        self.warm_up_flag = False

        if len(path_msg.poses) > 1:
            # 경로의 x, y 좌표 추출
            xs = [pose.pose.position.x for pose in path_msg.poses]
            ys = [pose.pose.position.y for pose in path_msg.poses]
            
            # 누적 거리 계산
            s = np.zeros(len(xs))
            for i in range(1, len(xs)):
                s[i] = s[i-1] + math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
            
            # Cubic spline 보간
            self.spline_x = CubicSpline(s, xs)
            self.spline_y = CubicSpline(s, ys)
            
            # 보간된 경로 생성 (대략 1m 간격으로 포인트 생성)
            num_points = int(s[-1]) if int(s[-1]) > 0 else 1 # 경로 길이가 0보다 클 때만 계산
            self.spline_points = np.linspace(0, s[-1], num_points) if num_points > 1 else np.array([0])
            self.spline_length = s[-1]

            self.publish_spline_path()
            self.publish_spline_marker()

            
            # 경로의 마지막 점을 목표점으로 설정
            self.goal_point = [xs[-1], ys[-1]]
            
            # path_length_list 업데이트 (기존 코드와의 호환성을 위해)
            self.path_length_list = [0]
            total_length = 0.0
            for i in range(1, len(path_msg.poses)):
                prev = path_msg.poses[i - 1].pose.position
                curr = path_msg.poses[i].pose.position
                dist = math.hypot(curr.x - prev.x, curr.y - prev.y)
                total_length += dist
                self.path_length_list.append(total_length)

    def pose_callback(self, msg):
        self.current_pose = msg
    
    def publish_spline_path(self):
        if self.spline_points is None:
            return

        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = rospy.Time.now()

        for s in self.spline_points:
            p = PoseStamped()
            p.header = path_msg.header
            p.pose.position.x = float(self.spline_x(s))
            p.pose.position.y = float(self.spline_y(s))
            p.pose.position.z = 0.0
            # 방향 정보는 없어서 identity quaternion
            p.pose.orientation.w = 1.0
            path_msg.poses.append(p)

        self.spline_path_pub.publish(path_msg)
        rospy.loginfo("✅ Spline path published with %d poses", len(path_msg.poses))
    
    def publish_spline_marker(self):
        if self.spline_points is None:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "spline_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.3  # 선 두께
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # 불투명

        marker.pose.orientation.w = 1.0  # orientation 설정 (필수)

        for s in self.spline_points:
            pt = Point()
            pt.x = float(self.spline_x(s))
            pt.y = float(self.spline_y(s))
            pt.z = 0.0
            marker.points.append(pt)

        self.spline_marker_pub.publish(marker)
        rospy.loginfo("✅ Spline marker published with %d points", len(marker.points))



    def server_to_robot_callback(self, msg):
        self.server_cmd_drive_mode = msg.Cmd_drive_mode
        if msg.Cmd_use_vel_control:
            self.max_speed = msg.Cmd_linear_velocity
            self.max_yaw_rate = msg.Cmd_angular_velocity

    def publish_traj_info(self):
        traj_info_msg = Float32MultiArray()
        traj_info = [
            self.path_length_list[-1],
            self.path_length_list[self.current_point_idx],
            self.current_point_idx,
            self.reach_goal_flag
        ]
        traj_info_msg.data = traj_info
        self.traj_info_pub.publish(traj_info_msg)

    def publish_drive(self, u):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = u[0]
        cmd_vel_msg.angular.z = u[1]
        self.cmd_vel_pub.publish(cmd_vel_msg)

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
        if self.spline_points is None:
            return
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0
        marker.color.a = 1.0
        marker.color.b = 1.0
        
        current_s = self.spline_points[self.current_point_idx]
        marker.pose.position.x = self.spline_x(current_s)
        marker.pose.position.y = self.spline_y(current_s)
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

    def find_closest_point(self, x):
        if self.spline_points is None:
            return 0
            
        min_dist = float('inf')
        closest_idx = 0
        
        # 현재 위치에서 가장 가까운 보간 포인트 찾기
        for i in range(self.current_point_idx, len(self.spline_points)):
            s = self.spline_points[i]
            px = self.spline_x(s)
            py = self.spline_y(s)
            dist = math.hypot(x[0] - px, x[1] - py)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
            if dist > self.current_point_search_radius_m:  # dist > 검색반경:5m
                break
                
        return closest_idx

    def find_lookahead_point(self, x, current_idx):
        if self.spline_points is None:
            return None
            
        # 현재 위치에서 lookahead 거리만큼 떨어진 포인트 찾기
        min_dist = float('inf')
        farthest_point_in_range = None
        
        for i in range(current_idx, len(self.spline_points)):
            s = self.spline_points[i]
            px = self.spline_x(s)
            py = self.spline_y(s)
            dist = math.hypot(x[0] - px, x[1] - py)
            
            if dist >= self.lookahead_distance:
                return [px, py]
                
            # min_dist: 지금까지 찾은 가장 먼 거리값
            if dist > min_dist:
                min_dist = dist
                farthest_point_in_range = [px, py]
                
        # lookahead 거리만큼 떨어진 포인트를 찾지 못한 경우 farthest_point_in_range 또는 마지막 포인트 반환
        if farthest_point_in_range is not None:
            print(f"[WARN] Lookahead point not found >= {self.lookahead_distance:.2f}m. Using farthest point in range (dist={min_dist:.2f}m).")
            return farthest_point_in_range
        else:
            last_s = self.spline_points[-1]
            last_point = [self.spline_x(last_s), self.spline_y(last_s)]
            print(f"[WARN] Lookahead point not found. Using last spline point: {last_point}")
            return last_point

    def run(self):
        x = [self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y, 
             self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation), 0.0, 0.0]
             
        while not rospy.is_shutdown():
            rospy.sleep(0.01)
            
            if self.path_callback_flag:
                self.path_callback_flag = False
                # Goal point reset
                self.goal_point = [self.path_msg.poses[-1].pose.position.x, self.path_msg.poses[-1].pose.position.y]
                
                # Current point reset using spline points
                if self.spline_points is not None:
                    self.current_point_idx = self.find_closest_point(x)
                    self.start_point = [self.spline_x(self.spline_points[self.current_point_idx]), 
                                      self.spline_y(self.spline_points[self.current_point_idx])]
                
                self.path_initialize_flag = True
                self.warm_up_flag = False  # Reset warm_up_flag when new path is received
            
            if self.path_initialize_flag and self.spline_points is not None:
                # 현재 위치에서 가장 가까운 포인트 찾기
                self.current_point_idx = self.find_closest_point(x)
                self.visualize_current_point()
                self.publish_traj_info()
                
                               
                # DWA control with target point
                # if self.server_cmd_drive_mode == 2:  # Uncomment this line and comment out the next line to use server_cmd_drive_mode
                if True:  # Comment out this line and uncomment the above line to use server_cmd_drive_mode
                     # 목표점 도달 여부 확인
                    self.prev_goal_flag = self.reach_goal_flag
                    self.reach_goal_flag = self.reached_goal(x, self.goal_point)
                    if self.reach_goal_flag:
                        self.publish_drive([0.0, 0.0])
                        if self.prev_goal_flag == False:
                            print("Goal reached!")
                    elif self.target_point == self.goal_point:
                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        # Add debug log for DWA output

                        if u[0] > 0:
                            u[0] = self.low_speed
                        else:
                            u[0] = -self.low_speed
                        
                        self.visualize_predicted_trajectory(predicted_trajectory)
                        x = self.moving(x, u)
                        self.publish_drive(u)
                        continue
                    elif self.warm_up_flag == False:
                        # Lookahead 포인트 찾기
                        self.target_point = self.find_lookahead_point(x, self.current_point_idx)
                        if self.target_point is not None:
                            self.visualize_target_point(self.target_point)
                            
                            u, predicted_trajectory = self.dwa_control(x, self.target_point)
                            # Add debug log for DWA output

                            self.visualize_predicted_trajectory(predicted_trajectory)
                            
                            if u[0] > 0:
                                u[0] = self.low_speed
                            else:
                                u[0] = -self.low_speed

                            x = self.moving(x, u)
                            self.publish_drive(u)

                            # Warm up flag update based on path_length_list
                            if self.path_length_list[self.current_point_idx] > self.current_point_search_radius_m:
                                self.warm_up_flag = True
                                print("Warm up completed!")
                    else:
                        # Lookahead 포인트 찾기
                        self.target_point = self.find_lookahead_point(x, self.current_point_idx)
                        if self.target_point is not None:
                            self.visualize_target_point(self.target_point)
                            
                            u, predicted_trajectory = self.dwa_control(x, self.target_point)
                            # Add debug log for DWA output

                            self.visualize_predicted_trajectory(predicted_trajectory)
                            
                            if u[0] > 0:
                                u[0] = self.low_speed
                            else:
                                u[0] = -self.low_speed

                            x = self.moving(x, u)
                            self.publish_drive(u)
                        else:
                            self.publish_drive([0.0, 0.0])
                else:
                    self.publish_drive([0.0, 0.0])

    # 나머지 메서드들은 기존 코드와 동일
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

    def get_yaw_from_quaternion(self, quaternion):
        euler = transformations.euler_from_quaternion([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return euler[2]

    def reached_goal(self, x, goal):
        distance_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        return distance_to_goal < 1.0

    def calc_dynamic_window(self, x):
        Vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        Vd = [x[3] - self.max_accel * self.dt, x[3] + self.max_accel * self.dt,
              x[4] - self.max_delta_yaw_rate * self.dt, x[4] + self.max_delta_yaw_rate * self.dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw

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
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.robot_stuck_flag_cons and abs(x[3]) < self.robot_stuck_flag_cons:
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory

    def predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y])
            trajectory = np.vstack((trajectory, x))
            time += self.dt
        return trajectory

    def motion(self, x, u):
        x[2] += u[1] * self.dt
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_to_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost

def main():
    rospy.init_node('dwa_node', anonymous=True)
    dwa = DWAControl()
    dwa.run()

if __name__ == "__main__":
    main() 