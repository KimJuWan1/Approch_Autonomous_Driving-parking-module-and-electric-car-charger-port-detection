#!/usr/bin/env python
# license removed for brevity

import rospy
from nav_msgs.msg import Path, Odometry
import tf.transformations as transformations
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from dynamic_window_approach.msg import server_to_robot
from std_msgs.msg import Float32MultiArray

import math

import numpy as np

class DWAControl:
    def __init__(self):
        # Initialize parameters
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
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        # self.robot_width = 0.245  # [m] for collision check
        # self.robot_length = 0.612  # [m] for collision check

        self.server_cmd_drive_mode = 0
        
        # Initialize subscribers and publisher
        # self.sub_path = rospy.Subscriber('local_planner/global_trajectory_planner/local', Path, self.path_callback)
        self.sub_path = rospy.Subscriber('/astar/path', Path, self.path_callback)
        self.sub_pose = rospy.Subscriber('lio_localizer/odometry/optimization', Odometry, self.pose_callback)
        self.sub_server_cmd = rospy.Subscriber("server_to_robot_topic", server_to_robot, self.server_to_robot_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.target_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.current_pub = rospy.Publisher('visualization_marker_2', Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('predicted_trajectory', Marker, queue_size=10)
        self.traj_info_pub = rospy.Publisher('/traj_info', Float32MultiArray, queue_size=10)
        
        # Initialize variables
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
        
       
    def path_callback(self, path_msg):
        # Callback function for receiving path
        self.path_msg = path_msg
        self.path_callback_flag = True
        self.path_initialize_flag = False
        self.warm_up_flag = False

        self.path_length_list = [0]
        total_length = 0.0
        if len(path_msg.poses) > 1:
            for i in range(1, len(path_msg.poses)):
                # 이전 포즈와 현재 포즈 사이의 거리를 계산
                prev = path_msg.poses[i - 1].pose.position
                curr = path_msg.poses[i].pose.position
                dist = math.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2 + (curr.z - prev.z)**2)
                total_length += dist
                self.path_length_list.append(total_length)
   
    def pose_callback(self, msg):
        # Callback function for receiving pose
        self.current_pose = msg

    def server_to_robot_callback(self, msg):
        self.server_cmd_drive_mode = msg.Cmd_drive_mode
        if msg.Cmd_use_vel_control == True:
            if self.max_speed != msg.Cmd_linear_velocity or self.max_yaw_rate != msg.Cmd_angular_velocity:
                self.max_speed = msg.Cmd_linear_velocity
                self.max_yaw_rate = msg.Cmd_angular_velocity
                print("self.max_speed: ", self.max_speed, ", self.max_yaw_rate: ", self.max_yaw_rate)
            # if self.max_speed > msg.Cmd_linear_velocity:
            #     self.max_speed = msg.Cmd_linear_velocity
            # if self.max_yaw_rate > msg.Cmd_angular_velocity:
            #     self.max_yaw_rate = msg.Cmd_angular_velocity

       
    def publish_drive(self, u):
        # Publishing control commands
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = u[0]
        cmd_vel_msg.angular.z = u[1]
        self.cmd_vel_pub.publish(cmd_vel_msg)
    
    def publish_traj_info(self):
        traj_info_msg = Float32MultiArray()
        traj_info = [self.path_length_list[-1], self.path_length_list[self.current_point_idx], self.current_point_idx, self.reach_goal_flag]
        traj_info_msg.data = traj_info
        self.traj_info_pub.publish(traj_info_msg)
        
    def visualize_target_point(self, target_point):
        # Visualize target point in rviz
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5  # 크기 조정
        marker.scale.y = 0.5
        marker.scale.z = 0
        marker.color.a = 1.0  # 투명도 설정 (1.0은 완전 불투명)
        marker.color.g = 1.0  # 색상 설정 (G: 초록색)
        marker.pose.position.x = target_point[0]
        marker.pose.position.y = target_point[1]
        self.target_pub.publish(marker)
        
    def visualize_current_point(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5 # 크기 조정
        marker.scale.y = 0.5
        marker.scale.z = 0
        marker.color.a = 1.0  # 투명도 설정 (1.0은 완전 불투명)
        marker.color.b = 1.0  # 색상 설정
        # 현재 위치의 좌표 설정
        marker.pose.position.x = self.path_msg.poses[self.current_point_idx].pose.position.x
        marker.pose.position.y = self.path_msg.poses[self.current_point_idx].pose.position.y
        self.current_pub.publish(marker)
        
    def visualize_predicted_trajectory(self, predicted_trajectory):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.2  # 선의 두께
        marker.color.a = 1.0  # 투명도 설정 (1.0은 완전 불투명)
        marker.color.r = 0.0  # 색상 설정 (R: 빨간색)
        marker.color.g = 0.0
        marker.color.b = 1.0

        # 예측된 경로의 점들을 Marker 메시지에 추가
        for point in predicted_trajectory:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 1.0  
            marker.points.append(p)

        self.trajectory_pub.publish(marker)

    def dwa_control(self, x, goal):
        # Dynmamic Window Approach control
        dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal)
        
        return u, trajectory
   
    def moving(self, x, u):
        # Move function based on current pose and control input (for update x)
        x[2] = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
        x[0] = self.current_pose.pose.pose.position.x
        x[1] = self.current_pose.pose.pose.position.y
        x[3] = u[0]
        x[4] = u[1]
        
        return x
   
    def motion(self, x, u):
        # Motion update based on control input (for calculating trajectory)
        x[2] += u[1] * self.dt
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        x[3] = u[0]
        x[4] = u[1]
        
        return x

    def calc_dynamic_window(self, x):
        # Calculate dynamic window
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
        # Predict trajectory based on motion model
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v,y])
            trajectory = np.vstack((trajectory, x))
            time += self.dt
            
        return trajectory
   
    def calc_control_and_trajectory(self, x, dw, goal):
        # Calculate control command and corresponding trajectory
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
       
        # print("num candi v: ", (dw[1] - dw[0]) / self.v_resolution, ", yaw: ", (dw[3]-dw[2]) / self.yaw_rate_resolution)
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
                    if abs(best_u[0]) < self.robot_stuck_flag_cons \
                            and abs(x[3]) < self.robot_stuck_flag_cons:
                        best_u[1] = -self.max_delta_yaw_rate
                       
        return best_u, best_trajectory
   
    def calc_to_goal_cost(self, trajectory, goal):
        # Calculate cost to goal
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        
        return cost
   
    def get_yaw_from_quaternion(self, quaternion):
        # Get yaw angle from quaternion
        euler = transformations.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        
        return euler[2]
   
    def reached_goal(self, x, goal):
        # Check if robot reached the goal
        distance_threshold = 1.0
        distance_to_goal = math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2)

        return distance_to_goal < distance_threshold

    def run(self):
        # Main Loop
        x = [self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y, self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation), 0.0, 0.0]
        while not rospy.is_shutdown():
            rospy.sleep(0.01)
            if self.path_callback_flag == True:
                self.path_callback_flag = False
                # Goal point reset
                self.goal_point = [self.path_msg.poses[-1].pose.position.x, self.path_msg.poses[-1].pose.position.y]
                
                # Current point reset
                # - coarse curr idx
                coarse_current_point_idx = -1
                for i in range(len(self.path_msg.poses)):
                    point_distance = math.sqrt(pow(x[0] - self.path_msg.poses[i].pose.position.x, 2) + pow(x[1] - self.path_msg.poses[i].pose.position.y, 2))
                    if point_distance < 5:
                        coarse_current_point_idx = i
                        break
                
                # - fine curr idx
                min_point_distance = float("inf")
                min_point_idx = -1
                for i in range(coarse_current_point_idx, len(self.path_msg.poses)):
                    point_distance = math.sqrt(pow(x[0] - self.path_msg.poses[i].pose.position.x, 2) + pow(x[1] - self.path_msg.poses[i].pose.position.y, 2))
                    if point_distance < min_point_distance:
                        min_point_distance = point_distance
                        min_point_idx = i
                    if point_distance > 8:
                        break
                    
                self.current_point_idx = min_point_idx
                self.start_point = [self.path_msg.poses[min_point_idx].pose.position.x, self.path_msg.poses[min_point_idx].pose.position.y]
                
                # Flag update
                self.path_initialize_flag = True
            
            if self.path_initialize_flag == True:
                
                # update current_point_idx using idx near previous current_point_idx
                min_point_distance = float("inf")
                min_point_idx = -1
                prev_current_point_idx = self.current_point_idx
                for i in range(prev_current_point_idx, len(self.path_msg.poses)):
                    point_distance = math.sqrt(pow(x[0] - self.path_msg.poses[i].pose.position.x, 2) + pow(x[1] - self.path_msg.poses[i].pose.position.y, 2))
                    if point_distance < min_point_distance:
                        min_point_distance = point_distance
                        min_point_idx = i
                    if point_distance > self.current_point_search_radius_m:
                        break
                self.current_point_idx = min_point_idx
                self.visualize_current_point()
                self.publish_traj_info()
                                
                # DWA control with target point
                #if self.server_cmd_drive_mode == 2:
                if True:
                    self.prev_goal_flag = self.reach_goal_flag
                    self.reach_goal_flag = self.reached_goal(x, self.goal_point)
                    if self.reach_goal_flag: # Check if robot reached the goal
                        # Stop the robot
                        self.publish_drive([0.0, 0.0])
                        
                        if self.prev_goal_flag == False:
                            print("Goal reached!")
                    elif self.target_point == self.goal_point: # Check if target point == goal point
                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        target_speed = u[0]
                        if target_speed > 0:
                            u[0] = self.low_speed
                        else:
                            u[0] = -self.low_speed
                        
                        # visualization predicted_trajectory
                        self.visualize_predicted_trajectory(predicted_trajectory)

                        x = self.moving(x, u)
                        self.publish_drive(u)
                        continue
                    elif self.warm_up_flag == False:
                        # Update lookahead point
                        distance = 0
                        for i in range(self.current_point_idx, len(self.path_msg.poses) - 1):
                            if distance < self.lookahead_distance:
                                distance += math.sqrt(pow(self.path_msg.poses[i+1].pose.position.x - self.path_msg.poses[i].pose.position.x, 2) + pow(self.path_msg.poses[i+1].pose.position.y - self.path_msg.poses[i].pose.position.y, 2))
                                lookahead_point_idx = i + 1
                            else:
                                break
                        self.target_point = [self.path_msg.poses[lookahead_point_idx].pose.position.x, self.path_msg.poses[lookahead_point_idx].pose.position.y]
                        # Visualize target point
                        self.visualize_target_point(self.target_point)
                        
                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        # visualization predicted_trajectory
                        self.visualize_predicted_trajectory(predicted_trajectory)
                        x = self.moving(x, u)
                        target_speed = u[0]
                        if target_speed > 0:
                            u[0] = self.low_speed
                        else:
                            u[0] = -self.low_speed
                        
                        self.publish_drive(u)

                        if(self.path_length_list[self.current_point_idx] > self.current_point_search_radius_m):
                            self.warm_up_flag = True

                    else:
                        # Update lookahead point
                        distance = 0
                        for i in range(self.current_point_idx, len(self.path_msg.poses) - 1):
                            if distance < self.lookahead_distance:
                                distance += math.sqrt(pow(self.path_msg.poses[i+1].pose.position.x - self.path_msg.poses[i].pose.position.x, 2) + pow(self.path_msg.poses[i+1].pose.position.y - self.path_msg.poses[i].pose.position.y, 2))
                                lookahead_point_idx = i + 1
                            else:
                                break
                        self.target_point = [self.path_msg.poses[lookahead_point_idx].pose.position.x, self.path_msg.poses[lookahead_point_idx].pose.position.y]
                        # Visualize target point
                        self.visualize_target_point(self.target_point)
                        
                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        # visualization predicted_trajectory
                        self.visualize_predicted_trajectory(predicted_trajectory)
                        x = self.moving(x, u)
                        self.publish_drive(u)
                else:
                    self.publish_drive([0.0, 0.0])
            

                
def main():
    print("start!!")
   
    rospy.init_node('dwa_node', anonymous=True) 
   
    dwa = DWAControl()
    dwa.run()


if __name__ == "__main__":
    main()
