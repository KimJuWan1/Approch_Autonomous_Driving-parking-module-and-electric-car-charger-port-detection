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

    def path_callback(self, path_msg):   # 수신한 경로 메시지를 내부에 저장, 경로의 총 길이 및 각 점까지의 누적 거리(path_length_list)를 계산 
        self.path_msg = path_msg
        self.path_callback_flag = True   # 새로운 경로가 수신되었다는 것을 알려주는 플래그
        self.path_initialize_flag = False
        self.warm_up_flag = False
        self.path_length_list = [0]
        total_length = 0.0
        if len(path_msg.poses) > 1:   # 거리 계산 로직
            for i in range(1, len(path_msg.poses)):
                prev = path_msg.poses[i - 1].pose.position
                curr = path_msg.poses[i].pose.position
                dist = math.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2 + (curr.z - prev.z)**2)
                total_length += dist
                self.path_length_list.append(total_length)   # self.path_length_list[i]는 0번점부터 i번점까지의 누적 거리

    def pose_callback(self, msg):  # 로봇의 현재 위치(Odometry)를 수신하면 호출
        self.current_pose = msg

    def server_to_robot_callback(self, msg):
        self.server_cmd_drive_mode = msg.Cmd_drive_mode
        if msg.Cmd_use_vel_control:
            self.max_speed = msg.Cmd_linear_velocity
            self.max_yaw_rate = msg.Cmd_angular_velocity

    def publish_traj_info(self):  # 현재 주행 상태 정보를 /traj_info라는 토픽에 퍼블리시 
        traj_info_msg = Float32MultiArray()
        traj_info = [
            self.path_length_list[-1],              # 전체 경로 길이 (마지막까지 누적 거리)
            self.path_length_list[self.current_point_idx],  # 현재까지 따라온 경로 길이
            self.current_point_idx,                 # 현재 경로 상 인덱스
            self.reach_goal_flag                    # 목표점 도달 여부 (True/False)
        ]
        traj_info_msg.data = traj_info
        self.traj_info_pub.publish(traj_info_msg)

    def publish_drive(self, u): # 실제로 로봇을 움직이게 하는 명령을 /cmd_vel로 퍼블리시,   DWA 제어를 통해 계산된 최적의 제어값을 이 함수로 전달
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

    def dwa_control(self, x, goal):
        dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal)
        return u, trajectory

    def moving(self, x, u):  # 현재 로봇의 위치 및 속도 상태를 업데이트
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

    def is_curve(self, idxs, threshold_deg=30):
        # idxs: 경로 상의 인덱스 리스트
        if len(idxs) < 3:   # 최소한 3개 이상의 포인트가 있어야 곡률을 판단할 수 있음
            return False
        for i in range(1, len(idxs)-1):
            p1 = self.path_msg.poses[idxs[i-1]].pose.position    # 세 점의 위치를 추출하여 벡터 구성
            p2 = self.path_msg.poses[idxs[i]].pose.position
            p3 = self.path_msg.poses[idxs[i+1]].pose.position
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            angle = math.acos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))   # 코사인 법칙으로 두 벡터 사이의 각도(라디안) 계산
            if np.degrees(angle) > threshold_deg:
                return True
        return False

    def run(self):
        x = [self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y, self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation), 0.0, 0.0]
        while not rospy.is_shutdown():
            rospy.sleep(0.01)
            if self.path_callback_flag:
                self.path_callback_flag = False
                # 경로의 마지막 점을 목표점으로 설정
                self.goal_point = [self.path_msg.poses[-1].pose.position.x, self.path_msg.poses[-1].pose.position.y]
                # coarse search : 현재 위치에서 경로를 따라가며 5m 이내인 첫 번째 점을 coarse하게 찾음
                coarse_current_point_idx = -1
                for i in range(len(self.path_msg.poses)):
                    point_distance = math.hypot(x[0] - self.path_msg.poses[i].pose.position.x, x[1] - self.path_msg.poses[i].pose.position.y)
                    if point_distance < 5:
                        coarse_current_point_idx = i
                        break
                # fine curr idx, coarse search 결과부터 시작해서, 그 이후 구간을 8m 범위 내에서만 정밀하게 탐색, 가장 가까운 점을 min_point_idx로 저장    
                min_point_distance = float("inf")
                min_point_idx = -1
                for i in range(coarse_current_point_idx, len(self.path_msg.poses)):
                    point_distance = math.hypot(x[0] - self.path_msg.poses[i].pose.position.x, x[1] - self.path_msg.poses[i].pose.position.y)
                    if point_distance < min_point_distance:
                        min_point_distance = point_distance
                        min_point_idx = i
                    if point_distance > 8:
                        break
                # 현재 경로 시작 인덱스를 저장, 그 위치를 start_point로 설정    
                self.current_point_idx = min_point_idx
                self.start_point = [self.path_msg.poses[min_point_idx].pose.position.x, self.path_msg.poses[min_point_idx].pose.position.y]

                self.path_initialize_flag = True

                # 요약 : 로봇이 새 경로를 수신하면 먼저 coarse search로 5m 이내의 경로 점을 빠르게 찾고, 이후 fine search로 8m 범위 내에서 가장 가까운 점을 정확히 찾아서 경로 추종의 시작점으로 지정


            if self.path_initialize_flag == True:
                # 1. current_point_idx 갱신 (가장 먼저!), 현재 위치 기준 가장 가까운 경로 인덱스 갱신
                min_point_distance = float("inf")
                min_point_idx = -1
                prev_current_point_idx = self.current_point_idx
                for i in range(prev_current_point_idx, len(self.path_msg.poses)):
                    point_distance = math.hypot(
                        x[0] - self.path_msg.poses[i].pose.position.x,
                        x[1] - self.path_msg.poses[i].pose.position.y
                    )
                    # 최소 거리를 만족하는 인덱스를 저장
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
                    # 목표점 도달 여부 판단, reached_goal()은 현재 위치가 목표점에서 1m 이내이면 True를 반환
                    self.prev_goal_flag = self.reach_goal_flag
                    self.reach_goal_flag = self.reached_goal(x, self.goal_point)
                    if self.reach_goal_flag: # Check if robot reached the goal
                        # Stop the robot
                        self.publish_drive([0.0, 0.0])
                        if self.prev_goal_flag == False:
                            print("Goal reached!")
                    # 목표점이 바로 타겟인 경우
                    elif self.target_point == self.goal_point:
                        # 현재 타겟이 최종 목표라면, DWA로 조심스럽게 접근
                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        self.visualize_predicted_trajectory(predicted_trajectory)
                        if u[0] > 0:
                            u[0] = self.low_speed
                        else:
                            u[0] = -self.low_speed
                        
                        self.visualize_predicted_trajectory(predicted_trajectory)

                        x = self.moving(x, u)
                        self.publish_drive(u)
                        continue  # 아래 코드 실행하지 않고 다음 루프로
                    
                    elif self.warm_up_flag == False:
                        # Update lookahead point
                        distance = 0
                        for i in range(self.current_point_idx, len(self.path_msg.poses) - 1):
                            if distance < self.lookahead_distance:
                                p1 = self.path_msg.poses[i].pose.position
                                p2 = self.path_msg.poses[i+1].pose.position
                                distance += math.hypot(p2.x - p1.x, p2.y - p1.y)
                                lookahead_point_idx = i + 1
                            else:
                                break
                        self.target_point = [self.path_msg.poses[lookahead_point_idx].pose.position.x,
                                            self.path_msg.poses[lookahead_point_idx].pose.position.y]
                        self.visualize_target_point(self.target_point)

                        u, predicted_trajectory = self.dwa_control(x, self.target_point)
                        self.visualize_predicted_trajectory(predicted_trajectory)
                        x = self.moving(x, u)
                        if u[0] > 0:
                            u[0] = self.low_speed
                        else:
                            u[0] = -self.low_speed

                        self.publish_drive(u)

                        # 경로 누적 길이가 5m 정도(=search radius) 이상 되면 워밍업 종료, 이후부터는 커브 판단 + 보간을 통한 정교한 주행
                        if self.path_length_list[self.current_point_idx] > self.current_point_search_radius_m:
                            self.warm_up_flag = True
                    
                    else:
                        # lookahead 구간 내 포인트 인덱스 수집
                        distance = 0
                        lookahead_point_idx = self.current_point_idx
                        idxs = [self.current_point_idx]
                        for i in range(self.current_point_idx, len(self.path_msg.poses) - 1):
                            if distance < self.lookahead_distance:
                                p1 = self.path_msg.poses[i].pose.position
                                p2 = self.path_msg.poses[i+1].pose.position
                                distance += math.hypot(p2.x - p1.x, p2.y - p1.y)
                                lookahead_point_idx = i + 1
                                idxs.append(lookahead_point_idx)
                            else:
                                break
                        
                        # 커브 여부 판단, idxs는 현재 로봇 위치에서부터 설정한 lookahead_distance 만큼 떨어진 구간 안에 속하는 “경로점들의 인덱스 목록”
                        if self.is_curve(idxs):
                            # 커브면 cubic spline 보간
                            xs = [self.path_msg.poses[i].pose.position.x for i in idxs]
                            ys = [self.path_msg.poses[i].pose.position.y for i in idxs]
                            s = np.cumsum([0] + [math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]) for i in range(1, len(xs))])  # s: 각 점까지의 누적 거리 리스트 (보간을 위한 x축 역할)
                            # 누적 거리 s를 기준으로 x 및 y 각각에 대해 Cubic Spline 생성
                            cs_x = CubicSpline(s, xs)
                            cs_y = CubicSpline(s, ys)
                            #  s에 대해 보간 수행 → 부드러운 곡선 궤적 생성 , sx, sy: 곡선을 따라 생성된 보간 포인트들, 0부터 s[-1]까지를 20개의 동일 간격 구간으로 나눠서 값들을 생성
                            s_interp = np.linspace(0, s[-1], 100)
                            sx = cs_x(s_interp) # 보간된 x 좌표들
                            sy = cs_y(s_interp) # 보간된 y 좌표들
                            # 현재 위치에서 가장 가까운 보간점 중 lookahead 거리 이상 떨어진 점을 타겟으로
                            for i in range(len(sx)):
                                if math.hypot(sx[i] - x[0], sy[i] - x[1]) >= self.lookahead_distance:
                                    self.target_point = [sx[i], sy[i]]
                                    break
                                else:
                                    self.target_point = [sx[-1], sy[-1]]
                        else:
                            # 직선구간은 기존 lookahead 방식
                            self.target_point = [self.path_msg.poses[lookahead_point_idx].pose.position.x, self.path_msg.poses[lookahead_point_idx].pose.position.y]
                        self.visualize_target_point(self.target_point)
                        
                        u, trajectory = self.dwa_control(x, self.target_point)
                        self.visualize_predicted_trajectory(trajectory)
                        if u[0] > 0:
                            u[0] = self.low_speed
                        else:
                            u[0] = -self.low_speed
                        x = self.moving(x, u)
                        self.publish_drive(u)
                else:
                    self.publish_drive([0.0, 0.0])

    def calc_dynamic_window(self, x):
        Vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        Vd = [x[3] - self.max_accel * self.dt, x[3] + self.max_accel * self.dt, x[4] - self.max_delta_yaw_rate * self.dt, x[4] + self.max_delta_yaw_rate * self.dt]
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
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)   # 예측된 마지막 위치에서 목표방향과의 각도 차이 (방향성)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])  # 현재 속도가 max_speed에 가까울수록 좋은 것으로 평가
                final_cost = to_goal_cost + speed_cost  # 최종 비용 = 방향 비용 + 속도 비용
                if min_cost >= final_cost:   # 가장 비용이 작은 제어 명령을 저장
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.robot_stuck_flag_cons and abs(x[3]) < self.robot_stuck_flag_cons:
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory   # 가장 좋은 제어 명령을 선택

    def predict_trajectory(self, x_init, v, y):   # 지금 제어 명령 [v, w]로 predict_time 동안 로봇이 어떻게 움직일지를 예측
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y])
            trajectory = np.vstack((trajectory, x))    # 로봇을 dt 간격으로 motion()을 통해 이동시키며 trajectory를 누적, trajectory의 shape은 (N, 5)이고, 각 row는 [x, y, yaw, v, w]
            time += self.dt
        return trajectory

    def motion(self, x, u):  # 로봇의 현재 상태 x에서 제어 명령 u = [v, w]를 적용했을 때 1 step 후의 상태를 반환
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