#!/usr/bin/env python3
# license removed for brevity

import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header, Int32MultiArray
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker

import xml.etree.ElementTree as ET
import utm
import math
import sys

from astar_map.msg import server_to_robot
import colorsys  # 색상을 다루기 위한 표준 라이브러리
import struct

class Node():
    def __init__(self, id, east, north, lat, lon, name=None):
        self.id = id
        self.east = east
        self.north = north
        self.lat = lat
        self.lon = lon 
        self.name = name
        self.parent = None
        self.cost = sys.maxsize

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"ID: {self.id}, Name: {self.name}, Cost: {self.cost}"

class Edge():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

class AStarPlanner():
    def __init__(self):
        self.node_list = []
        self.edge_list = []
        self.origin = None
        self.start_id = None
        self.goal_id = None
        self.pub_marker = rospy.Publisher('/astar/graph_markers', Marker, queue_size=10)
        # self.pub_path = rospy.Publisher('/astar/path_markers', Marker, queue_size=10)
        self.pub_path = rospy.Publisher('/astar/path', Path, queue_size=10)
        self.pub_path_wgs84 = rospy.Publisher('/astar/path_wgs84', Path, queue_size=10)
        self.pub_path_node_id_list = rospy.Publisher('/astar/path_node_id_list', Int32MultiArray, queue_size=10)        
        self.pub_server_dst_list = rospy.Publisher('/astar/server_dst_node_list', PointCloud2, queue_size=10)
        self.sub_start_from_rviz = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.callback_start)
        self.sub_start_from_pose = rospy.Subscriber('lio_localizer/odometry/optimization', Odometry, self.pose_callback)
        self.sub_goal_from_rviz = rospy.Subscriber('/move_base_simple/goal',PoseStamped,self.callback_goal_from_rviz)
        self.sub_goal_from_server = rospy.Subscriber('/server_to_robot_topic',server_to_robot,self.callback_goal_from_server)
        self.start_init_flag = False
        self.new_goal_flag = False
        self.server_dst_node_list = []

    def load_osm_data(self, osm_file, ref_file):
        tree = ET.parse(osm_file) # osm 파일 파싱 후 XML 트리 객체 생성
        root = tree.getroot() # osm 파일 루트 엘리먼트 저장

        # XML 트리에서 'node' 엘리멘트 찾기
        for node_elem in root.findall('.//node'):
            # id, lat, lon 값 추출 후 Node 생성
            node_id = int(node_elem.attrib['id'])
            lat = float(node_elem.attrib['lat'])
            lon = float(node_elem.attrib['lon'])
            # 'name' tag 값이 존재하면 name으로 설정
            name = node_elem.find('tag[@k="name"]')
            name = name.attrib['v'] if name is not None else None
            # lat, lon 값을 utm 좌표로 변환 후 변환된 좌표로 node 생성
            east, north, _, _ = utm.from_latlon(lat, lon)
            # node_list에 추가
            self.node_list.append(Node(node_id, east, north, lat, lon, name))

        if self.origin == None:
            # 파일을 열고 내용을 읽는다.
            with open(ref_file, 'r') as file:
                # 첫 번째 줄은 헤더이므로 건너뛴다.
                next(file)
                # 두 번째 줄의 데이터를 읽는다.
                data = file.readline()

                # 쉼표로 구분된 값을 리스트로 변환한다.
                values = data.split(',')

                # 각 변수에 값을 저장한다.
                ref_east = float(values[3])
                ref_north = float(values[4])
                ref_height = float(values[5])  # ref_height가 두 번 나오므로, 두 번째 값을 다른 변수에 저장
                self.origin = [ref_east, ref_north]

                # 변수 값 확인을 위한 출력
                print(f"ref_east: {ref_east}")
                print(f"ref_north: {ref_north}")
                print(f"ref_height_again: {ref_height}")

        # XML 트리에서 'way' 엘리멘트 찾기
        for way_elem in root.findall('.//way'):
            # 'nd' tag에 연결된 노드 순서 파악하여 'node'객체로 'Edge' 생성
            node_ids = way_elem.findall('nd')
            # 각 엣지를 edge_list에 추가
            for i in range(len(node_ids) - 1):
                src_node = self.findNodeById(int(node_ids[i].attrib['ref']))
                dst_node = self.findNodeById(int(node_ids[i + 1].attrib['ref']))
                self.edge_list.append(Edge(src_node, dst_node))
                self.edge_list.append(Edge(dst_node, src_node))
                # 노드 사이에 여러개의 노드 추가해서 가장 가까운 노드를 찾는 데 더 효율적이도록 해야함
    
    def set_dst_node_list(self, i_server_dst_node_list):
        self.server_dst_node_list = i_server_dst_node_list
        return

    def graph_setup(self):
        for node in self.node_list:
            node.parent = None
            node.cost = sys.maxsize
        return

    # 'id'에 해당하는 노드를 node_list에서 찾기
    def findNodeById(self, id):
        # id가 일치하면 해당 노드 반환
        for node in self.node_list:
            if node.id == id:
                return node
        # 일치하는 노드가 없으면 None 반환
        return None

    # 'name'에 해당하는 노드를 node_list에서 찾기
    def findNodeByName(self, name):
        # name이 일치하면 해당 노드 반환
        return_node = None
        for node in self.node_list:
            if node.name == name:
                print("find start node: ", node.id)
                return_node = node
                # return node/
        # 일치하는 노드가 없으면 None 반환
        # return None
        return return_node

    # 시작노드에서 목표노드까지 최적경로 계획
    def planning(self, start_id, goal_id):
        # name으로 노드 찾기
        start_node = self.findNodeById(start_id)
        goal_node = self.findNodeById(goal_id)
    
        # 초기화
        open_set, closed_set = dict(), dict()
        # 시작노드를 open_set에 추가
        open_set[start_node.id] = start_node
        # 시작비용을 0으로 설정
        start_node.cost = 0

        # open_set이 비어있지 않는동안 반복
        while open_set:
            # open_set에서 가장 작은 비용을 가진 노드 선택
            c_id = min(open_set, key=lambda o: open_set[o].cost + self.distance(goal_node, open_set[o]))
            current = open_set[c_id]

            # 선택된 노드와 목표노드가 일치하면 문구 출력
            if current == goal_node:
                print("Find Goal")
                break
            
            # 선택된 노드를 open_set에서 제거
            del open_set[c_id]
            # 선택된 노드를 closeㅇ_set에 추가(이미 처리한 노드로 간주)
            closed_set[c_id] = current

            # 선택된 노드에 연결된 간선 반복
            for edge in self.edges(current):
                # 현재 간선의 목표노드의 id 가져옴
                n_id = edge.dst.id
                # 목표노드가 이미 closed_set에 있다면 무시
                if n_id in closed_set:
                    continue
                
                # 목표 노드까지의 현재 비용이 더 작다면 업데이트
                if edge.dst.cost > edge.src.cost + self.distance(edge.src, edge.dst):
                    edge.dst.cost = edge.src.cost + self.distance(edge.src, edge.dst)
                    edge.dst.parent = edge.src

                # 목표노드가 open_set에 없다면 open_set에 추가
                if n_id not in open_set:
                    open_set[n_id] = edge.dst
        
        # 최적경로 탐색 및 반환
        path = self.backtracking(goal_node)
        return path if path else []

    # 특정노드에 연결된 간선 반환
    def edges(self, node):
        ret_edges = []
        # 그래프에 있는 모든 간선 반복
        for edge in self.edge_list:
            # 현재 간선의 출발노드가 주어진 노드와 같다면 해당간선을 ret_edges에 추가
            if edge.src == node:
                ret_edges.append(edge)
        return ret_edges

    # 두 노드 간의 유클리드 거리 계산(=현재노드와 목표노드까지의 거리)
    def distance(self, goal_node, input_node):
        # goal_node와 input_node 간의 위도, 경도 차이
        diff_x = goal_node.east - input_node.east
        diff_y = goal_node.north - input_node.north
        # 유클리드 거리 계산
        return math.sqrt(diff_x**2 + diff_y**2)
    
    # 경로 탐색 후 거슬러 올라가면서 경로 추적
    def backtracking(self, node):
        ret_nodes = []
        current = node # 주어진 노드를 현재노드로 설정
        while current.parent is not None:
            ret_nodes.append(current.id) # 각 단계에서 현재노드의 id을 ret_nodes 리스트 추가
            current = current.parent # asdf현재노드의 부모노드를 현재노드로 설정하여 상위로 이동
        ret_nodes.append(current.id) # Add start node
        ret_nodes.reverse() # 리스트를 뒤집어 최적 경로 획득
        return ret_nodes # 최적 경로 반환

    # 그래프 시각화
    def findNode(self, node_id):
        node = self.findNodeById(int(node_id))  
        return node
    
    def callback_start(self,data):
        start_node = self.find_nearest_node(data.pose.pose.position)
        if start_node:
            self.start_id = start_node.id
            self.start_init_flag = True
            print("Start x: ", data.pose.pose.position.x, ", y: ", data.pose.pose.position.y, ", id: ", start_node.id)
        else:
            print("Cannot find a valid node for Start")

    def pose_callback(self, data):
        start_node = self.find_nearest_node(data.pose.pose.position)
        if start_node:
            self.start_id = start_node.id
            self.start_init_flag = True
            # print("Start: ", data.pose.pose.position, "id: ", start_node.id)
        else:
            print("Cannot find a valid node for Start")

    def callback_goal_from_rviz(self,data):
        goal_node = self.find_nearest_node(data.pose.position)
        if goal_node:
            self.goal_id = goal_node.id
            self.new_goal_flag = True
            print("New RVIZ Goal x: ", data.pose.position.x, ", y: ", data.pose.position.y, ", id: ", goal_node.id)
        else:
            print("Cannot find a valid node for Goal")

    def callback_goal_from_server(self,data):
        if len(self.server_dst_node_list) != 0:
            if data.Cmd_dest_index > len(self.server_dst_node_list) - 1:
                print("Out of server dst node list index. list_size: ", len(self.server_dst_node_list))
                return     
                
            goal_node_id = self.server_dst_node_list[data.Cmd_dest_index]
            goal_node = self.findNodeById(goal_node_id)
            if goal_node is None:
                print("Cannot find a valid node for Goal")
            elif self.goal_id != goal_node.id:
                self.goal_id = goal_node.id
                self.new_goal_flag = True
                print("New Server Goal id: ", goal_node.id)
        elif data.Cmd_dest_lat > 0.01 and data.Cmd_dest_lon > 0.01:
            east, north, _, _ = utm.from_latlon(data.Cmd_dest_lat, data.Cmd_dest_lon)
            cmd_position = Point()
            cmd_position.x = east - self.origin[0]
            cmd_position.y = north - self.origin[1]

            goal_node = self.find_nearest_node(cmd_position)
            if goal_node is None:
                print("Cannot find a valid node for Goal")
            elif self.goal_id != goal_node.id:
                self.goal_id = goal_node.id
                self.new_goal_flag = True
                print("New Server Goal x: ", cmd_position.x, ", y: ", cmd_position.y, ", id: ", goal_node.id)

        

    # def server_cmd_callback(self, data):


    def find_nearest_node(self, position):
        x, y = position.x, position.y
        min_distance = float('inf')
        nearest_node = None

        for node in self.node_list:
            distance = self.distance_from_position(node, x, y)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    def distance_from_position(self, node, x, y):
        diff_x = node.east - self.origin[0] - x
        diff_y = node.north - self.origin[1] - y
        return math.sqrt(diff_x**2 + diff_y**2)

    # 최종 경로 강조하여 그래프 다시 표시
    def show_path(self, path, show=True):
        # marker = Marker()
        # marker.header.frame_id = "map"
        # marker.header.stamp = rospy.Time.now()
        # marker.type = Marker.LINE_LIST
        # marker.action = Marker.ADD
        # marker.scale.x = 0.5  # Line width
        # marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        path_msg_wgs84 = Path()
        path_msg_wgs84.header.frame_id = "map"
        path_msg_wgs84.header.stamp = rospy.Time.now()

        # Populate the marker with points for the path
        node_list = []
        for node_id in path:
            try:
                node = self.findNode(node_id)    
                node_list.append(node)
            except ValueError:
                print("It's not valid node.  Try again...")

        for src, dst in zip(node_list, node_list[1:]):
            src_x, src_y = (src.east - self.origin[0]), (src.north - self.origin[1])
            dst_x, dst_y = (dst.east - self.origin[0]), (dst.north - self.origin[1])
            
            # # 시각화를 위해 UTM 좌표를 맵 좌표로 변환
            # marker.points.append(Point(src_x, src_y, 0))
            # marker.points.append(Point(dst_x, dst_y, 0))

            # # 선분에 대한 색상 설정
            # marker.colors.append(ColorRGBA(1, 0, 0, 1))  # 선의 첫 번째 점에 대한 초록색
            # marker.colors.append(ColorRGBA(1, 0, 0, 1))  # 선에 대한 초록색
            
            point = PoseStamped()
            point.header = path_msg.header
            point.pose.position.x = src_x
            point.pose.position.y = src_y
            point.pose.position.z = 0
            path_msg.poses.append(point)

            point_wgs84 = PoseStamped()
            point_wgs84.header = path_msg.header
            point_wgs84.pose.position.x = src.lat
            point_wgs84.pose.position.y = src.lon
            point_wgs84.pose.position.z = 0
            path_msg_wgs84.poses.append(point_wgs84)

        point = PoseStamped()
        point.header = path_msg.header
        point.pose.position.x = node_list[-1].east - self.origin[0]
        point.pose.position.y = node_list[-1].north - self.origin[1]
        point.pose.position.z = 0
        path_msg.poses.append(point)

        point_wgs84 = PoseStamped()
        point_wgs84.header = path_msg.header
        point_wgs84.pose.position.x = node_list[-1].lat
        point_wgs84.pose.position.y = node_list[-1].lon
        point_wgs84.pose.position.z = 0
        path_msg_wgs84.poses.append(point_wgs84)

        # Display 
        # marker.pose.orientation.w = 1.0  # Assuming identity quaternion
        self.pub_path.publish(path_msg)
        self.pub_path_wgs84.publish(path_msg_wgs84)

    # 최종 경로 강조하여 그래프 다시 표시
    def show_server_dst_nodes(self):

        num_points = len(self.server_dst_node_list)
        intensity = 255
        start_hue = 0.67  # 빨간색의 Hue 값
        end_hue = 0.833  # 보라색의 Hue 값
        if num_points == 1:
            hues = [start_hue]  # 단 한 점만 있을 경우
        else:
            hues = [(start_hue - 1 / (num_points) * i) % 1 for i in range(num_points)]

        colors = [struct.unpack('I', struct.pack('BBBB',
                    *[int(c * intensity) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)], 255))[0]
                    for hue in hues]

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1),
        ]

        buf = []
        node_list = []
        for node_id in self.server_dst_node_list:
            try:
                node = self.findNode(node_id)
                if node is None: 
                    print("It's not valid node. node_id: ", node_id)
                    return
                node_list.append(node)
            except ValueError:
                print("It's not valid node.  Try again...")

        for node, color in zip(node_list, colors):
            node_x, node_y = (node.east - self.origin[0]), (node.north - self.origin[1])
            buf.append(struct.pack('fffI', node_x, node_y, 0, color))

        header = Header(frame_id="map", stamp=rospy.Time.now())
        cloud = PointCloud2(
                    header=header,
                    height=1,
                    width=num_points,
                    is_dense=True,
                    is_bigendian=False,
                    fields=fields,
                    point_step=16,
                    row_step=16 * num_points,
                    data=b''.join(buf)
                )

        self.pub_server_dst_list.publish(cloud)
        return

    def visualize_graph(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.3
        marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

        # 마커에 점과 색상을 추가
        for edge in self.edge_list:
            src_east, src_north = edge.src.east, edge.src.north
            dst_east, dst_north = edge.dst.east, edge.dst.north

            # UTM 좌표를 맵 좌표로 변환
            src_x, src_y = (src_east - self.origin[0]), (src_north - self.origin[1])
            dst_x, dst_y = (dst_east - self.origin[0]), (dst_north - self.origin[1])

            # 시각화를 위해 UTM 좌표를 맵 좌표로 변환
            marker.points.append(Point(src_x, src_y, -0.1))
            marker.points.append(Point(dst_x, dst_y, -0.1))

            # 선분에 대한 색상 설정
            marker.colors.append(ColorRGBA(1, 1, 1, 1))  # 선의 첫 번째 점에 대한 초록색
            marker.colors.append(ColorRGBA(1, 1, 1, 1))  # 선에 대한 초록색
        self.pub_marker.publish(marker)

if __name__ == '__main__':
    try:
        rospy.init_node('astar_map_node')
        path_pub = rospy.Publisher('/path', Path, queue_size=10)
        a_star = AStarPlanner()
        osm_file_path = rospy.get_param("astar_map_node/osm_file")
        ref_file_path = rospy.get_param("astar_map_node/ref_file")
        a_star.load_osm_data(osm_file_path, ref_file_path)
        server_dst_node_list_str = rospy.get_param('~server_dst_node_list')
        if len(server_dst_node_list_str) > 0:
            server_dst_node_list = [int(num) for num in server_dst_node_list_str.split(',')]
            a_star.set_dst_node_list(server_dst_node_list)

        rate = rospy.Rate(2)
        path_gen_flag = False
        path_nodes = []
        while not rospy.is_shutdown():
            a_star.visualize_graph()
            a_star.show_server_dst_nodes()
            rate.sleep()

            if a_star.start_init_flag == True and a_star.new_goal_flag == True:
                a_star.graph_setup()
                path_nodes = a_star.planning(a_star.start_id, a_star.goal_id)
                path_gen_flag = True

                a_star.new_goal_flag = False

                print("Generate path. path length: ", len(path_nodes))
                print("input start id: ", a_star.start_id, ", end id: ", a_star.goal_id)
                print("result start id: ", path_nodes[0], ", end id: ", path_nodes[-1])
                path_node_id_list_msg = Int32MultiArray()
                path_node_id_list_msg.data = path_nodes
                a_star.pub_path_node_id_list.publish(path_node_id_list_msg)
                a_star.show_path(path_nodes)

                # print("Start: ", data.pose.pose.position, "id: ", start_node.id)
            
            if path_gen_flag == True:
                # 경로 및 그래프 시각화
                a_star.show_path(path_nodes)
                
    except rospy.ROSInterruptException:
        pass