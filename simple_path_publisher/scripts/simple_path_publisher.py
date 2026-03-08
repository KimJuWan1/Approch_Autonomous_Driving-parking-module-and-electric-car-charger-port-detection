import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import csv
import tf

def read_csv_to_path(csv_file_path):
    path = Path()
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader, None)  # 헤더를 건너뛰기 위함
        for row in csv_reader:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(row[0])
            pose.pose.position.y = float(row[1])
            pose.pose.position.z = float(row[2])
            quaternion = tf.transformations.quaternion_from_euler(
                float(row[3]),  # roll
                float(row[4]),  # pitch
                float(row[5])   # yaw
            )
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            path.poses.append(pose)
    return path

def publish_path(path):
    rospy.init_node('path_publisher', anonymous=True)
    path_pub = rospy.Publisher('simple_path', Path, queue_size=10)
    rate = rospy.Rate(5)  # 10hz
    while not rospy.is_shutdown():
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        path_pub.publish(path)
        rate.sleep()

if __name__ == '__main__':
    csv_file_path = '/home/aimlab/waveai_ws/src/lio-localizer/map/campus_0229_0-1226/trajectory_short.csv'  # CSV 파일 경로 설정
    path = read_csv_to_path(csv_file_path)
    publish_path(path)