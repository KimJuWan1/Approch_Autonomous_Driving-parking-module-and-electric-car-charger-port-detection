cmake_minimum_required(VERSION 3.0.2)
project(parking_occupancy)

find_package(catkin REQUIRED COMPONENTS
  rospy sensor_msgs geometry_msgs std_msgs visualization_msgs tf tf2_ros
)

catkin_package()

catkin_install_python(
  PROGRAMS
    nodes/fixed_parking_occupancy.py
    nodes/slot_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
