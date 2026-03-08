cmake_minimum_required(VERSION 3.0.2)
project(parking_executor)
find_package(catkin REQUIRED COMPONENTS rospy geometry_msgs std_msgs nav_msgs tf2_ros visualization_msgs)
catkin_package()
catkin_install_python(PROGRAMS
  scripts/parking_controller.py
  # scripts/parking_navigator.py   # 사용 시 주석 해제
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

