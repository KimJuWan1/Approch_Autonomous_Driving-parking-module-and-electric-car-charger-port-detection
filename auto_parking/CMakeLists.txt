cmake_minimum_required(VERSION 3.0.2)
project(auto_parking)

## -----------------------------------------------------------------------------
## Find catkin and required components
## -----------------------------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  actionlib
  actionlib_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
  tf2_sensor_msgs
  visualization_msgs
)

## -----------------------------------------------------------------------------
## Messages / Actions
## -----------------------------------------------------------------------------
add_message_files(
  FILES
  ParkingSlot.msg
  ParkingSlotArray.msg
)

add_action_files(
  FILES
  ParkIn.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
)

## -----------------------------------------------------------------------------
## Catkin package
## -----------------------------------------------------------------------------
catkin_package(
  CATKIN_DEPENDS
    message_runtime
    actionlib
    actionlib_msgs
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    tf2_ros
    tf2_sensor_msgs
    visualization_msgs
)

## -----------------------------------------------------------------------------
## Install Python nodes
## -----------------------------------------------------------------------------
catkin_install_python(PROGRAMS
  nodes/parking_perception_node.py
  nodes/parking_planner_node.py
  nodes/parking_controller_node.py
  nodes/parking_manager_node.py
  nodes/cmd_vel_mux.py
  nodes/parking_debug_viz.py
  nodes/parking_slots_logger.py
  nodes/points_frame_relay.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## -----------------------------------------------------------------------------
## Install shareable resources (launch & rviz configs)
## -----------------------------------------------------------------------------
install(DIRECTORY launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
