#include "local_planner/global_trajectory_planner.h"

GlobalTrajectoryPlanner::GlobalTrajectoryPlanner(ros::NodeHandle& nh) 
    : nh_(nh)
{

}

bool GlobalTrajectoryPlanner::init(void) 
{   
    // Parse parameters
    nh_.getParam("local_planner/trajectory_file_path", trajectory_file_path_);
    nh_.getParam("local_planner/pose_topic", pose_topic_);
    nh_.getParam("local_planner/travel_distance", travel_distance_);

    // Load global Trajectory
    if(!loadTrajectory(trajectory_file_path_)) {
        return false;
    }

    // Register Subscriber & Publisher
    sub_pose_ =  nh_.subscribe<nav_msgs::Odometry>(pose_topic_, 10, &GlobalTrajectoryPlanner::pose_handler, this);


    pub_local_trajectory_ = nh_.advertise<nav_msgs::Path>("local_planner/global_trajectory_planner/local", 1);
    pub_local_trajectory_viz_ = nh_.advertise<visualization_msgs::MarkerArray>("local_planner/global_trajectory_planner/local_viz", 1);
    pub_global_trajectory_viz_ = nh_.advertise<visualization_msgs::MarkerArray>("local_planner/global_trajectory_planner/global_viz", 1);

    return true;
}

bool GlobalTrajectoryPlanner::loadTrajectory(std::string trajectory_file_path) {
    // Load global trajectory file
    std::ifstream csv_ifstream(trajectory_file_path);
    if(csv_ifstream.fail()) {
        return false;
    }

    std::string cell;
    std::string	line;
    std::getline(csv_ifstream, line);
    while(std::getline(csv_ifstream, line)) {
        // Read a line
        std::stringstream lineStream(line);
        std::vector<std::string> cutCells;
        while (std::getline(lineStream, cell, ',')) {
            cutCells.push_back(cell);
        }
        if(cutCells.size() != 7) {
            continue;
        }

        // Parse data
        try {
            // Get data
            double x = std::stod(cutCells[0]);
            double y = std::stod(cutCells[1]);
            double z = std::stod(cutCells[2]);
            double roll = std::stod(cutCells[3]);
            double pitch = std::stod(cutCells[4]);
            double yaw = std::stod(cutCells[5]);
            double time = std::stod(cutCells[6]);

            // Transform data
            Eigen::Affine3d affinePose = toAffine3d(x,y,z,roll, pitch, yaw);
            trajectories_.push_back(affinePose);
            
        } catch (const std::string& expn) {
            std::cout << expn << ": The value in the cell is not a number \n";
        }
    }

    csv_ifstream.close();
    if(trajectories_.empty()) {
        return false;
    }

    return true;
}


void GlobalTrajectoryPlanner::pose_handler(const nav_msgs::OdometryConstPtr &msg_ptr) {
    nav_msgs::Odometry odom_msg = *msg_ptr;
    if(trajectories_.size() < 2){
        return;
    }

    // Get data
    double x = odom_msg.pose.pose.position.x;
    double y = odom_msg.pose.pose.position.y;
    double z = odom_msg.pose.pose.position.z;
    Eigen::Vector3d trans_pose;
    trans_pose << x, y, z;

    // Find Index of Nearest Element
    int trajectory_size = static_cast<int>(trajectories_.size());
    double minDist = __DBL_MAX__;
    int minIdx = -1;
    for(int idx = 0 ; idx < trajectory_size; idx++) {
        Eigen::Vector3d trans_trajectory = trajectories_[idx].translation();
        double dist = (trans_pose - trans_trajectory).norm();
        if(minDist > dist) {
            minDist = dist;
            minIdx = idx;
        }
    }

    // Find trajectory from minIdx
    std::vector<Eigen::Affine3d> findTrajectory;
    findTrajectory.push_back(trajectories_[minIdx]);
    double travelDist = 0.;
    for(int idxPres = minIdx, idxNext = minIdx+1; idxNext < trajectory_size; idxPres++, idxNext++) {
        findTrajectory.push_back(trajectories_[idxNext]);
        
        // Get traveling distance
        Eigen::Vector3d trans_pres = trajectories_[idxPres].translation();
        Eigen::Vector3d trans_next = trajectories_[idxNext].translation();
        double dist = (trans_pres - trans_next).norm();
        travelDist += dist;

        // Check traveling distance
        if(travelDist > travel_distance_) {
            break;
        }
    }

    // Publish local Trajectory
    publishTrajectory(pub_local_trajectory_, odom_msg.header.stamp, findTrajectory);

    // Publish local Trajectory for Visualization
    Eigen::Vector4d localNodeColorRGBA, localEdgeColorRGBA;
    localNodeColorRGBA << 1., 1., 0., 0.8;
    localEdgeColorRGBA << 1., 0., 1., 0.8;
    publishTrajectoryViz(pub_local_trajectory_viz_, odom_msg.header.stamp, findTrajectory, localNodeColorRGBA, localEdgeColorRGBA);

    // Publish global trajectory for Visualization
    Eigen::Vector4d globalNodeColorRGBA, globalEdgeColorRGBA;
    globalNodeColorRGBA << 0., 0.8, 0., 0.5;
    globalEdgeColorRGBA << 0.8, 0., 0., 0.5;
    publishTrajectoryViz(pub_global_trajectory_viz_, ros::Time::now(), trajectories_, globalNodeColorRGBA, globalEdgeColorRGBA);
}

void GlobalTrajectoryPlanner::publishTrajectory(const ros::Publisher& thisPub, const ros::Time& time, const std::vector<Eigen::Affine3d>& trajectory){
    if(trajectory.empty()) {
        return;
    }
    int node_size = static_cast<int>(trajectory.size());

    nav_msgs::Path path;
    path.poses.clear();
    path.header.frame_id = "map";
    path.header.stamp = time;
    path.poses.resize(trajectory.size());
    for(int idx = 0; idx < node_size; idx++){
        tf::poseEigenToMsg(trajectory[idx], path.poses[idx].pose);
    }

    thisPub.publish(path);
}

void GlobalTrajectoryPlanner::publishTrajectoryViz(const ros::Publisher& thisPub, const ros::Time& time, const std::vector<Eigen::Affine3d>& trajectory, const Eigen::Vector4d& nodeColorRGBA, const Eigen::Vector4d& edgeColorRGBA) {
    if(trajectory.empty()) {
        return;
    }
    int node_size = static_cast<int>(trajectory.size());
        
    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "map";
    markerNode.header.stamp = time;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "local_trajectory_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
    markerNode.color.r = nodeColorRGBA(0); 
    markerNode.color.g = nodeColorRGBA(1); 
    markerNode.color.b = nodeColorRGBA(2);
    markerNode.color.a = nodeColorRGBA(3);
    for(int idx = 0; idx < node_size; idx++ ){
        Eigen::Vector3d trans = trajectory[idx].translation();
        geometry_msgs::Point p; 
        p.x = trans(0);
        p.y = trans(1);
        p.z = trans(2);
        markerNode.points.push_back(p);
    }

    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "map";
    markerEdge.header.stamp = time;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "local_trajectory_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = edgeColorRGBA(0); 
    markerEdge.color.g = edgeColorRGBA(1); 
    markerEdge.color.b = edgeColorRGBA(2);
    markerEdge.color.a = edgeColorRGBA(3);

    for(int idxPres = 0, idxNext = 1; idxNext < node_size; idxPres++, idxNext++ ){
        geometry_msgs::Point p_pres; 
        Eigen::Vector3d trans_pres = trajectory[idxPres].translation();
        p_pres.x = trans_pres(0);
        p_pres.y = trans_pres(1);
        p_pres.z = trans_pres(2);
        markerEdge.points.push_back(p_pres);
        
        geometry_msgs::Point p_next; 
        Eigen::Vector3d trans_next = trajectory[idxNext].translation();
        p_next.x = trans_next(0);
        p_next.y = trans_next(1);
        p_next.z = trans_next(2);
        markerEdge.points.push_back(p_next);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    thisPub.publish(markerArray);
}

Eigen::Affine3d GlobalTrajectoryPlanner::toAffine3d(const double& x, const double& y, const double& z, const double& roll, const double& pitch, const double& yaw) {
    tf::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    tf::Matrix3x3 rotMat(q);
    Eigen::Matrix3d eigenRotMat(3,3);
    tf::matrixTFToEigen(rotMat, eigenRotMat);

    Eigen::Matrix4d eigenPose = Eigen::Matrix4d::Identity();
    eigenPose.block<3, 3>(0, 0) = eigenRotMat;
    eigenPose(0,3) = x;
    eigenPose(1,3) = y;
    eigenPose(2,3) = z;

    Eigen::Affine3d affinePose;
    affinePose = eigenPose;
    return affinePose;
}