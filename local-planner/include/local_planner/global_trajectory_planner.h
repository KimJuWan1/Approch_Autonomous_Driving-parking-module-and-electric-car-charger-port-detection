#include <vector>
#include <fstream>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

class GlobalTrajectoryPlanner {
public:
    GlobalTrajectoryPlanner(ros::NodeHandle& nh);
    bool init(void);

    bool loadTrajectory(std::string trajectory_file_path);

    // Handler
    void pose_handler(const nav_msgs::OdometryConstPtr &msg_ptr);

    // Publisher
    void publishTrajectory(const ros::Publisher& thisPub, const ros::Time& time, const std::vector<Eigen::Affine3d>& trajectory);
    void publishTrajectoryViz(const ros::Publisher& thisPub, const ros::Time& time, const std::vector<Eigen::Affine3d>& trajectory, const Eigen::Vector4d& nodeColorRGBA, const Eigen::Vector4d& edgeColorRGBA);

    // Miscellaneous
    Eigen::Affine3d toAffine3d(const double& x, const double& y, const double& z, const double& roll, const double& pitch, const double& yaw);
private:
    ros::NodeHandle nh_;

    // Parameters
    std::string trajectory_file_path_;
    std::string pose_topic_;
    double travel_distance_;

    // Internal varilables
    std::vector<Eigen::Affine3d> trajectories_;    

    // Subscriber & Publisher
    ros::Subscriber sub_pose_;

    ros::Publisher pub_local_trajectory_;
    ros::Publisher pub_local_trajectory_viz_;
    ros::Publisher pub_global_trajectory_viz_;
};