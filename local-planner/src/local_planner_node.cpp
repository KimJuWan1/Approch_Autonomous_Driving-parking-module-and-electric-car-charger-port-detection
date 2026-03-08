#include "local_planner/global_trajectory_planner.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "local_planner");
    ros::NodeHandle nh;
    GlobalTrajectoryPlanner gt_planner(nh);
    gt_planner.init();
    
    ros::spin();

    return 0;
}
