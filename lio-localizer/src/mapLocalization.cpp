#include "utility.h"

#include <ros/ros.h>

#include <lio_localizer/cloud_info.h>

#include <pcl/registration/gicp.h>
#include <Eigen/Dense>

#include "nanoflann.hpp"
#include "lidarFactor.hpp"

template <typename T>
struct NanoPointCloud
{
	struct NanoPoint
	{
		T  x,y,z;
	};

	std::vector<NanoPoint>  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline T kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		if (dim == 0) return pts[idx].x;
		else if (dim == 1) return pts[idx].y;
		else return pts[idx].z;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, NanoPointCloud<double>>,NanoPointCloud<double>,3> nano_kdtree_index;


enum POSE_INITIALIZATION_MODE {
    POSE_INITIALIZATION_SKIP = 0,
    POSE_INITIALIZATION_ON,
    POSE_INITIALIZATION_ALREADY
};

class MapLocalization : public ParamServer
{
public:  
    MapLocalization() {
        // Import Map
        std::string GlobalMapFile = mapFilePath + "/GlobalMap.pcd";
        global_map_.reset(new pcl::PointCloud<PointType>);
        if (pcl::io::loadPCDFile<PointType>(GlobalMapFile, *global_map_) == -1) {
            std::cout << "[Fail] Load PCD Map File!" << std::endl;
            return;
        }		
        std::cout << "[Success] Import PCD map!" << std::endl;

        // Import reference coordinate
        std::ifstream csvfile(mapFilePath + "/map_reference_coordinate.csv");
        if (csvfile.is_open()) {
            std::string line;
            std::getline(csvfile, line);
            std::getline(csvfile, line);

            std::istringstream sstream(line);
            std::string field;
            std::vector<std::string> fields;

            while (getline(sstream, field, ',')) {
                fields.push_back(field);
            }

            f_ref_lat_ = stof(fields[0]);
            f_ref_lon_ = stof(fields[1]);
            f_ref_height_ = stof(fields[2]);

            std::cout << "ref lat: " << f_ref_lat_
                    << ", lon: " << f_ref_lon_ << ", "
                    << ", height: " << f_ref_height_ << "\n";
        }

        // Flatten of map for GICP of initial positioning
        pcl::PointCloud<PointType>::Ptr flatten_map_pointcloud_ptr(new pcl::PointCloud<PointType>());
        flatten_map_pointcloud_ptr->points.resize(global_map_->size());
        for (int i = 0; i < static_cast<int>(global_map_->size()); i++) {
            flatten_map_pointcloud_ptr->points[i].x = global_map_->points[i].x;
            flatten_map_pointcloud_ptr->points[i].y = global_map_->points[i].y;
            flatten_map_pointcloud_ptr->points[i].z = 0.;
        }
        flatten_map_kdtree_.setInputCloud(flatten_map_pointcloud_ptr);
        std::cout << "[Success] Build kdtree for GICP!" << std::endl;

        // Load Map Data
        if(mapMatchingMethod == "LOAM"){
            // Load Corner Map
            std::string CornerMapFile = mapFilePath + "/CornerMap.pcd";
            corner_map_.reset(new pcl::PointCloud<PointType>);
            if (pcl::io::loadPCDFile<PointType>(CornerMapFile, *corner_map_) == -1) {
                std::cout << "[Fail] Load Corner Map File!" << std::endl;
                return;
            }		
            std::cout << "[Success] Import Corner map!" << std::endl;

            corner_map_kdtree_.setInputCloud(corner_map_);
            std::cout << "[Success] Build Corner kdtree for LOAM Map Matching!" << std::endl;

            // Load Surface Map
            std::string SurfaceMapFile =  mapFilePath + "/SurfMap.pcd";
            surface_map_.reset(new pcl::PointCloud<PointType>);
            if (pcl::io::loadPCDFile<PointType>(SurfaceMapFile, *surface_map_) == -1) {
                std::cout << "[Fail] Load Surface Map File!" << std::endl;
                return;
            }		
            std::cout << "[Success] Import Surface map!" << std::endl;

            surface_map_kdtree_.setInputCloud(surface_map_);
            std::cout << "[Success] Build Surface kdtree for LOAM Map Matching!" << std::endl;

        }
        else if(mapMatchingMethod == "ICP"){
            global_map_kdtree_.setInputCloud(global_map_);
            std::cout << "[Success] Build kdtree for ICP Map Matching!" << std::endl;
        }
        else if(mapMatchingMethod == "GICP"){

        }
        else {
            std::cout << "[Fail] Map Matching Option is invalid!" << std::endl;
        }

        
        // Voxel Grid Filter
        voxel_grid_filter_.setLeafSize(voxelGridFilterSize, voxelGridFilterSize, voxelGridFilterSize);

        // register Subscriber
        sub_initial_pose_ = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 10, &MapLocalization::initial_pose_handler, this, ros::TransportHints().tcpNoDelay());
        sub_cloud_ = nh.subscribe<lio_localizer::cloud_info>("lio_localizer/feature/cloud_info", 1, &MapLocalization::cloud_info_handler, this, ros::TransportHints().tcpNoDelay());
        sub_gnss_ = nh.subscribe<sensor_msgs::NavSatFix>(gpsTopic, 200, &MapLocalization::gnss_handler, this, ros::TransportHints().tcpNoDelay());

        // Register Publisher
        pub_initialized_pose_ = nh.advertise<nav_msgs::Odometry> ("lio_localizer/odometry/initialization", 1);
        pub_optimized_pose_ = nh.advertise<nav_msgs::Odometry> ("lio_localizer/odometry/optimization", 1);
        pub_optimized_pose_wgs84_ = nh.advertise<nav_msgs::Odometry> ("lio_localizer/odometry/optimization_wgs84", 1);
        pub_optimized_pose_path_ = nh.advertise<nav_msgs::Path>("lio_localizer/odometry/optimization_path", 1);
        pub_global_map_ = nh.advertise<sensor_msgs::PointCloud2>("lio_localizer/localization/global_map", 1); 
        pub_corner_map_ = nh.advertise<sensor_msgs::PointCloud2>("lio_localizer/localization/corner_map", 1); 
        pub_surface_map_ = nh.advertise<sensor_msgs::PointCloud2>("lio_localizer/localization/surface_map", 1); 
        pub_dewkew_cloud_ = nh.advertise<sensor_msgs::PointCloud2> ("lio_localizer/localization/cloud_deskewed", 1);
        pub_corner_cloud_ = nh.advertise<sensor_msgs::PointCloud2> ("lio_localizer/localization/cloud_corner", 1);
        pub_surface_cloud_ = nh.advertise<sensor_msgs::PointCloud2> ("lio_localizer/localization/cloud_surface", 1);

        // Map Flag
        flag_map_initialized_ = true;
        int a = 0;

        // Initial pose configure
        if(cfg_use_initial_pose_setup == true)
        {
            PointType searchPoint;
            searchPoint.x = cfg_initial_pose_m_deg[0];
            searchPoint.y = cfg_initial_pose_m_deg[1];
            searchPoint.z = cfg_initial_pose_m_deg[2];

            // Find only one Nearest Neighbor
            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            if(flatten_map_kdtree_.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) < 1)  return;
            PointType nearPoint = global_map_->at(pointIdxNKNSearch[0]);
            
            Eigen::Translation3d translation(nearPoint.x, nearPoint.y, nearPoint.z);
            Eigen::AngleAxisd rollAngle(cfg_initial_pose_m_deg[3] / 180.0 * M_PI, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd pitchAngle(cfg_initial_pose_m_deg[4] / 180.0 * M_PI, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yawAngle(cfg_initial_pose_m_deg[5] / 180.0 * M_PI, Eigen::Vector3d::UnitZ());

            Eigen::Quaterniond quaternion = yawAngle * pitchAngle * rollAngle;
            initial_pose_ = translation * quaternion;

            flag_new_initial_pose_ = true;
        }
    }

    // Member Functions
    POSE_INITIALIZATION_MODE initial_pose_correction(pcl::PointCloud<PointType>::Ptr meas_ptr, Eigen::Affine3d& corrected_pose) {
        std::lock_guard<std::mutex> lock(initial_pose_lock_);
        if(!flag_map_initialized_) {
            return POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_SKIP;
        }

        if(!flag_new_initial_pose_) {
            if(flag_initialized_) {
                return POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_ALREADY;
            }
            else {
                return POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_SKIP;
            }
        }
        if(initializationMethod == "GICP"){
            std::cout << "start initial pose correction \n";
            // Crop map points for speed!
            Eigen::Vector3d trans = initial_pose_.translation();
            float boundary = initialMapBoundary * initialMapBoundary;
            pcl::PointCloud<PointType>::Ptr crop_map_ptr(new pcl::PointCloud<PointType>());
            crop_map_ptr->points.reserve(global_map_->points.size());
            for(const auto& point: *global_map_){
                double dx = point.x - trans(0);
                double dy = point.y - trans(1);

                if ( dx * dx + dy * dy < boundary){
                crop_map_ptr->push_back(point);
                }
            }

            // Apply GICP from initial pose
            pcl::PointCloud<PointType> final_pointcloud;
            pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
            gicp.setTransformationEpsilon(0.00001);
            gicp.setMaximumIterations(initialNumOfGicpIteration);
            gicp.setInputSource(meas_ptr);
            gicp.setInputTarget(crop_map_ptr);
            gicp.align(final_pointcloud, initial_pose_.matrix().cast<float>());
            if(gicp.hasConverged() != true) {
                if(flag_initialized_) {
                    return POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_ALREADY;
                }
                else {
                    return POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_SKIP;
                }
            }

            corrected_pose = gicp.getFinalTransformation().cast<double>();
            
            flag_new_initial_pose_ = false;
            flag_initialized_ = true;

            // corrected_pose.linear().eulerAngles(2, 1, 0)
            // std::cout << "init ori x: " << drawed_pose.orientation.x << ", q.y: " << drawed_pose.orientation.y << ", q.z: " << drawed_pose.orientation.z << ", q.w: " << drawed_pose.orientation.w << "\n";
            std::cout << "Done initial pose correction \n";
            return POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_ON;
        }
        else {
            corrected_pose = initial_pose_;
            flag_new_initial_pose_ = false;
            flag_initialized_ = true;
            return POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_ON;
        }
    }

    nav_msgs::Odometry get_odometry(Eigen::Affine3d pose, ros::Time ros_time, std::string frame_id) {
        // Conversion
        Eigen::Quaternion<double> quaternion(pose.linear());
        Eigen::Vector3d translation = pose.translation();

        // Set Odometry
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros_time;
        odom_msg.header.frame_id = frame_id;
        odom_msg.pose.pose.position.x = translation(0);
        odom_msg.pose.pose.position.y = translation(1);
        odom_msg.pose.pose.position.z = translation(2);
        odom_msg.pose.pose.orientation.w = quaternion.w();
        odom_msg.pose.pose.orientation.x = quaternion.x();
        odom_msg.pose.pose.orientation.y = quaternion.y();
        odom_msg.pose.pose.orientation.z = quaternion.z();
        return odom_msg;
    }

    void add_corner_cost_factor(const Eigen::Affine3d& pose_optimized, const pcl::PointCloud<PointType>::Ptr &pc_in, const pcl::PointCloud<PointType>::Ptr &map_in, const pcl::KdTreeFLANN<PointType>& kdtree, ceres::Problem &problem, ceres::LossFunction *loss_function, double* parameters)
    {
        // Transform Map Point Cloud
        pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*pc_in, *transformed_cloud, pose_optimized.matrix());

        int corner_num = 0;
        for (int i = 0; i < (int)pc_in->points.size(); i++) 
        {
            PointType point_temp = transformed_cloud->points[i];

            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            
            // Search nearest 5 points
            kdtree.nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
            if (pointSearchSqDis[4] < pointCorrespondenceDist)
            {
                std::vector<Eigen::Vector3d> nearCorners;
                Eigen::Vector3d center(0, 0, 0);
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d tmp(map_in->points[pointSearchInd[j]].x,
                                        map_in->points[pointSearchInd[j]].y,
                                        map_in->points[pointSearchInd[j]].z);
                    center = center + tmp;
                    nearCorners.push_back(tmp);
                }
                // Compute corner direction
                center = center / 5.0;
                Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                    covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                }
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);

                // After checking line information using eigen values, add cost function
                if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                {
                    Eigen::Vector3d point_on_line = center;
                    Eigen::Vector3d point_a, point_b;
                    point_a = 0.1 * unit_direction + point_on_line;
                    point_b = -0.1 * unit_direction + point_on_line;
                    ceres::CostFunction *cost_function = EdgeAnalyticCostFunction::Create(curr_point, point_a, point_b);
                    problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                    corner_num++;
                }
            }
        }
        // std::cout << "Number of Corner Points: " << corner_num << std::endl;
    }

    void add_surface_cost_factor(const Eigen::Affine3d& pose_optimized, const pcl::PointCloud<PointType>::Ptr &pc_in, const pcl::PointCloud<PointType>::Ptr &map_in, const pcl::KdTreeFLANN<PointType>& kdtree, ceres::Problem &problem, ceres::LossFunction *loss_function, double* parameters)
    {
        // Transform Map Point Cloud
        pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*pc_in, *transformed_cloud, pose_optimized.matrix());

        int surf_num = 0;
        for (int i = 0; i < (int)transformed_cloud->points.size(); i++)
        {
            PointType point_temp = transformed_cloud->points[i];
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtree.nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            if (pointSearchSqDis[4] < pointCorrespondenceDist)
            {

                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
                    matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
                    matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
                }
                // find the norm of plane
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double negative_OA_dot_norm = 1 / norm.norm();
                norm.normalize();

                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    // if OX * n > 0.2, then plane is not fit well
                    if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                             norm(1) * map_in->points[pointSearchInd[j]].y +
                             norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }
                Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
                if (planeValid)
                {
                    ceres::CostFunction *cost_function = SurfNormAnalyticCostFunction::Create(curr_point, norm, negative_OA_dot_norm);
                    problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);

                    surf_num++;
                }
            }
        }
        // std::cout << "Number of Surface Points: " << surf_num << std::endl;
    }

    bool map_matching_loam(Eigen::Affine3d pose_predicted, pcl::PointCloud<PointType>::Ptr corner_cloud, pcl::PointCloud<PointType>::Ptr surface_cloud, Eigen::Affine3d& pose_matched) {
        if(!flag_map_initialized_) {
            return false;
        }

        // Generate optimized target
        double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
        Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
        Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

        // Make optimization problem
        ceres::LossFunction *loss_function = NULL;
        // ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(parameters, 4, q_parameterization);
        problem.AddParameterBlock(parameters + 4, 3);

        TicToc t_matching;
        Eigen::Affine3d pose_optimized = pose_predicted;
        for( int i = 0 ; i < numOfFindCorrespondence; i++) {
            // Pose conversion to quterniond and vector3d
            Eigen::Quaternion<double> quaternion(pose_optimized.linear());
            Eigen::Vector3d translation = pose_optimized.translation();
            q_w_curr = quaternion;
            t_w_curr = translation;

            // Add Corner and Surface Edge
            add_corner_cost_factor(pose_optimized, corner_cloud, corner_map_, corner_map_kdtree_, problem, loss_function, parameters);
            add_surface_cost_factor(pose_optimized, surface_cloud, surface_map_, surface_map_kdtree_, problem, loss_function, parameters);

            // Solve the solver
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = numOfOptimizationIteration;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            options.num_threads = numOfOptimizationCores;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            // Update Pose
            q_w_curr.normalize();
            pose_optimized.translation() = t_w_curr;
            pose_optimized.linear() = q_w_curr.toRotationMatrix();
        }

        // Get Final Pose
        pose_matched = pose_optimized;
        double processing_time_ms = t_matching.toc();
        std::cout << "Processing Time: " << processing_time_ms << " ms" << std::endl;

        return true;
    }

    std::vector<std::pair<int,int>> make_correspondence(Eigen::Affine3d pose_input, pcl::PointCloud<PointType>::Ptr meas_cloud, pcl::KdTreeFLANN<PointType>& kdtree) {
        std::vector<std::pair<int,int>> pairs;

        // Transform Map Point Cloud
        pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*meas_cloud, *transformed_cloud, pose_input.matrix());

        // Find nearest map point from measured point
        for (int src_idx = 0; src_idx < static_cast<int>(transformed_cloud->size());src_idx++) {
            PointType searchPoint;
            searchPoint.x = (*transformed_cloud)[src_idx].x;
            searchPoint.y = (*transformed_cloud)[src_idx].y;
            searchPoint.z = (*transformed_cloud)[src_idx].z;

            // Find only one Nearest Neighbor
            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            if(kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) < 1) {
                continue;
            }
            if(pointNKNSquaredDistance[0] > pointCorrespondenceDist * pointCorrespondenceDist) {
                continue;
            }
            int map_idx = pointIdxNKNSearch[0];

            pairs.push_back(std::make_pair(src_idx, map_idx));
        }
        return pairs;
    }

    bool map_matching_icp(Eigen::Affine3d pose_predicted, pcl::PointCloud<PointType>::Ptr meas_cloud, Eigen::Affine3d& pose_matched) {
        if(!flag_map_initialized_) {
            return false;
        }

        // Generate optimized target
        double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
        Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
        Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

        // Make optimization problem
        ceres::LossFunction *loss_function = NULL;
        // ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(parameters, 4, q_parameterization);
        problem.AddParameterBlock(parameters + 4, 3);

        TicToc t_matching;
        Eigen::Affine3d pose_optimized = pose_predicted;
        for( int i = 0 ; i < numOfFindCorrespondence; i++) {
            // Pose conversion to quterniond and vector3d
            Eigen::Quaternion<double> quaternion(pose_optimized.linear());
            Eigen::Vector3d translation = pose_optimized.translation();
            q_w_curr = quaternion;
            t_w_curr = translation;

            // Find Correspondence
            std::vector<std::pair<int,int>> pairs = make_correspondence(pose_optimized, meas_cloud, global_map_kdtree_);
            
            // Find nearest map point from measured point
            for(const auto& pair: pairs){
                // Get Index
                int src_idx = pair.first;
                int map_idx = pair.second;

                // Add Residual
                Eigen::Vector3d curr_point((*meas_cloud)[src_idx].x, (*meas_cloud)[src_idx].y, (*meas_cloud)[src_idx].z);
                Eigen::Vector3d map_point((*global_map_)[map_idx].x, (*global_map_)[map_idx].y, (*global_map_)[map_idx].z);
                ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, map_point);
                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
            }

            // Solve the solver
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = numOfOptimizationIteration;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            options.num_threads = numOfOptimizationCores;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            
            // Update Pose
            q_w_curr.normalize();
            pose_optimized.translation() = t_w_curr;
            pose_optimized.linear() = q_w_curr.toRotationMatrix();
        }

        // Get Final Pose
        pose_matched = pose_optimized;
        double processing_time_ms = t_matching.toc();
        std::cout << "Processing Time: " << processing_time_ms << " ms" << std::endl;

        return true;
    }

    bool map_matching_gicp(Eigen::Affine3d pose_predicted, pcl::PointCloud<PointType>::Ptr meas_cloud, Eigen::Affine3d& pose_matched) {
        if(!flag_map_initialized_) {
            return false;
        }

        // Crop map points for speed!
        Eigen::Vector3d trans = pose_predicted.translation();
        float boundary = initialMapBoundary * initialMapBoundary;
        pcl::PointCloud<PointType>::Ptr crop_map_ptr(new pcl::PointCloud<PointType>());
        crop_map_ptr->points.reserve(global_map_->points.size());
        for(const auto& point: *global_map_){
            double dx = point.x - trans(0);
            double dy = point.y - trans(1);

            if ( dx * dx + dy * dy < boundary){
                crop_map_ptr->push_back(point);
            }
        }

        TicToc t_matching;
        pcl::PointCloud<PointType> final_pointcloud;
        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
        gicp.setTransformationEpsilon(0.00001);
        gicp.setMaximumIterations(numOfOptimizationIteration);
        gicp.setInputSource(meas_cloud);
        gicp.setInputTarget(crop_map_ptr);
        gicp.align(final_pointcloud, pose_predicted.matrix().cast<float>());
        if(gicp.hasConverged() != true) {
            return false;
        }
        double processing_time_ms = t_matching.toc();
        std::cout << "Matching of Points [" << meas_cloud->size() << "]: " << processing_time_ms << " ms" << std::endl;
        pose_matched = gicp.getFinalTransformation().cast<double>();
        return true;
    }

    void publish_odometry(Eigen::Affine3d pose, ros::Time ros_time, std::string frame_id) {
        nav_msgs::Odometry optimized_pose = get_odometry(pose, ros_time, frame_id);
        pub_optimized_pose_.publish(optimized_pose);

        // optimized_pose_path_.
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = optimized_pose.header;
        pose_stamped.pose.position = optimized_pose.pose.pose.position;
        pose_stamped.pose.orientation = optimized_pose.pose.pose.orientation;
        optimized_pose_path_.poses.push_back(pose_stamped);
        optimized_pose_path_.header = optimized_pose.header;
        pub_optimized_pose_path_.publish(optimized_pose_path_);

        GeoPoint ref;
        ref.lat = f_ref_lat_;
        ref.lon = f_ref_lon_;
        ref.height = f_ref_height_;
        ENUPoint src;
        src.x = optimized_pose.pose.pose.position.x;
        src.y = optimized_pose.pose.pose.position.y;
        src.z = optimized_pose.pose.pose.position.z;
        GeoPoint dst;
        ENUToWGS84(&ref, &src, &dst);

        nav_msgs::Odometry optimized_pose_wgs84;
        optimized_pose_wgs84.header = optimized_pose.header;
        optimized_pose_wgs84.pose.pose.position.x = dst.lat;
        optimized_pose_wgs84.pose.pose.position.y = dst.lon;
        optimized_pose_wgs84.pose.pose.position.z = dst.height;
        optimized_pose_wgs84.pose.pose.orientation = optimized_pose.pose.pose.orientation;
        pub_optimized_pose_wgs84_.publish(optimized_pose_wgs84);

    }

    void publish_transform(Eigen::Affine3d pose, ros::Time ros_time, std::string parent_frame, std::string child_frame) {
        static tf::TransformBroadcaster br;
        // tf 를 broad cast할 행렬
        tf::Transform transform;
        tf::transformEigenToTF (pose, transform);

        // map - base_link의 tf를 broadcast
        br.sendTransform(tf::StampedTransform(transform, ros_time, parent_frame, child_frame));
    }

    // Handler
    void initial_pose_handler(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg_ptr) {
        if(!flag_map_initialized_) {
            return;
        }
        ROS_INFO("\033[1;32m ======= Draw Initial Pose ======= \033[0m");
        PointType searchPoint;
        searchPoint.x = msg_ptr->pose.pose.position.x;
        searchPoint.y = msg_ptr->pose.pose.position.y;
        searchPoint.z = msg_ptr->pose.pose.position.z;

        // Find only one Nearest Neighbor
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
        if(flatten_map_kdtree_.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) < 1)  return;
        PointType nearPoint = global_map_->at(pointIdxNKNSearch[0]);

        // Get drawed pose
        geometry_msgs::Pose drawed_pose;
        drawed_pose.position.x = nearPoint.x;
        drawed_pose.position.y = nearPoint.y;
        drawed_pose.position.z = nearPoint.z;
        drawed_pose.orientation = msg_ptr->pose.pose.orientation;

        std::cout << "init ori q.x: " << drawed_pose.orientation.x << ", q.y: " << drawed_pose.orientation.y << ", q.z: " << drawed_pose.orientation.z << ", q.w: " << drawed_pose.orientation.w << "\n";
        
        std::lock_guard<std::mutex> lock(initial_pose_lock_);
        tf::poseMsgToEigen(drawed_pose, initial_pose_);
        flag_new_initial_pose_ = true;
    }
    
    void cloud_info_handler(const lio_localizer::cloud_info::ConstPtr &msg_ptr) {   
        // std::cout << "\033[1;36m[mapLocalization] cloud_info_handler: " << std::setprecision(30) << msg_ptr->header.stamp.toSec() << "\033[0m" << std::endl;     
        // Get Time
        ros::Time time = msg_ptr->header.stamp;

        // Get deskewed Point cloud
        pcl::PointCloud<PointType>::Ptr cloud_deskewed(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr filter_cloud_deskewed(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(msg_ptr->cloud_deskewed, *cloud_deskewed);
        voxel_grid_filter_.setInputCloud(cloud_deskewed);
        voxel_grid_filter_.filter(*filter_cloud_deskewed);

        pcl::PointCloud<PointType>::Ptr cloud_corner(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr filter_cloud_corner(new pcl::PointCloud<PointType>());

        pcl::PointCloud<PointType>::Ptr cloud_surface(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr filter_cloud_surface(new pcl::PointCloud<PointType>());
        if(mapMatchingMethod == "LOAM"){
            pcl::fromROSMsg(msg_ptr->cloud_corner, *cloud_corner);
            voxel_grid_filter_.setInputCloud(cloud_corner);
            voxel_grid_filter_.filter(*filter_cloud_corner);

            pcl::fromROSMsg(msg_ptr->cloud_surface, *cloud_surface);
            voxel_grid_filter_.setInputCloud(cloud_deskewed);
            voxel_grid_filter_.filter(*filter_cloud_surface);
        }

        // Initialization of pose
        Eigen::Affine3d corrected_pose = Eigen::Affine3d::Identity();
        POSE_INITIALIZATION_MODE pose_initialization_mode = initial_pose_correction(filter_cloud_deskewed, corrected_pose);
        if(pose_initialization_mode == POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_SKIP){
            return;
        }
        else if(pose_initialization_mode == POSE_INITIALIZATION_MODE::POSE_INITIALIZATION_ON) {
            std::cout << "\033[1;32m[mapLocalization] Pose Initialziation \033[0m" << std::endl;
            nav_msgs::Odometry initialized_pose = get_odometry(corrected_pose, time, "map");
            pub_initialized_pose_.publish(initialized_pose);

            // Update prev pose
            prev_pose_from_opt_ = corrected_pose;
            
            // Publish pose
            publish_odometry(prev_pose_from_opt_, time, "map");
            publish_transform(prev_pose_from_opt_, time,  "map", "base_link");

            // Publish Point cloud
            publishCloud(pub_dewkew_cloud_, filter_cloud_deskewed, time, "base_link");
            publishCloud(pub_corner_cloud_, filter_cloud_corner, time, "base_link");
            publishCloud(pub_surface_cloud_, filter_cloud_surface, time, "base_link");
            return;
        }

        // std::cout << "\033[1;32m===== Map Matching Process =====\033[0m" << std::endl;

        // std::cout << "imuAvailable Flag: "<< msg_ptr->imuAvailable << std::endl;
        // std::cout << "odomAvailable Flag: "<< msg_ptr->odomAvailable << std::endl;

        // Update Initial Odometry Pose
        if(msg_ptr->odomAvailable) {
            static bool odomInitialUpdateFlag = true;
            if(odomInitialUpdateFlag){
                odomInitialUpdateFlag = false;
                prev_pose_from_imu_ = pcl::getTransformation(msg_ptr->initialGuessX, msg_ptr->initialGuessY, msg_ptr->initialGuessZ, msg_ptr->initialGuessRoll, msg_ptr->initialGuessPitch, msg_ptr->initialGuessYaw).cast<double>();
            }
        }

        // Get predicted pose from IMU
        Eigen::Affine3d pose_from_imu = pcl::getTransformation(msg_ptr->initialGuessX, msg_ptr->initialGuessY, msg_ptr->initialGuessZ, msg_ptr->initialGuessRoll, msg_ptr->initialGuessPitch, msg_ptr->initialGuessYaw).cast<double>();
        Eigen::Affine3d pose_incremental = prev_pose_from_imu_.inverse() * pose_from_imu;
        Eigen::Affine3d pose_predicted = prev_pose_from_opt_ * pose_incremental;

        // Map Matching
        Eigen::Affine3d pose_from_opt = Eigen::Affine3d::Identity();
        if(mapMatchingMethod == "LOAM"){
            if (!map_matching_loam(pose_predicted, filter_cloud_corner, filter_cloud_surface, pose_from_opt) ){
                std::cout << "[Fail] LOAM Map matching" << std::endl;
                return;
            }
        }
        else if(mapMatchingMethod == "ICP"){
            if (!map_matching_icp(pose_predicted, filter_cloud_deskewed, pose_from_opt) ){
                std::cout << "[Fail] ICP Map matching" << std::endl;
                return;
            }
        }
        else if(mapMatchingMethod == "GICP") {
            if (!map_matching_gicp(pose_predicted, filter_cloud_deskewed, pose_from_opt) ){
                std::cout << "[Fail] GICP Map matching" << std::endl;
                return;
            }
        }
        else {
            std::cout << "[Fail] Map Matching Option is invalid!" << std::endl;
            return;
        }

        // Upate Pose
        prev_pose_from_imu_ = pose_from_imu;
        prev_pose_from_opt_ = pose_from_opt;

        // Publish pose
        publish_odometry(pose_from_opt, time, "map");
        publish_transform(pose_from_opt, time,  "map", "base_link");

        // Publish Point cloud
        publishCloud(pub_dewkew_cloud_, filter_cloud_deskewed, time, "base_link");
        publishCloud(pub_corner_cloud_, filter_cloud_corner, time, "base_link");
        publishCloud(pub_surface_cloud_, filter_cloud_surface, time, "base_link");
    }

    void gnss_handler(const sensor_msgs::NavSatFix::ConstPtr &msg_ptr) {
        if(!flag_map_initialized_) {
            return;
        }

        // Set initial pose
        if(flag_initialized_ == false)
        {
            // float noise_x = msg_ptr->position_covariance[0];
            // float noise_y = msg_ptr->position_covariance[4];
            // float noise_z = msg_ptr->position_covariance[8];
            // if (noise_x < gpsCovThreshold && noise_y < gpsCovThreshold)
            // {
            //     GeoPoint ref;
            //     ref.lat = f_ref_lat_;
            //     ref.lon = f_ref_lon_;
            //     ref.height = f_ref_height_;
            //     GeoPoint src;
            //     src.lat = msg_ptr->latitude;
            //     src.lon = msg_ptr->longitude;
            //     src.height = msg_ptr->altitude;
            //     ENUPoint dst;
            //     WGS84ToENU(&ref, &src, &dst);

            //     PointType searchPoint;
            //     searchPoint.x = dst.x;
            //     searchPoint.y = dst.y;
            //     searchPoint.z = dst.z;

            //     ROS_INFO("\033[1;32m ======= Set Initial Pose From GNSS ======= \033[0m");

            //     // Find only one Nearest Neighbor
            //     std::vector<int> pointIdxNKNSearch;
            //     std::vector<float> pointNKNSquaredDistance;
            //     if(flatten_map_kdtree_.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) < 1)  return;
            //     PointType nearPoint = global_map_->at(pointIdxNKNSearch[0]);

            //     // Get drawed pose
            //     geometry_msgs::Pose drawed_pose;
            //     drawed_pose.position.x = nearPoint.x;
            //     drawed_pose.position.y = nearPoint.y;
            //     drawed_pose.position.z = nearPoint.z;
            //     // drawed_pose.orientation = msg_ptr->pose.pose.orientation;
                
            //     std::lock_guard<std::mutex> lock(initial_pose_lock_);
            //     tf::poseMsgToEigen(drawed_pose, initial_pose_);
            //     flag_new_initial_pose_ = true;
            // }
        }
    }

    void visualizeGlobalMapThread() {
        ros::Rate loop_rate(ros::Duration(5));
        while (ros::ok()){
            if(!flag_map_initialized_) continue;

            if(global_map_ != NULL){
                sensor_msgs::PointCloud2 output; 
                pcl::toROSMsg(*global_map_, output); 
                output.header.frame_id = "map";
                pub_global_map_.publish(output);
            }
            if(corner_map_ != NULL){
                sensor_msgs::PointCloud2 output; 
                pcl::toROSMsg(*corner_map_, output); 
                output.header.frame_id = "map";
                pub_corner_map_.publish(output);
            }
            if(surface_map_ != NULL){
                sensor_msgs::PointCloud2 output; 
                pcl::toROSMsg(*surface_map_, output); 
                output.header.frame_id = "map";
                pub_surface_map_.publish(output);
            }
            loop_rate.sleep();
        }
    }

    void printAffine3d(std::string name, Eigen::Affine3d aff) {
        Eigen::Vector3d ea = aff.rotation().eulerAngles(2, 1, 0);
        Eigen::Vector3d trans = aff.translation();
        std::cout << name << "[xyzrpy]: " << trans(0) << ", " << trans(1) << ", " << trans(2) << ", " << ea(2) << ", " << ea(1) << ", " << ea(0) << std::endl;
    }

public:
    // Member Variables
    ros::Subscriber sub_initial_pose_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_gnss_;

    ros::Publisher pub_initialized_pose_;
    ros::Publisher pub_optimized_pose_;
    ros::Publisher pub_optimized_pose_wgs84_;
    ros::Publisher pub_optimized_pose_path_;
    ros::Publisher pub_global_map_;
    ros::Publisher pub_corner_map_;
    ros::Publisher pub_surface_map_;
    ros::Publisher pub_dewkew_cloud_;
    ros::Publisher pub_corner_cloud_;
    ros::Publisher pub_surface_cloud_;

    bool flag_map_initialized_ = false;
    pcl::PointCloud<PointType>::Ptr global_map_ = NULL;
    pcl::PointCloud<PointType>::Ptr corner_map_ = NULL;
    pcl::PointCloud<PointType>::Ptr surface_map_ = NULL;
    pcl::KdTreeFLANN<PointType> flatten_map_kdtree_;
    pcl::KdTreeFLANN<PointType> surface_map_kdtree_;
    pcl::KdTreeFLANN<PointType> corner_map_kdtree_;
    pcl::KdTreeFLANN<PointType> global_map_kdtree_;
    pcl::VoxelGrid<PointType> voxel_grid_filter_;

    // Initialization pose
    std::mutex initial_pose_lock_;
    bool flag_new_initial_pose_ = false;
    bool flag_initialized_ = false;
    Eigen::Affine3d initial_pose_ = Eigen::Affine3d::Identity();

    // Pose
    Eigen::Affine3d prev_pose_from_opt_ = Eigen::Affine3d::Identity();
    Eigen::Affine3d prev_pose_from_imu_ = Eigen::Affine3d::Identity();
    nav_msgs::Path optimized_pose_path_;

    // Gnss
    float f_ref_lat_ = 0;
    float f_ref_lon_ = 0;
    float f_ref_height_ = 0;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lio_localizer_mapLocalization");

    MapLocalization ML;
    std::thread visualizeMapThread(&MapLocalization::visualizeGlobalMapThread, &ML);

    ROS_INFO("\033[1;32m----> Map Localization Started.\033[0m");
    ros::spin();

    return 0;
}
