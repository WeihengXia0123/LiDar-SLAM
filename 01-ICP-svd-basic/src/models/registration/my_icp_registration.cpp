/*
 * @Description: my_ICP 匹配模块
 * @Author: Weiheng Xia
 * @Date: 2021-01-28 13:10:00
 */
#include "lidar_localization/models/registration/my_icp_registration.hpp"

#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <chrono>

namespace lidar_localization {

    myICPRegistration::myICPRegistration(const YAML::Node& node) {
        
        float min_threshold = node["min_threshold"].as<float>();
        int max_iter = node["max_iteration"].as<int>();
        float max_corr_dist = node["max_corr_dist"].as<float>();

        SetRegistrationParam(min_threshold, max_iter, max_corr_dist);
    }

    myICPRegistration::myICPRegistration(float min_threshold, int max_iter, float max_corr_dist) {
        SetRegistrationParam(min_threshold, max_iter, max_corr_dist);
    }

    bool myICPRegistration::SetRegistrationParam(
        float min_threshold,
        int max_iter,
        float max_corr_dist
    ) {
        this->minThreshold = min_threshold;
        this->maxIterations = max_iter;
        this->max_corr_dist = max_corr_dist;
        
        std::cout << "minThreshold: " << this->minThreshold << std::endl
                << "max_iter: " << this->maxIterations << std::endl
                << "max_corr_dist: " << this->max_corr_dist << std::endl
                << std::endl;

        return true;
    }

    bool myICPRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
        /*set input target cloud*/
        this->input_target_cloud = input_target;
        /*build kd tree*/
        this->kd_tree.setInputCloud(input_target);

        return true;
    }

    bool myICPRegistration::ScanMatch(const CloudData::CLOUD_PTR& input_source, 
                                    const Eigen::Matrix4f& predict_pose, 
                                    CloudData::CLOUD_PTR& result_cloud_ptr,
                                    Eigen::Matrix4f& result_pose) {
        /*set input source cloud*/
        this->input_source_cloud = input_source;

        /*ICP main function*/
        this->IterativeClosestPoint(predict_pose, result_cloud_ptr, result_pose);
    
        return true;
    }

    void myICPRegistration::IterativeClosestPoint(const Eigen::Matrix4f& predict_pose, CloudData::CLOUD_PTR& result_cloud_ptr, Eigen::Matrix4f& result_pose){
        int iterations = 0;
	    double oldError = 0.0;
	    bool step = true;

        // Pre-process: apply predict R, t to input_source_cloud
        CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
        pcl::transformPointCloud(*this->input_source_cloud, *transformed_input_source, predict_pose);

        // init estimation:
        this->transformation_.setIdentity();

        // Convergence: apply ICP-SVD
        while (iterations < this->maxIterations) {
            /*step 0 : apply transformation*/
            CloudData::CLOUD_PTR curr_input_source(new CloudData::CLOUD());
            pcl::transformPointCloud(*transformed_input_source, *curr_input_source, this->transformation_);

            /*step 1 : Data association - for each point in the data set, find the nearest neighbor*/
            // get correspondent points into xs and ys:
            std::vector<Eigen::Vector3f> xs;
            std::vector<Eigen::Vector3f> ys;

            if(this->FindNearestNeighbor(curr_input_source, xs, ys) < 3){
                std::cout << "not enough neighbors" << std::endl;
                break;
            }

            /*step 2 : Data transformation - from R and T matrix to tranform data set close to model set*/
            Eigen::Matrix4f delta_transformation;
            delta_transformation.setIdentity();
            this->CalculateTransformationMatrix(xs, ys, delta_transformation);

            /*calculate the error*/
            // this->error = this->CalculateDistanceError(curr_input_source);

            /*add delta_transformaton into transformation*/
            this->transformation_ = delta_transformation * this->transformation_;

            iterations++;
	    }
    
        // set output:
        result_pose = this->transformation_ * predict_pose;
        pcl::transformPointCloud(*this->input_source_cloud, *result_cloud_ptr, result_pose);
    }

    size_t myICPRegistration::FindNearestNeighbor(
        const CloudData::CLOUD_PTR &input_source,
        std::vector<Eigen::Vector3f> &xs,
        std::vector<Eigen::Vector3f> &ys
    ) {	
        const int K = 1;
        const float MAX_CORR_DIST_SQR = this->max_corr_dist * this->max_corr_dist;
        size_t num_corr = 0;
    
        for (uint64_t i = 0; i < input_source->size(); i++) {        
                // Do a knn search
                std::vector<int> pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);
                
                this->kd_tree.nearestKSearch(input_source->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
                
                if(pointNKNSquaredDistance[0] > MAX_CORR_DIST_SQR)
                    continue;
                
                // Add correspondence:
                Eigen::Vector3f x(
                    this->input_target_cloud->at(pointIdxNKNSearch.at(0)).x,
                    this->input_target_cloud->at(pointIdxNKNSearch.at(0)).y,
                    this->input_target_cloud->at(pointIdxNKNSearch.at(0)).z
                );
                Eigen::Vector3f y(
                    input_source->at(i).x,
                    input_source->at(i).y,
                    input_source->at(i).z
                );

                xs.push_back(x);
                ys.push_back(y);

                num_corr++;
        }

        return num_corr;
    }

    void myICPRegistration::CalculateTransformationMatrix(
        const std::vector<Eigen::Vector3f> &xs, 
        const std::vector<Eigen::Vector3f> &ys,
        Eigen::Matrix4f &transformation_
        ) {
        pcl::PointXYZ center_srcCloud, center_targetCloud;
        Eigen::MatrixXf R = Eigen::MatrixXf::Zero(3,3);
        Eigen::MatrixXf T = Eigen::MatrixXf::Zero(3,1);  

        std::vector<pcl::PointXYZ> sourcePCL_minus_Center, targetPCL_minus_Center;
        Eigen::MatrixXf covariance;

        /*step 1 : find center of mass of both the datasets*/
        /*for source point cloud*/
        for (uint64_t i = 0; i < ys.size(); i++) {
            center_srcCloud.x += ys[i][0];
            center_srcCloud.y += ys[i][1];
            center_srcCloud.z += ys[i][2];
        }
        center_srcCloud.x = center_srcCloud.x * (1.0 / ys.size());
        center_srcCloud.y = center_srcCloud.y * (1.0 / ys.size());
        center_srcCloud.z = center_srcCloud.z * (1.0 / ys.size());
        /*for target point cloud*/
        for (uint64_t i = 0; i < xs.size(); i++) {
            center_targetCloud.x += xs[i][0];
            center_targetCloud.y += xs[i][1];
            center_targetCloud.z += xs[i][2];
        }
        center_targetCloud.x = center_targetCloud.x * (1.0 / xs.size());
        center_targetCloud.y = center_targetCloud.y * (1.0 / xs.size());
        center_targetCloud.z = center_targetCloud.z * (1.0 / xs.size());

        /*step 2 : center the point cloud as per the center of mass calculated*/
        /*for data point cloud*/
        for (uint64_t i = 0; i < ys.size(); i++) {
            pcl::PointXYZ pt;
            pt.x = ys[i][0] - center_srcCloud.x;
            pt.y = ys[i][1] - center_srcCloud.y;
            pt.z = ys[i][2] - center_srcCloud.z;
            sourcePCL_minus_Center.emplace_back(pt);
        }
        /*for nearest model point cloud*/
        for (uint64_t i = 0; i < xs.size(); i++) {
            pcl::PointXYZ pt;
            // pt = this->input_target_cloud->points[i] - center_targetCloud;
            pt.x = xs[i][0] - center_targetCloud.x;
            pt.y = xs[i][1] - center_targetCloud.y;
            pt.z = xs[i][2] - center_targetCloud.z;
            targetPCL_minus_Center.emplace_back(pt);
        }
        /*step 3 : calculate covariance matrix*/
        int vectorSize = xs.size();
        covariance = this->CalcualateICPCovarianceMtx(vectorSize, sourcePCL_minus_Center, targetPCL_minus_Center);

        #if SVD_REGISTRATION

            /*SVD registrayion method*/
            /*step 4: eigenvalues and eigenvectors from covariance matrix*/
            Eigen::MatrixXf U,V;
            Eigen::JacobiSVD<Eigen::MatrixXf> svd_solver(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV );
            V = svd_solver.matrixV();
            U = svd_solver.matrixU();

            /*step 5: calculate rotation matrix*/
            R = V * U.transpose();
            std::cout << "determinant of R: " << R.determinant() << std::endl;
            if (R.determinant() < 0.) {
                std::cout << "Reflection detected..." << std::endl;
                V(2, 0) *= -1.;
                V(2, 1) *= -1.;
                V(2, 2) *= -1.;
                R = V * U.transpose();
            }

            // std::cout << "Rotation matrix calculated" << std::endl;

        #endif

        /*calculate translation matrix*/
        Eigen::MatrixXf center_sourceCloud_matrix = Eigen::MatrixXf::Zero(3,1);
        Eigen::MatrixXf center_targetCloud_matrix = Eigen::MatrixXf::Zero(3,1);
        center_sourceCloud_matrix(0, 0) = center_srcCloud.x;
        center_sourceCloud_matrix(1, 0) = center_srcCloud.y;
        center_sourceCloud_matrix(2, 0) = center_srcCloud.z;
        center_targetCloud_matrix(0, 0) = center_targetCloud.x;
        center_targetCloud_matrix(1, 0) = center_targetCloud.y;
        center_targetCloud_matrix(2, 0) = center_targetCloud.z;
        
        T = center_targetCloud_matrix - (R * center_sourceCloud_matrix);

        transformation_.setIdentity();
        transformation_.block<3,3>(0,0) = R;
        transformation_.block<3,1>(0,3) = T;
    }

    /*calculate the euclideam difference between the transformed matrix and the */
    float myICPRegistration::CalculateDistanceError(const CloudData::CLOUD_PTR &curr_input_source) {
        float error = 0;
        
        // std::cout << "CalculateDistanceError size(curr_input_source):" << curr_input_source->size() << std::endl;
        for (uint64_t i = 0; i < curr_input_source->size(); i++) {
            error += pcl::geometry::squaredDistance(curr_input_source->points[i], this->input_target_cloud->points[i]);
        }
        error /= curr_input_source->size(); 
        std::cout << "Point Cloud error:" << error << std::endl;

        return error;
    }

    Eigen::MatrixXf myICPRegistration::CalcualateICPCovarianceMtx(int length, std::vector<pcl::PointXYZ>& centerPCL, std::vector<pcl::PointXYZ>& centerMCL)
    {
        Eigen::MatrixXf covariance = Eigen::MatrixXf::Zero(3,3);
        double sumXX = 0., sumXY = 0., sumXZ = 0.;
        double sumYX = 0., sumYY = 0., sumYZ = 0.;
        double sumZX = 0., sumZY = 0., sumZZ = 0.;
        for (int i = 0; i < length; i++) {
            // std::cout << this->nearestPts[i] << std::endl;
            sumXX += centerPCL[i].x * centerMCL[i].x;
            sumXY += centerPCL[i].x * centerMCL[i].y;
            sumXZ += centerPCL[i].x * centerMCL[i].z;
            sumYX += centerPCL[i].y * centerMCL[i].x;
            sumYY += centerPCL[i].y * centerMCL[i].y;
            sumYZ += centerPCL[i].y * centerMCL[i].z;
            sumZX += centerPCL[i].z * centerMCL[i].x;
            sumZY += centerPCL[i].z * centerMCL[i].y;
            sumZZ += centerPCL[i].z * centerMCL[i].z;
        }
        covariance(0, 0) = sumXX / length;
        covariance(0, 1) = sumXY / length;
        covariance(0, 2) = sumXZ / length;
        covariance(1, 0) = sumYX / length;
        covariance(1, 1) = sumYY / length;
        covariance(1, 2) = sumYZ / length;
        covariance(2, 0) = sumZX / length;
        covariance(2, 1) = sumZY / length;
        covariance(2, 2) = sumZZ / length;

        return covariance;
    }
}