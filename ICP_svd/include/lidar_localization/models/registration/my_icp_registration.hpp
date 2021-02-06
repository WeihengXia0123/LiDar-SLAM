/*
 * @Description: ICP (implementation)
 * @Author: Weiheng Xia
 * @Date: 2021-01-28 10:49:00
 */
#ifndef LIDAR_LOCALIZATION_MODELS_REGISTRATION_MY_ICP_REGISTRATION_HPP_
#define LIDAR_LOCALIZATION_MODELS_REGISTRATION_MY_ICP_REGISTRATION_HPP_

#include "lidar_localization/models/registration/registration_interface.hpp"
#include <ros/console.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "lidar_localization/nanoflann/nanoflann.hpp"

#define SVD_REGISTRATION	1			 /*Only one of these registration techniques */
#define QUAT_REGISTRATION	0			 /*should be activated at a time*/

namespace lidar_localization {
  class myICPRegistration: public RegistrationInterface {
    public:
      myICPRegistration(const YAML::Node& node);
      myICPRegistration(
        float min_threshold,
        int max_iter,
        float max_corr_dist
      );

      bool SetInputTarget(const CloudData::CLOUD_PTR& input_target) override;
      bool ScanMatch(const CloudData::CLOUD_PTR& input_source, 
                    const Eigen::Matrix4f& predict_pose, 
                    CloudData::CLOUD_PTR& result_cloud_ptr,
                    Eigen::Matrix4f& result_pose) override;
    
    private:
      bool SetRegistrationParam(
        float min_threshold,
        int max_iter,
        float max_corr_dist
      );

    private:
      /******************************************VARIABLES************************************/
      /*model and data point cloud variables*/
      pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;

      int maxIterations = 40;
      float minThreshold = 0.0001;
      float max_corr_dist = 1.0;
      float overlapParameter = 0.5;
      float error = std::numeric_limits<double>::max();

      /* point cloud and pose*/
      CloudData::CLOUD_PTR input_target_cloud;
      CloudData::CLOUD_PTR input_source_cloud;
      const Eigen::Matrix4f predict_pose;
      Eigen::Matrix4f transformation_;


      /******************************************FUNCTIONS************************************/
      void IterativeClosestPoint(const Eigen::Matrix4f& predict_pose, CloudData::CLOUD_PTR& result_cloud_ptr, Eigen::Matrix4f& result_pose);
      size_t FindNearestNeighbor(const CloudData::CLOUD_PTR &input_source, std::vector<Eigen::Vector3f> &xs, std::vector<Eigen::Vector3f> &ys);
      void CalculateTransformationMatrix(const std::vector<Eigen::Vector3f> &xs, const std::vector<Eigen::Vector3f> &ys, Eigen::Matrix4f &delta_transformation_);
      float CalculateDistanceError(const CloudData::CLOUD_PTR &curr_input_source);

  };
}

#endif