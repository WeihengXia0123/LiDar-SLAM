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

#include "lidar_localization/nanoflann/nanoflann.hpp"

#define SVD_REGISTRATION	1			 /*Only one of these registration techniques */
#define QUAT_REGISTRATION	0			 /*should be activated at a time*/

namespace lidar_localization {
  class myICPRegistration: public RegistrationInterface {
    public:
      myICPRegistration(const YAML::Node& node);
      myICPRegistration(
        float min_threshold,
        int max_iter
      );

      bool SetInputTarget(const CloudData::CLOUD_PTR& input_target) override;
      bool ScanMatch(const CloudData::CLOUD_PTR& input_source, 
                    const Eigen::Matrix4f& predict_pose, 
                    CloudData::CLOUD_PTR& result_cloud_ptr,
                    Eigen::Matrix4f& result_pose) override;
    
    private:
      bool SetRegistrationParam(
        float min_threshold,
        int max_iter
      );

    private:
      /******************************************DATA STRUCTURES************************************/
      /*point cloud structure as needed by nanoflann*/
      typedef struct PointCloud
      {
        std::vector<pcl::PointXYZ>  pts;

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return pts.size(); }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline double kdtree_get_pt(const size_t idx, const size_t dim) const
        {
          if (dim == 0) return pts[idx].x;
          else if (dim == 1) return pts[idx].y;
          else return pts[idx].z;
        }

        // // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        // //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        // //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

      }allPtCloud;

      /*model and data point cloud variables*/
      allPtCloud modelPCL, dataPCL;
      typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<double, allPtCloud>, allPtCloud, 3> myKDTree;

      /*****************************************VARIABLES***********************************************/
      std::vector<std::pair<uint, size_t>> nearestPts;
      std::vector<std::pair<double, uint16_t>> squareDist;
      std::vector<std::pair<double, uint16_t>> trimmedPts;

      int maxIterations = 40;
      double minThreshold = 0.0001;
      double overlapParameter = 0.5;
      double error = std::numeric_limits<double>::max();
      Eigen::MatrixXf Rotation = Eigen::MatrixXf::Zero(3,3);
      Eigen::MatrixXf Translation = Eigen::MatrixXf::Zero(3,1);

      /* point cloud and pose*/
      CloudData::CLOUD_PTR input_target_cloud;
      CloudData::CLOUD_PTR input_source_cloud;
      CloudData::CLOUD_PTR result_cloud_ptr;

      const Eigen::Matrix4f predict_pose;
      Eigen::Matrix4f result_pose = Eigen::Matrix4f::Zero();

      /******************************************FUNCTIONS************************************/
      void LoadData(allPtCloud& points, const CloudData::CLOUD_PTR& input_cloud);
      void IterativeClosestPoint(const Eigen::Matrix4f& predict_pose, CloudData::CLOUD_PTR& result_cloud_ptr, Eigen::Matrix4f& result_pose);
      void FindNearestNeighbor();
      void CalculateTransformationMatrix();
      Eigen::MatrixXf CalcualateICPCovarianceMtx(int length, std::vector<pcl::PointXYZ>& centerPCL, std::vector<pcl::PointXYZ>& centerMCL);
      double CalculateDistanceError();

  };
}

#endif