/*
 * @Description: my_ICP 匹配模块
 * @Author: Weiheng Xia
 * @Date: 2021-01-28 13:10:00
 */
#include "lidar_localization/models/registration/my_icp_registration.hpp"
#include <chrono>

namespace lidar_localization {

    myICPRegistration::myICPRegistration(const YAML::Node& node) {
        
        float min_threshold = node["min_threshold"].as<float>();
        int max_iter = node["max_iteration"].as<int>();

        SetRegistrationParam(min_threshold, max_iter);
    }

    myICPRegistration::myICPRegistration(float min_threshold, int max_iter) {
        SetRegistrationParam(min_threshold, max_iter);
    }

    bool myICPRegistration::SetRegistrationParam(
        float min_threshold,
        int max_iter
    ) {
        this->minThreshold = min_threshold;
        this->maxIterations = max_iter;
        
        std::cout << "minThreshold: " << this->minThreshold << std::endl
                << "max_iter: " << this->maxIterations << std::endl
                << std::endl;

        return true;
    }

    bool myICPRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {

        this->input_target_cloud = input_target;
        // std::vector<int> index;
        // pcl::removeNaNFromPointCloud ( *this->input_target_cloud, *this->input_target_cloud, index );

        this->LoadData(modelPCL, this->input_target_cloud);
        ROS_INFO("Load model Point Cloud Complete");

        return true;
    }

    bool myICPRegistration::ScanMatch(const CloudData::CLOUD_PTR& input_source, 
                                    const Eigen::Matrix4f& predict_pose, 
                                    CloudData::CLOUD_PTR& result_cloud_ptr,
                                    Eigen::Matrix4f& result_pose) {
        /*set input source cloud*/
        this->input_source_cloud = input_source;
        this->LoadData(this->dataPCL, this->input_source_cloud);
        std::cout << "size of the input source cloud: " << this->input_source_cloud->size() << std::endl;

        /*ICP main function*/
        this->IterativeClosestPoint(predict_pose, result_cloud_ptr, result_pose);

        // icp_ptr_->align(*result_cloud_ptr, predict_pose);
        // result_pose = icp_ptr_->getFinalTransformation();
    
        return true;
    }

    void myICPRegistration::LoadData(allPtCloud& points, const CloudData::CLOUD_PTR& input_cloud) {
        pcl::PointXYZ P;
        for(auto& input_point : *input_cloud) {
            /*Point coordinates - x, y, z*/
            P.x = input_point.x;
            P.y = input_point.y;
            P.z = input_point.z;
            points.pts.emplace_back(P);
        }
    }

    void myICPRegistration::IterativeClosestPoint(const Eigen::Matrix4f& predict_pose, CloudData::CLOUD_PTR& result_cloud_ptr, Eigen::Matrix4f& result_pose){
        int iterations = 1;
	    double oldError = 0.0;
	    bool step = true;

        // Initialization: apply predict R, t
        this->Rotation = predict_pose.block<3,3>(0,0);
        this->Translation = predict_pose.block<3,1>(0,3);
        // Transform and rotate the point cloud
        for (uint64_t i = 0; i < this->dataPCL.pts.size(); i++) {
            Eigen::MatrixXf dataPt = Eigen::MatrixXf::Zero(3, 1);
            dataPt(0, 0) = this->dataPCL.pts[i].x;
            dataPt(1, 0) = this->dataPCL.pts[i].y;
            dataPt(2, 0) = this->dataPCL.pts[i].z;
            dataPt = this->Rotation * dataPt;
            dataPt += this->Translation.cast<float>();
            this->dataPCL.pts[i].x = dataPt(0, 0);
            this->dataPCL.pts[i].y = dataPt(1, 0);
            this->dataPCL.pts[i].z = dataPt(2, 0);
        }
        std::cout << "Applied Initial Transformation" << std::endl;

        // Convergence: apply ICP-SVD
        while (step) {
            if ((iterations <= this->maxIterations) && !(abs(oldError - this->error) < this->minThreshold)) {
                std::cout << "****************iteration: " << iterations << " ****************************" << std::endl;
                /*step 2 : Data association - for each point in the data set, find the nearest neighbor*/
                this->FindNearestNeighbor();
                std::cout  << "Found nearest points with count:" << this->nearestPts.size() << std::endl;
                /*sort the nearestPts wrt the first index in ascending order because parallel loop
                can shuffle the indices */
                // sort(this->nearestPts.begin(), this->nearestPts.end());
                /*step 3 : Data transformation - from R and T matrix to tranform data set close to model set*/
                this->CalculateTransformationMatrix();

                /*apply transformations to the datapoint*/
                for (uint64_t i = 0; i < dataPCL.pts.size(); i++) {
                    Eigen::MatrixXf dataPt = Eigen::MatrixXf::Zero(3, 1);
                    dataPt(0, 0) = dataPCL.pts[i].x;
                    dataPt(1, 0) = dataPCL.pts[i].y;
                    dataPt(2, 0) = dataPCL.pts[i].z;
                    dataPt = this->Rotation * dataPt;
                    dataPt += this->Translation.cast<float>();
                    dataPCL.pts[i].x = dataPt(0, 0);
                    dataPCL.pts[i].y = dataPt(1, 0);
                    dataPCL.pts[i].z = dataPt(2, 0);
                }
                std::cout << "Difference in error: " << abs(oldError - this->error) << std::endl;
                oldError = this->error;
                /*calculate the error*/
                this->error = this->CalculateDistanceError();
                nearestPts.clear();
                iterations++;
                step = true;
            }
            if (abs(oldError - this->error) <= this->minThreshold) {
                std::cout << "*********************************************************\n";
                std::cout << "Converged with delta of error:" << std::abs(oldError - this->error) << std::endl;
                step = false;
                break;
            }
            if (iterations > this->maxIterations) {
                std::cout << "*********************************************************\n";
                std::cout << "Max iterations over, not converged" << std::endl;
                step = false;
                break;
            }
	    }
        // last step: Save result_pose and result_cloud_ptr
        std::cout << "assign result_pose " <<std::endl;
        result_pose.block<3,3>(0,0) = this->Rotation.cast<float>();
        result_pose.block<3,1>(0,3) = this->Translation.cast<float>();


        result_cloud_ptr->resize(this->input_source_cloud->size());

        std::cout << "assign result_cloud_ptr " <<std::endl;
        for (uint64_t i=0; i<dataPCL.pts.size(); i++){
            result_cloud_ptr->points[i].x = dataPCL.pts[i].x;
            result_cloud_ptr->points[i].y = dataPCL.pts[i].y;
            result_cloud_ptr->points[i].z = dataPCL.pts[i].z;
        }

        std::cout << "ICP function complete" <<std::endl;
    }

    /*Use nanoflann to find nearest neighbor*/
    /* https://github.com/jlblancoc/nanoflann/blob/master/examples/pointcloud_example.cpp */
    void myICPRegistration::FindNearestNeighbor() {	
        this->nearestPts.clear();
        this->squareDist.clear();
        this->trimmedPts.clear();
        //#pragma omp parallel for
        std::cout << "Size of the modelPCL: " << modelPCL.kdtree_get_point_count() << std::endl;
        myICPRegistration::myKDTree index(3, modelPCL, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        /*start timer*/
        auto start = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < dataPCL.pts.size(); i++) {        
                double  query_pt[3] = { dataPCL.pts[i].x, dataPCL.pts[i].y, dataPCL.pts[i].z };
                size_t ret_index;
                double out_dist_sqr;

                // do a knn search
                const size_t num_results = 1;
                nanoflann::KNNResultSet<double> resultSet(num_results);
                resultSet.init(&ret_index, &out_dist_sqr);
                index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
                //std::cout << "knnSearch(nn=" << num_results << "): \n";
                //std::cout << "i="  << i <<  " ret_index=" << ret_index << " out_dist_sqr=" << out_dist_sqr << endl;

                std::pair<uint16_t, size_t> nearComb;
                std::pair<double, uint16_t> trim;
                nearComb.first = i;
                nearComb.second = ret_index;
                
                /*nearest points needed for ICP*/
                this->nearestPts.push_back(nearComb);
        }
        /*end the timer*/
        auto finish = std::chrono::high_resolution_clock::now();    
        /*calculate the time needed for the whole process*/
        std::chrono::duration<double> elapsed = (finish - start);
        std::cout << "Time of FindNearestNeighbor " << elapsed.count() << " seconds" << std::endl;


    }

    void myICPRegistration::CalculateTransformationMatrix() {
        pcl::PointXYZ pclCOM, mclCOM;
        Eigen::MatrixXf R(3,3);
        Eigen::MatrixXf T(3,1);  
        R = Eigen::MatrixXf::Zero(3,3);
        T = Eigen::MatrixXf::Zero(3,1);

        std::vector<pcl::PointXYZ> centerPCL, centerMCL;
        int vectorSize = 0;
        Eigen::MatrixXf covariance;

        /*step 1 : find center of mass of both the datasets*/
        /*for data point cloud*/
        for (uint64_t i = 0; i < dataPCL.pts.size(); i++) {
            // pclCOM += dataPCL.pts[i];
            pclCOM.x += dataPCL.pts[i].x;
            pclCOM.y += dataPCL.pts[i].y;
            pclCOM.z += dataPCL.pts[i].z;
        }
        pclCOM.x = pclCOM.x * (1.0 / dataPCL.pts.size());
        pclCOM.y = pclCOM.y * (1.0 / dataPCL.pts.size());
        pclCOM.z = pclCOM.z * (1.0 / dataPCL.pts.size());
        /*for nearest model point clouds*/
        for (uint64_t i = 0; i < modelPCL.pts.size(); i++) {
            // mclCOM += modelPCL.pts[i];
            mclCOM.x += modelPCL.pts[i].x;
            mclCOM.y += modelPCL.pts[i].y;
            mclCOM.z += modelPCL.pts[i].z;
        }
        mclCOM.x = mclCOM.x * (1.0 / modelPCL.pts.size());
        mclCOM.y = mclCOM.y * (1.0 / modelPCL.pts.size());
        mclCOM.z = mclCOM.z * (1.0 / modelPCL.pts.size());
        std::cout << "pclCOM: " << pclCOM << " mclCOM: " << mclCOM << std::endl;
        /*step 2 : center the point cloud as per the center of mass calculated*/
        /*for data point cloud*/
        for (uint64_t i = 0; i < dataPCL.pts.size(); i++) {
            pcl::PointXYZ pt;
            pt.x = dataPCL.pts[i].x - pclCOM.x;
            pt.y = dataPCL.pts[i].y - pclCOM.y;
            pt.z = dataPCL.pts[i].z - pclCOM.z;
            centerPCL.emplace_back(pt);
        }
        /*for nearest model point cloud*/
        for (uint64_t i = 0; i < modelPCL.pts.size(); i++) {
            pcl::PointXYZ pt;
            // pt = modelPCL.pts[i] - mclCOM;
            pt.x = modelPCL.pts[i].x - mclCOM.x;
            pt.y = modelPCL.pts[i].y - mclCOM.y;
            pt.z = modelPCL.pts[i].z - mclCOM.z;
            centerMCL.emplace_back(pt);
        }
        /*step 3 : calculate covariance matrix*/
        vectorSize = nearestPts.size();
        covariance = this->CalcualateICPCovarianceMtx(vectorSize, centerPCL, centerMCL);


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

            ROS_INFO("Rotation matrix calculated");

        #endif

        /*calculate translation matrix*/
        Eigen::MatrixXf matPclCOM, matMclCOM;
        matPclCOM = Eigen::MatrixXf::Zero(3,1);
        matMclCOM = Eigen::MatrixXf::Zero(3,1);
        matPclCOM(0, 0) = pclCOM.x;
        matPclCOM(1, 0) = pclCOM.y;
        matPclCOM(2, 0) = pclCOM.z;

        matMclCOM(0, 0) = mclCOM.x;
        matMclCOM(1, 0) = mclCOM.y;
        matMclCOM(2, 0) = mclCOM.z;

        T = matMclCOM - (R * matPclCOM);
        // cout << "Translation matrix is:" << endl;
        // for (int r = 0; r < T.rows; r++) {
        //     for (int c = 0; c < T.cols; c++) {
        //         cout << T.at<double>(r, c) << " ";
        //     }
        // }
        
        this->Rotation = R;
        this->Translation = T;
    }

    Eigen::MatrixXf myICPRegistration::CalcualateICPCovarianceMtx(int length, std::vector<pcl::PointXYZ>& centerPCL, std::vector<pcl::PointXYZ>& centerMCL)
    {
        Eigen::MatrixXf covariance = Eigen::MatrixXf::Zero(3,3);
        double sumXX = 0., sumXY = 0., sumXZ = 0.;
        double sumYX = 0., sumYY = 0., sumYZ = 0.;
        double sumZX = 0., sumZY = 0., sumZZ = 0.;
        for (int i = 0; i < length; i++) {
            // std::cout << this->nearestPts[i] << std::endl;
            sumXX += centerPCL[i].x * centerMCL[this->nearestPts[i].second].x;
            sumXY += centerPCL[i].x * centerMCL[this->nearestPts[i].second].y;
            sumXZ += centerPCL[i].x * centerMCL[this->nearestPts[i].second].z;
            sumYX += centerPCL[i].y * centerMCL[this->nearestPts[i].second].x;
            sumYY += centerPCL[i].y * centerMCL[this->nearestPts[i].second].y;
            sumYZ += centerPCL[i].y * centerMCL[this->nearestPts[i].second].z;
            sumZX += centerPCL[i].z * centerMCL[this->nearestPts[i].second].x;
            sumZY += centerPCL[i].z * centerMCL[this->nearestPts[i].second].y;
            sumZZ += centerPCL[i].z * centerMCL[this->nearestPts[i].second].z;
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
    /*calculate the euclideam difference between the transformed matrix and the */
    double myICPRegistration::CalculateDistanceError() {
        double error = 0;
        for (uint64_t i = 0; i < dataPCL.pts.size(); i++) {
            Eigen::MatrixXf dataPt = Eigen::MatrixXf::Zero(3, 1);
            Eigen::MatrixXf modelPt = Eigen::MatrixXf::Zero(3, 1);
            dataPt(0, 0) = dataPCL.pts[i].x;
            dataPt(1, 0) = dataPCL.pts[i].y;
            dataPt(2, 0) = dataPCL.pts[i].z;
            modelPt(0, 0) = modelPCL.pts[i].x;
            modelPt(1, 0) = modelPCL.pts[i].y;
            modelPt(2, 0) = modelPCL.pts[i].z;
            error += ((this->Rotation * dataPt + this->Translation) - modelPt).norm();
            // std::cout << (this->Rotation * dataPt + this->Translation) - modelPt<< std::endl;
        }
        error /= dataPCL.pts.size();
        std::cout << "error between transformed PCL and the model PCL:" << error << std::endl;
        return error;
    }
}